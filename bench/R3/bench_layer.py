"""Whole-layer emulation of Inkling-Small at decode (B=1, L=1) and prefill (L=2048).

One decoder layer is built from random weights at the exact shapes of
Inkling-Small (hidden 4096, 32 q / 8 kv heads x 128, d_rel 16, sconv k=4 on
k, v, post-attention and post-MLP, router [258, 4096] fp32, 6-of-256 routed
experts + 2 shared experts of intermediate 2048), then chained 42 times
(two distinct weight sets alternate so no layer's weights stay in the SLC)
plus the unembed, to estimate a per-token budget. Variants:

  attn=bf16|8   attention/unembed weights bf16 (as the mlx-community
                checkpoints ship) or 8-bit affine
  experts=2|3|4 routed + shared expert bits
  compile       wrap the per-layer decode function in mx.compile
  mask=vendored|cheap  rel-bias additive mask built like inkling_mlx
                (matmul + take_along_axis + where) or a single gather

Sliding layers (35 of 42) attend to a 512-slot rotating buffer; global
layers (7) attend to a CTX-key cache (default 4096). Prefill uses one 2048
token chunk with no cache and the vendored full [1,32,L,L] additive mask.
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from benchutil import Recorder, qmatrix, timeit

H, I, E, K, NQ, NKV, D, DREL, V = 4096, 2048, 256, 6, 32, 8, 128, 16, 201024
GS = 64
WIN, RELG = 512, 1024
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41}

p = argparse.ArgumentParser()
p.add_argument("--ctx", type=int, default=4096)
p.add_argument("--experts", type=int, nargs="+", default=[3])
p.add_argument("--attn", nargs="+", default=["bf16", "8"])
p.add_argument("--prefill", type=int, default=2048)
p.add_argument("--iters", type=int, default=15)
args = p.parse_args()
rec = Recorder("layer")


class QL:
    """quantized_matmul closure with the packed layout MLX expects."""

    def __init__(self, n_out, n_in, bits):
        self.w, self.s, self.b = qmatrix(n_out, n_in, bits)
        self.bits = bits
        self.nbytes = self.w.nbytes + self.s.nbytes + self.b.nbytes

    def __call__(self, x):
        return mx.quantized_matmul(x, self.w, self.s, self.b, transpose=True, group_size=GS, bits=self.bits)


class BL:
    def __init__(self, n_out, n_in):
        self.w = mx.random.normal((n_out, n_in)).astype(mx.bfloat16)
        mx.eval(self.w)
        self.nbytes = self.w.nbytes

    def __call__(self, x):
        return x @ self.w.T


class GQ:
    def __init__(self, n_out, n_in, bits, n_exp):
        self.w, self.s, self.b = qmatrix(n_out, n_in, bits, lead=(n_exp,))
        self.bits = bits
        self.nbytes = self.w.nbytes + self.s.nbytes + self.b.nbytes

    def __call__(self, x, idx, sorted_indices=False):
        return mx.gather_qmm(x, self.w, self.s, self.b, rhs_indices=idx, transpose=True,
                             group_size=GS, bits=self.bits, sorted_indices=sorted_indices)


def make_layer(attn, ebits, is_global):
    lin = (lambda o, i: BL(o, i)) if attn == "bf16" else (lambda o, i: QL(o, i, int(attn)))
    L = dict(
        wq=lin(NQ * D, H), wk=lin(NKV * D, H), wv=lin(NKV * D, H), wr=lin(NQ * DREL, H), wo=lin(H, NQ * D),
        attn_norm=mx.ones((H,)), mlp_norm=mx.ones((H,)), qn=mx.ones((D,)), kn=mx.ones((D,)),
        ksc=mx.random.normal((NKV * D, 4, 1)), vsc=mx.random.normal((NKV * D, 4, 1)),
        asc=mx.random.normal((H, 4, 1)), msc=mx.random.normal((H, 4, 1)),
        proj=mx.random.normal((DREL, RELG if is_global else WIN)).astype(mx.bfloat16),
        router_w=mx.random.normal((E + 2, H)), router_b=mx.random.normal((E,)) * 0.01,
        up=GQ(I, H, ebits, E), gate=GQ(I, H, ebits, E), down=GQ(H, I, ebits, E),
        sup=GQ(I, H, ebits, 2), sgate=GQ(I, H, ebits, 2), sdown=GQ(H, I, ebits, 2),
        is_global=is_global,
    )
    mx.eval([v for v in L.values() if isinstance(v, mx.array)])
    return L


def sconv(x, w, state):
    """Vendored semantics: fp32, state = last 3 inputs, residual add. Returns (y, new_state)."""
    xf = x.astype(mx.float32)
    xin = mx.concatenate([state, xf], axis=1)
    y = mx.conv1d(xin, w, padding=0, groups=w.shape[0]) + xf
    return y.astype(x.dtype), xin[:, -3:, :]


def router(h, L):
    flat = h.reshape(-1, H).astype(mx.float32)
    logits = flat @ L["router_w"].T
    scores = mx.sigmoid(logits)[:, :E] + L["router_b"]
    top = mx.argpartition(-scores, kth=K - 1, axis=-1)[:, :K]
    sel = mx.concatenate([mx.take_along_axis(logits[:, :E], top, axis=-1), logits[:, E:]], axis=-1)
    w = mx.softmax(-mx.logaddexp(mx.zeros_like(sel), -sel), axis=-1) * 8.0
    return top, w[:, :K].astype(h.dtype), w[:, K:].astype(h.dtype)


def moe(h, L, T):
    top, tw, gam = router(h, L)
    x = h.reshape(T, 1, 1, H)
    if T * K >= 64:  # mlx-lm sorted path
        flat = top.flatten()
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        xs, ids = x.flatten(0, -3)[order // K], flat[order]
        u, g = L["up"](xs, ids, True), L["gate"](xs, ids, True)
        y = L["down"](nn.silu(g) * u, ids, True)
        y = mx.unflatten(y[inv], 0, (T, K))
    else:
        u, g = L["up"](x, top), L["gate"](x, top)
        y = L["down"](nn.silu(g) * u, top)
    routed = (y.squeeze(-2) * tw[..., None]).sum(axis=1)
    sidx = mx.broadcast_to(mx.array([[0, 1]]), (T, 2))
    su, sg = L["sup"](x, sidx), L["sgate"](x, sidx)
    sy = L["sdown"](nn.silu(sg) * su, sidx).squeeze(-2)
    shared = (sy.astype(mx.float32) * gam[..., None].astype(mx.float32)).sum(axis=1).astype(h.dtype)
    return (routed + shared).reshape(1, T, H)


def decode_layer(L, x, st, kbuf, vbuf, pos, mask_mode):
    """x [1,1,H]; st = (k_state, v_state, a_state, m_state); kbuf/vbuf [1,8,S,128]."""
    S = kbuf.shape[2]
    h = mx.fast.rms_norm(x, L["attn_norm"], 1e-6)
    q, k, v, r = L["wq"](h), L["wk"](h), L["wv"](h), L["wr"](h)
    k, ks = sconv(k, L["ksc"], st[0])
    v, vs = sconv(v, L["vsc"], st[1])
    q = mx.fast.rms_norm(q.reshape(1, 1, NQ, D), L["qn"], 1e-6).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, 1, NKV, D), L["kn"], 1e-6).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, NKV, D).transpose(0, 2, 1, 3)
    slot = pos % S
    kbuf = kbuf.at[..., slot:slot + 1, :].add(k - kbuf[..., slot:slot + 1, :])  # in-place style write
    vbuf = vbuf.at[..., slot:slot + 1, :].add(v - vbuf[..., slot:slot + 1, :])
    # relative bias over the S slots: distance = pos - slot_position
    rel_logits = mx.swapaxes(r.reshape(1, 1, NQ, DREL) @ L["proj"], 1, 2)  # [1,32,1,R]
    R = L["proj"].shape[1]
    slot_pos = pos - ((slot - mx.arange(S)) % S)  # position held by each slot
    dist = pos - slot_pos
    if mask_mode == "vendored":
        gather = mx.broadcast_to(mx.clip(dist, 0, R - 1).reshape(1, 1, 1, S), (1, NQ, 1, S))
        bias = mx.take_along_axis(rel_logits, gather, axis=-1)
        bias = mx.where((dist < R).reshape(1, 1, 1, S), bias, mx.array(0.0, mx.bfloat16))
    else:  # cheap: one gather from a zero-padded table, no where
        table = mx.concatenate([rel_logits, mx.zeros((1, NQ, 1, 1), mx.bfloat16)], axis=-1)
        gather = mx.broadcast_to(mx.minimum(dist, R).reshape(1, 1, 1, S), (1, NQ, 1, S))
        bias = mx.take_along_axis(table, gather, axis=-1)
    o = mx.fast.scaled_dot_product_attention(q, kbuf, vbuf, scale=1.0 / D, mask=bias)
    o = L["wo"](o.transpose(0, 2, 1, 3).reshape(1, 1, NQ * D))
    o, as_ = sconv(o, L["asc"], st[2])
    x = x + o
    h = mx.fast.rms_norm(x, L["mlp_norm"], 1e-6)
    m = moe(h, L, 1)
    m, ms = sconv(m, L["msc"], st[3])
    return x + m, (ks, vs, as_, ms), kbuf, vbuf


def prefill_layer(L, x):
    """x [1,T,H], no cache, vendored full additive mask."""
    T = x.shape[1]
    h = mx.fast.rms_norm(x, L["attn_norm"], 1e-6)
    q, k, v, r = L["wq"](h), L["wk"](h), L["wv"](h), L["wr"](h)
    z = mx.zeros((1, 3, NKV * D), mx.float32)
    k, _ = sconv(k, L["ksc"], z)
    v, _ = sconv(v, L["vsc"], z)
    q = mx.fast.rms_norm(q.reshape(1, T, NQ, D), L["qn"], 1e-6).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, T, NKV, D), L["kn"], 1e-6).transpose(0, 2, 1, 3)
    v = v.reshape(1, T, NKV, D).transpose(0, 2, 1, 3)
    rel_logits = mx.swapaxes(r.reshape(1, T, NQ, DREL) @ L["proj"], 1, 2)  # [1,32,T,R]
    R = L["proj"].shape[1]
    li, ri = mx.arange(T)[:, None], mx.arange(T)[None]
    dist = li - ri
    gather = mx.broadcast_to(mx.clip(dist, 0, R - 1)[None, None], (1, NQ, T, T))
    bias = mx.take_along_axis(rel_logits, gather, axis=-1)
    allowed = (dist >= 0) & (dist < R) if not L["is_global"] else (dist >= 0)
    bias = mx.where(allowed[None, None], bias, mx.array(0.0, mx.bfloat16))
    bias = mx.where((dist >= 0)[None, None] & (allowed[None, None] if not L["is_global"] else True),
                    bias, mx.array(-1e30, mx.bfloat16))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / D, mask=bias)
    o = L["wo"](o.transpose(0, 2, 1, 3).reshape(1, T, NQ * D))
    z = mx.zeros((1, 3, H), mx.float32)
    o, _ = sconv(o, L["asc"], z)
    x = x + o
    h = mx.fast.rms_norm(x, L["mlp_norm"], 1e-6)
    m = moe(h, L, T)
    m, _ = sconv(m, L["msc"], z)
    return x + m


for ebits in args.experts:
    for attn in args.attn:
        rec.note(section=f"layer emulation experts={ebits}b attn={attn}")
        Lg = [make_layer(attn, ebits, True), make_layer(attn, ebits, True)]
        Ls = [make_layer(attn, ebits, False), make_layer(attn, ebits, False)]
        head = BL(V, H) if attn == "bf16" else QL(V, H, int(attn))
        bytes_layer_s = sum(getattr(v, "nbytes", 0) for v in Ls[0].values() if not isinstance(v, mx.array))
        per_tok_bytes = (35 * (Ls[0]["wq"].nbytes + Ls[0]["wk"].nbytes + Ls[0]["wv"].nbytes + Ls[0]["wr"].nbytes + Ls[0]["wo"].nbytes)
                         + 42 * K * (Ls[0]["up"].nbytes + Ls[0]["gate"].nbytes + Ls[0]["down"].nbytes) // E
                         + 42 * (Ls[0]["sup"].nbytes + Ls[0]["sgate"].nbytes + Ls[0]["sdown"].nbytes)
                         + 7 * (Lg[0]["wq"].nbytes + Lg[0]["wk"].nbytes + Lg[0]["wv"].nbytes + Lg[0]["wr"].nbytes + Lg[0]["wo"].nbytes)
                         + head.nbytes)
        rec.note(label="weight bytes read per decoded token (est.)", gb=round(per_tok_bytes / 1e9, 2),
                 experts=ebits, attn=attn)

        x = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
        st = tuple(mx.zeros((1, 3, c), mx.float32) for c in (NKV * D, NKV * D, H, H))
        kb_s = mx.random.normal((1, NKV, WIN, D)).astype(mx.bfloat16)
        vb_s = mx.random.normal((1, NKV, WIN, D)).astype(mx.bfloat16)
        kb_g = mx.random.normal((1, NKV, args.ctx, D)).astype(mx.bfloat16)
        vb_g = mx.random.normal((1, NKV, args.ctx, D)).astype(mx.bfloat16)
        mx.eval(x, st, kb_s, vb_s, kb_g, vb_g)
        pos = args.ctx + 100

        for mask_mode in ("vendored", "cheap"):
            for use_compile in (False, True):
                fns = {}
                for name, L in (("g0", Lg[0]), ("g1", Lg[1]), ("s0", Ls[0]), ("s1", Ls[1])):
                    f = (lambda L: (lambda x, st, kb, vb: decode_layer(L, x, st, kb, vb, pos, mask_mode)))(L)
                    fns[name] = mx.compile(f) if use_compile else f

                def step():
                    h = x
                    s = st
                    outs = []
                    for i in range(42):
                        g = i in GLOBAL_LAYERS
                        f = fns[("g" if g else "s") + str(i % 2)]
                        h, s, kb, vb = f(h, s, kb_g if g else kb_s, vb_g if g else vb_s)
                        outs.append(kb)
                        outs.append(vb)
                    logits = head(mx.fast.rms_norm(h, Ls[0]["mlp_norm"], 1e-6) / 16.0)
                    return logits, s, outs

                label = f"decode 42 layers + head experts={ebits}b attn={attn} mask={mask_mode} compile={use_compile} ctx={args.ctx}"
                r = rec.row(label, step, nbytes=per_tok_bytes, reps=(1,), iters=args.iters, experts=ebits, attn=attn,
                            mask=mask_mode, compile=use_compile)
                rec.note(label="tok/s", value=round(1e3 / r["med_ms_r1"], 1), best=round(1e3 / r["best_ms_r1"], 1),
                         experts=ebits, attn=attn, mask=mask_mode, compile=use_compile)

        # single-layer breakdown (sliding layer, vendored mask, no compile)
        f = lambda: decode_layer(Ls[0], x, st, kb_s, vb_s, pos, "vendored")
        rec.row(f"decode one sliding layer experts={ebits}b attn={attn}", f, reps=(1, 8), iters=20)
        f = lambda: decode_layer(Lg[0], x, st, kb_g, vb_g, pos, "vendored")
        rec.row(f"decode one global layer ctx={args.ctx} experts={ebits}b attn={attn}", f, reps=(1, 8), iters=20)

        # prefill chunk
        T = args.prefill
        xT = mx.random.normal((1, T, H)).astype(mx.bfloat16)
        mx.eval(xT)
        r = rec.row(f"prefill one sliding layer T={T} experts={ebits}b attn={attn} (vendored mask)",
                    lambda: prefill_layer(Ls[0], xT), reps=(1,), iters=5)
        r2 = rec.row(f"prefill one global layer T={T} experts={ebits}b attn={attn} (vendored mask)",
                     lambda: prefill_layer(Lg[0], xT), reps=(1,), iters=5)
        est = 35 * r["med_ms_r1"] + 7 * r2["med_ms_r1"]
        rec.note(label=f"prefill 42-layer estimate T={T}", ms=round(est, 1), tok_per_s=round(T / (est * 1e-3), 0),
                 experts=ebits, attn=attn)
        del Lg, Ls, head, xT
        mx.clear_cache()
