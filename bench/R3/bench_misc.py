"""(d) lm_head, (e) depthwise conv1d (sconv), (f) attention projections, (g) dense MLP.

All at Inkling-Small shapes, batch 1 decode unless stated. Quantized weights
are random packed bit patterns (timing-exact).
"""

import mlx.core as mx
import mlx.nn as nn

from benchutil import Recorder, affine_bytes, qmatrix

H, V = 4096, 201024
GS = 64
rec = Recorder("misc")

x = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
mx.eval(x)


def qlinear(n_out, n_in, bits):
    w, s, b = qmatrix(n_out, n_in, bits)
    return lambda inp: mx.quantized_matmul(inp, w, s, b, transpose=True, group_size=GS, bits=bits)


# ---------------- (d) lm_head ----------------
rec.note(section="lm_head 4096 -> 201024")
w16 = mx.random.normal((V, H)).astype(mx.bfloat16)
mx.eval(w16)
rec.row("lm_head bf16 (as shipped in mlx-community checkpoints)", lambda: x @ w16.T, nbytes=V * H * 2)
del w16
mx.clear_cache()
for bits in (8, 6, 4):
    f = qlinear(V, H, bits)
    rec.row(f"lm_head {bits}-bit affine g64", lambda f=f: f(x), nbytes=affine_bytes(V, H, bits), bits=bits)
    del f
    mx.clear_cache()

# ---------------- (f) attention projections per layer ----------------
rec.note(section="attention projections per layer (q 4096->4096, k/v 4096->1024, r 4096->512, o 4096->4096)")
projs = [("wq_du", H, H), ("wk_dv", 1024, H), ("wv_dv", 1024, H), ("wr_du", 512, H), ("wo_ud", H, H)]
for kind in ("bf16", 8, 4):
    fns = []
    nbytes = 0
    for name, n_out, n_in in projs:
        if kind == "bf16":
            w = mx.random.normal((n_out, n_in)).astype(mx.bfloat16)
            mx.eval(w)
            fns.append(lambda inp, w=w: inp @ w.T)
            nbytes += n_out * n_in * 2
        else:
            fns.append(qlinear(n_out, n_in, kind))
            nbytes += affine_bytes(n_out, n_in, kind)

    def all_projs(fns=fns):
        return [f(x) for f in fns]

    r = rec.row(f"attention 5 projections {kind}", all_projs, nbytes=nbytes, kind=str(kind))
    rec.note(label=f"attention projections {kind} x42 layers", ms_per_token=round(r["med_ms_r8"] * 42, 3))
    del fns
    mx.clear_cache()

# ---------------- (g) dense MLP (layers 0,1): 4096 -> 16384 x2, 16384 -> 4096 ----------------
rec.note(section="dense MLP layers 0-1")
for bits in (8, 4):
    up, gate, down = qlinear(16384, H, bits), qlinear(16384, H, bits), qlinear(H, 16384, bits)
    rec.row(f"dense GLU 4096->16384->4096 {bits}-bit", lambda: down(nn.silu(gate(x)) * up(x)),
            nbytes=2 * affine_bytes(16384, H, bits) + affine_bytes(H, 16384, bits), bits=bits)
    del up, gate, down
    mx.clear_cache()

# ---------------- (e) depthwise conv1d, kernel 4, 4096 channels ----------------
rec.note(section="sconv depthwise conv1d k=4 C=4096")
for dt in (mx.float32, mx.bfloat16):
    w = mx.random.normal((H, 4, 1)).astype(dt)
    state = mx.random.normal((1, 3, H)).astype(dt)
    x1 = mx.random.normal((1, 1, H)).astype(dt)
    mx.eval(w, state, x1)

    def sconv_decode():
        xin = mx.concatenate([state, x1], axis=1)
        out = mx.conv1d(xin, w, padding=0, groups=H)
        return out + x1, xin[:, -3:, :]

    rec.row(f"sconv decode T=1 conv1d groups=C padding=0 (+state concat/slice) {dt}", sconv_decode,
            nbytes=(4 + 1 + 3) * H * dt.size + H * 4 * dt.size)

    def sconv_decode_elementwise():
        xin = mx.concatenate([state, x1], axis=1)  # [1,4,H]
        out = (xin * w.reshape(1, 4, H)).sum(axis=1, keepdims=True)
        return out + x1, xin[:, -3:, :]

    rec.row(f"sconv decode T=1 elementwise mul+sum {dt}", sconv_decode_elementwise,
            nbytes=(4 + 1 + 3) * H * dt.size + H * 4 * dt.size)

    T = 2048
    xT = mx.random.normal((1, T, H)).astype(dt)
    mx.eval(xT)

    def sconv_prefill_fast():
        xin = mx.concatenate([mx.zeros((1, 3, H), dt), xT], axis=1)
        return mx.conv1d(xin, w, padding=0, groups=H) + xT

    rec.row(f"sconv prefill T={T} conv1d padding=0 after explicit pad (depthwise fast path) {dt}",
            sconv_prefill_fast, nbytes=2 * T * H * dt.size, reps=(1, 4), iters=10)

    def sconv_prefill_pad():
        return mx.conv1d(xT, w, padding=3, groups=H)[:, :T, :] + xT

    rec.row(f"sconv prefill T={T} conv1d padding=3 (vendored no-cache path) {dt}", sconv_prefill_pad,
            nbytes=2 * T * H * dt.size, reps=(1, 4), iters=10)

    def sconv_prefill_elementwise():
        xin = mx.concatenate([mx.zeros((1, 3, H), dt), xT], axis=1)
        acc = xin[:, 0:T, :] * w[:, 0, 0]
        for i in range(1, 4):
            acc = acc + xin[:, i:i + T, :] * w[:, i, 0]
        return acc + xT

    rec.row(f"sconv prefill T={T} elementwise 4-tap {dt}", sconv_prefill_elementwise,
            nbytes=2 * T * H * dt.size, reps=(1, 4), iters=10)
    del w, state, x1, xT
    mx.clear_cache()

# ---------------- norms and small ops per layer (launch-overhead floor) ----------------
rec.note(section="small per-layer ops")
nw = mx.ones((H,))
rec.row("rms_norm 4096 (one call)", lambda: mx.fast.rms_norm(x, nw, 1e-6), nbytes=2 * H * 2)
qh = mx.random.normal((1, 1, 32, 128)).astype(mx.bfloat16)
hw = mx.ones((128,))
mx.eval(qh, hw)
rec.row("rms_norm per-head [1,1,32,128]", lambda: mx.fast.rms_norm(qh, hw, 1e-6), nbytes=2 * 32 * 128 * 2)
rec.row("add residual [1,1,4096]", lambda: x + x, nbytes=3 * H * 2)

# per-op dispatch floor in a long dependent graph (what a 42-layer forward pays per op)
rec.note(section="per-op floor: 200 dependent tiny ops in one eval")


def chain_rms(n=200):
    h = x
    for _ in range(n):
        h = mx.fast.rms_norm(h, nw, 1e-6)
    return h


def chain_matvec(n=200, w=mx.random.normal((H, H)).astype(mx.bfloat16)):
    h = x
    for _ in range(n):
        h = h @ w.T
    return h


def chain_elementwise(n=200):
    h = x
    for _ in range(n):
        h = h * 1.01 + 0.5
    return h


chain_elementwise_compiled = mx.compile(chain_elementwise)
for label, f in (("rms_norm", chain_rms), ("matvec 4096x4096 bf16 (32 MB each)", chain_matvec),
                 ("elementwise mul+add", chain_elementwise), ("elementwise mul+add mx.compile", chain_elementwise_compiled)):
    r = rec.row(f"chain of 200 dependent {label}", f, reps=(1,), iters=15)
    rec.note(label=f"per-op us in chain ({label})", us=round(r["med_ms_r1"] * 1e3 / 200, 1), best_us=round(r["best_ms_r1"] * 1e3 / 200, 1))
