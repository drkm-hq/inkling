#!/usr/bin/env python
"""Isolated cost of each custom Metal kernel in mlx_vlm/models/inkling/language.py
and of the MLX primitives they feed, at Inkling-Small shapes; plus the pure
Python/dispatch overhead of one decode step measured on a tiny-shape model with
the same op graph.

Kernels (file:line in language.py @ 138d761):
  inkling_banded_mask (v1, L93)       - only used when q_offset != S-LQ (never in normal decode/prefill)
  inkling_banded_mask_v2 (L136)       - every attention call, decode and prefill
  inkling_sconv_decode (L291)         - decode only (L<=8, mask None, cache present): 4 sites/layer
  inkling_moe_route (L595)            - decode and prefill (GPU): 1 per MoE layer
  inkling_moe_down_combine (L654)     - decode only (N<=8) and only for q4/g64 affine experts
Results -> results/kernels_bench.json
"""
import json
import os
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling import language as L
from mlx_vlm.models.inkling.config import TextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
res = {}


def log(*a):
    print(*a, flush=True)


def med(run, warm=3, reps=9):
    for _ in range(warm):
        mx.eval(run())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(run())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def chained(fn, x0, chain=8, warm=2, reps=7):
    def run():
        x = x0
        for _ in range(chain):
            x = fn(x)
        return x

    return med(run, warm, reps) / chain


def rec(name, ms, note=""):
    res[name] = {"ms": ms, "note": note}
    log(f"{name:70s} {ms:9.4f} ms  {note}")


bf = mx.bfloat16

# ---------------- banded mask v2 + SDPA, decode ----------------
# decode LQ=1: SWA layer sees S=512 after the KV slice; global sees S=ctx
proj_g = (mx.random.normal((16, 1024)) * 0.05).astype(bf)
proj_s = (mx.random.normal((16, 512)) * 0.05).astype(bf)
mx.eval(proj_g, proj_s)
for S, sliding, proj, rel_extent in [(512, 512, proj_s, 512), (2048, 0, proj_g, 1024), (8192, 0, proj_g, 1024), (32768, 0, proj_g, 1024)]:
    r = mx.random.normal((1, 1, 32, 16)).astype(bf)
    k = mx.random.normal((1, 8, S, 128)).astype(bf)
    v = mx.random.normal((1, 8, S, 128)).astype(bf)
    q = mx.random.normal((1, 32, 1, 128)).astype(bf)
    mx.eval(r, k, v, q)
    kind = "SWA" if sliding else "global"
    ms_mask = chained(lambda x: L.banded_additive_mask(x, proj, S - 1, S, sliding, rel_extent, shape_ref=k)[:, :, :, :16].reshape(1, 1, 32, 16) + x, r)
    rec(f"decode {kind} S={S}: banded_mask_v2 [1,32,1,{S}]", ms_mask, f"writes {32*S*2/1e6:.2f} MB")
    mask = L.banded_additive_mask(r, proj, S - 1, S, sliding, rel_extent, shape_ref=k)
    mx.eval(mask)
    ms_sdpa_mask = chained(lambda x: mx.fast.scaled_dot_product_attention(x, k, v, scale=1 / 128, mask=mask) + x, q)
    rec(f"decode {kind} S={S}: SDPA with additive mask", ms_sdpa_mask, f"K+V read {2*8*S*128*2/1e6:.2f} MB")
    ms_sdpa_none = chained(lambda x: mx.fast.scaled_dot_product_attention(x, k, v, scale=1 / 128, mask=None) + x, q)
    rec(f"decode {kind} S={S}: SDPA mask=None", ms_sdpa_none)
    ms_fused_pair = chained(lambda x: mx.fast.scaled_dot_product_attention(x, k, v, scale=1 / 128, mask=L.banded_additive_mask(r, proj, S - 1, S, sliding, rel_extent, shape_ref=k)) + x, q)
    rec(f"decode {kind} S={S}: mask_v2 + SDPA (as in InklingAttention)", ms_fused_pair)
    if sliding == 0:
        # the log-scaling branch on global layers (language.py L468-473)
        tau = mx.array(1.0, dtype=bf).reshape(1, 1, 1, 1)
        ms_tau = chained(lambda x: mx.where(x > -1e29, x * tau, x), mask)
        rec(f"decode global S={S}: tau where() over the mask (L473)", ms_tau)
    del k, v, mask

# ---------------- banded mask v2 + SDPA, prefill ----------------
for LQ, S, sliding, proj, rel_extent in [(2048, 2048, 512, proj_s, 512), (2048, 2559, 512, proj_s, 512), (2048, 2048, 0, proj_g, 1024), (2048, 8192, 0, proj_g, 1024), (8192, 8192, 0, proj_g, 1024)]:
    r = mx.random.normal((1, LQ, 32, 16)).astype(bf)
    k = mx.random.normal((1, 8, S, 128)).astype(bf)
    v = mx.random.normal((1, 8, S, 128)).astype(bf)
    q = mx.random.normal((1, 32, LQ, 128)).astype(bf)
    mx.eval(r, k, v, q)
    kind = "SWA" if sliding else "global"
    reps = 3 if LQ * S > 2048 * 8192 else 5
    ms_mask = med(lambda: L.banded_additive_mask(r, proj, S - LQ, S, sliding, rel_extent, shape_ref=k), warm=1, reps=reps)
    rec(f"prefill {kind} LQ={LQ} S={S}: banded_mask_v2 [1,32,{LQ},{S}]", ms_mask, f"writes {32*LQ*S*2/1e9:.2f} GB")
    mask = L.banded_additive_mask(r, proj, S - LQ, S, sliding, rel_extent, shape_ref=k)
    mx.eval(mask)
    rec(f"prefill {kind} LQ={LQ} S={S}: SDPA with additive mask", med(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=mask), warm=1, reps=reps))
    if S == LQ:
        rec(f"prefill {kind} LQ={LQ} S={S}: SDPA mask='causal' (no bias; reference)", med(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask="causal"), warm=1, reps=reps))
    if sliding == 0:
        tau = (1.0 + 0.1 * mx.log(mx.maximum((mx.arange(LQ) + S - LQ + 1).astype(mx.float32) / 128000, 1.0))).reshape(1, 1, LQ, 1).astype(bf)
        rec(f"prefill global LQ={LQ} S={S}: tau where() over the mask (L473)", med(lambda: mx.where(mask > -1e29, mask * tau, mask), warm=1, reps=reps))
    del k, v, q, mask, r
    mx.clear_cache()

# ---------------- sconv decode kernel vs unfused ----------------
for C in (1024, 4096):
    sc = L.InklingShortConvolution(C, 4, conv_idx=0)
    sc.conv.weight = (mx.random.normal(sc.conv.weight.shape) * 0.1).astype(bf)
    x = mx.random.normal((1, 1, C)).astype(bf)
    cache = ArraysCache(4)
    cache[0] = mx.random.normal((1, 3, C))
    mx.eval(sc.conv.weight, x, cache.cache)
    rec(f"sconv decode fused kernel C={C} (+residual)", chained(lambda t: sc(t, cache=cache, residual=t), x))
    # unfused path: force it by passing a (all-true) mask -> mask is not None
    m = mx.ones((1, 1), dtype=mx.bool_)
    rec(f"sconv decode UNFUSED path C={C} (mask forces mx ops)", chained(lambda t: sc(t, cache=cache, mask=m, residual=t), x))
    xT = mx.random.normal((1, 2048, C)).astype(bf)
    mx.eval(xT)
    caches = [ArraysCache(4) for _ in range(12)]
    it = iter(caches)
    rec(f"sconv prefill T=2048 C={C} (conv1d path)", med(lambda: sc(xT, cache=next(it), residual=xT), warm=2, reps=7))

# ---------------- MoE route kernel vs pure-MLX ----------------
cfg = TextConfig(hidden_size=4096, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=2, intermediate_size=2048, route_scale=8.0)
moe = L.InklingSparseMoE(cfg)
moe.gate_weight = (mx.random.normal((258, 4096)) * 0.05).astype(bf)
moe.e_score_correction_bias = mx.random.normal((256,)) * 0.01
moe.global_scale = mx.ones((1,))
mx.eval(moe.gate_weight, moe.e_score_correction_bias)


def route_pure(logits):
    # verbatim pure-MLX branch of InklingSparseMoE._route (language.py L772-793)
    scores = mx.sigmoid(logits.astype(mx.float32))
    sfc = scores[:, :256] + moe.e_score_correction_bias
    idx = mx.argpartition(-sfc, 5, axis=-1)[:, :6]
    routed_logits = logits[:, :256]
    shared_logits = logits[:, -2:]
    tl = mx.concatenate([mx.take_along_axis(routed_logits, idx, axis=-1), shared_logits], axis=-1).astype(mx.float32)
    lp = -mx.logaddexp(mx.zeros_like(tl), -tl)
    w = mx.exp(lp - mx.logsumexp(lp, axis=-1, keepdims=True)) * 8.0 * moe.global_scale
    topk_w = w[:, :6].astype(logits.dtype)
    gamma = mx.repeat(w[:, -2:].astype(logits.dtype), 2048, axis=-1)
    return idx.astype(mx.uint32), topk_w, gamma


for N in (1, 8, 2048):
    xf = mx.random.normal((N, 4096)).astype(bf)
    logits = xf @ moe.gate_weight.T
    mx.eval(logits)
    rec(f"route kernel N={N} (inkling_moe_route)", med(lambda: moe._route(logits)))
    rec(f"route pure-MLX N={N} (L772-793 chain)", med(lambda: route_pure(logits)))
    rec(f"router gemv N={N} (xf @ gate_weight.T, bf16 258x4096)", med(lambda: xf @ moe.gate_weight.T))

# ---------------- down-combine kernel (q4) vs gather_qmm, N=1 and N=8 ----------------
for bits in (4, 2):
    moe = L.InklingSparseMoE(cfg)  # fresh module per bit width (to_quantized is one-way)
    moe.gate_weight = (mx.random.normal((258, 4096)) * 0.05).astype(bf)
    moe.e_score_correction_bias = mx.random.normal((256,)) * 0.01
    moe.global_scale = mx.ones((1,))
    for p in ("gate_proj", "up_proj", "down_proj"):
        setattr(moe.switch_mlp, p, getattr(moe.switch_mlp, p).to_quantized(group_size=64, bits=bits))
    moe.shared_experts.gate_proj = nn.Linear(4096, 4096, bias=False).to_quantized(group_size=64, bits=bits)
    moe.shared_experts.up_proj = nn.Linear(4096, 4096, bias=False).to_quantized(group_size=64, bits=bits)
    moe.shared_experts.down_proj = nn.Linear(4096, 4096, bias=False).to_quantized(group_size=64, bits=bits)
    moe.set_dtype(bf)
    moe.switch_mlp.gate_scale = mx.ones((256,))
    moe.switch_mlp.out_scale = mx.ones((256,))
    moe.switch_mlp._scales_trivial = None
    moe._wscale = None
    mx.eval(moe.parameters())
    for N in (1, 3, 8):
        x = mx.random.normal((1, N, 4096)).astype(bf)
        mx.eval(x)
        L._DOWN_COMBINE = True
        a = chained(lambda t: moe(t), x, chain=6)
        L._DOWN_COMBINE = False
        b = chained(lambda t: moe(t), x, chain=6)
        L._DOWN_COMBINE = True
        rec(f"MoE block q{bits} N={N}: default path (down_combine kernel if q4)", a)
        rec(f"MoE block q{bits} N={N}: gather_qmm down path (_DOWN_COMBINE=False)", b)
    # 2048-token prefill MoE block at this bit width
    x = mx.random.normal((1, 2048, 4096)).astype(bf)
    mx.eval(x)
    rec(f"MoE block q{bits} N=2048 (prefill, sorted gather_qmm)", med(lambda: moe(x), warm=1, reps=3))
    xr = x.reshape(-1, 4096)
    idx = mx.random.randint(0, 256, (2048, 6)).astype(mx.uint32)
    mx.eval(idx)
    xe = mx.expand_dims(xr, (-2, -3))
    xs, idxs, inv = L._gather_sort(xe, idx)
    mx.eval(xs, idxs, inv)
    rec(f"  gather_qmm q{bits} up_proj N=2048 sorted (x gathered [12288,1,4096])", med(lambda: moe.switch_mlp.up_proj(xs, idxs, sorted_indices=True), warm=1, reps=3))
    rec(f"  gather_qmm q{bits} up_proj N=2048 UNsorted", med(lambda: moe.switch_mlp.up_proj(xe, idx, sorted_indices=False), warm=1, reps=3))
    act = mx.random.normal((12288, 1, 2048)).astype(bf)
    mx.eval(act)
    rec(f"  gather_qmm q{bits} down_proj N=2048 sorted", med(lambda: moe.switch_mlp.down_proj(act, idxs, sorted_indices=True), warm=1, reps=3))
    # dense reference: the same FLOPs as a dense bf16 GEMM [2048,4096]x[4096,2048] x 6 x 3
    w = (mx.random.normal((2048, 4096)) * 0.02).astype(bf)
    mx.eval(w)
    rec(f"  reference bf16 GEMM [2048,4096]x[4096,2048] (1/18 of the MoE FLOPs)", med(lambda: xr @ w.T, warm=2, reps=5))
    wq = nn.Linear(4096, 2048, bias=False).to_quantized(group_size=64, bits=bits)
    mx.eval(wq.parameters())
    rec(f"  reference q{bits} quantized_matmul [2048,4096]x[4096,2048]", med(lambda: wq(xr), warm=2, reps=5))

# ---------------- Python / dispatch overhead: tiny-shape model, same graph ----------------
tiny = TextConfig(
    hidden_size=512, num_hidden_layers=3, vocab_size=4096, unpadded_vocab_size=4000,
    num_attention_heads=8, num_key_value_heads=2, head_dim=64, swa_num_attention_heads=8,
    swa_num_key_value_heads=2, swa_head_dim=64, sliding_window_size=512,
    layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid"], mlp_layer_types=["dense", "sparse", "sparse"],
    d_rel=16, rel_extent=1024, log_scaling_n_floor=128000, sconv_kernel_size=4, dense_mlp_idx=2,
    intermediate_size=64, dense_intermediate_size=1024, n_routed_experts=256, num_experts_per_tok=6,
    n_shared_experts=2, route_scale=8.0, logits_mup_width_multiplier=16.0,
)
tm = L.LanguageModel(tiny)
for layer in tm.model.layers:
    layer.self_attn.rel_proj = mx.random.normal(layer.self_attn.rel_proj.shape) * 0.05
    if isinstance(layer.mlp, L.InklingSparseMoE):
        layer.mlp.gate_weight = mx.random.normal(layer.mlp.gate_weight.shape) * 0.05
tm.set_dtype(bf)
nn.quantize(tm, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, (nn.Linear,)) or hasattr(m, "to_quantized") and not isinstance(m, nn.Embedding))
for layer in tm.model.layers:
    layer.mlp.global_scale = mx.ones((1,))
    if isinstance(layer.mlp, L.InklingSparseMoE):
        layer.mlp.e_score_correction_bias = mx.zeros((256,))
        layer.mlp.switch_mlp.gate_scale = mx.ones((256,))
        layer.mlp.switch_mlp.out_scale = mx.ones((256,))
mx.eval(tm.parameters())
tm.eval()
caches = tm.make_cache()
for c in caches:
    c[0].update_and_fetch(mx.zeros((1, 2, 1024, 64), dtype=bf), mx.zeros((1, 2, 1024, 64), dtype=bf))
    for i, C in enumerate((128, 128, 512, 512)):
        c[1][i] = mx.zeros((1, 3, C))


def tstep(y):
    out = tm(y, cache=caches)
    return mx.argmax(out.logits[:, -1, :], axis=-1)


y = mx.array([[5]])
nxt = tstep(y)
mx.eval(nxt)
ts, builds = [], []
nxt = tstep(nxt[None])
mx.async_eval(nxt)
for _ in range(40):
    t0 = time.perf_counter()
    n2 = tstep(nxt[None])
    tb = time.perf_counter()
    mx.async_eval(n2)
    mx.eval(nxt)
    ts.append(time.perf_counter() - t0)
    builds.append(tb - t0)
    nxt = n2
rec("tiny-shape 3-layer decode step: wall per token (async loop) ~ dispatch floor", statistics.median(ts) * 1e3, "same op graph as the real model, GPU work negligible")
rec("tiny-shape 3-layer decode step: python graph-build time per token", statistics.median(builds) * 1e3)
res["tiny_per_layer_floor_ms"] = {"note": "(wall - head)/3 per layer; x42 = per-token floor",}

with open(os.path.join(OUT, "kernels_bench.json"), "w") as f:
    json.dump(res, f, indent=1)
log("wrote", os.path.join(OUT, "kernels_bench.json"))
