"""Does mx.fast.scaled_dot_product_attention stay fused when InklingAttention hands
it a dense additive mask [1,32,LQ,S]? Time + peak-memory delta per call, for
mask=array (what language.py L475 does), mask="causal", mask=None, force_fused=True
with the array mask, and the pure-MLX unfused reference. A peak-memory delta of the
order of 32*LQ*S*4 bytes means the scores were materialized (fallback path).
Results -> results/sdpa_probe.json"""
import json
import os
import statistics
import time

import mlx.core as mx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {}
bf = mx.bfloat16


def timed(fn, warm=2, reps=5):
    for _ in range(warm):
        mx.eval(fn())
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    peak = mx.get_peak_memory() - base
    return statistics.median(ts) * 1e3, peak / 1e9


def unfused(q, k, v, mask, scale):
    n_rep = q.shape[1] // k.shape[1]
    kk = mx.repeat(k, n_rep, axis=1)
    vv = mx.repeat(v, n_rep, axis=1)
    s = (q * scale) @ kk.transpose(0, 1, 3, 2) + mask
    return mx.softmax(s, axis=-1, precise=True) @ vv


# warm the GPU clocks
a = mx.random.normal((4096, 4096)).astype(bf)
mx.eval(a)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 3:
    a = (a @ a) * 1e-3
    mx.eval(a)

for LQ, S in [(1, 512), (1, 8192), (1, 32768), (8, 2048), (512, 512), (2048, 2048), (2048, 2559), (2048, 8192), (8192, 8192)]:
    q = mx.random.normal((1, 32, LQ, 128)).astype(bf)
    k = mx.random.normal((1, 8, S, 128)).astype(bf)
    v = mx.random.normal((1, 8, S, 128)).astype(bf)
    # a banded additive mask like the model's: 0 inside the window, -1e30 outside
    qi = mx.arange(LQ)[:, None] + (S - LQ)
    kj = mx.arange(S)[None, :]
    dist = qi - kj
    base = mx.where((dist >= 0) & (dist < 1024), mx.random.normal((LQ, S)) * 0.1, mx.array(-1e30))
    mask = mx.broadcast_to(base[None, None], (1, 32, LQ, S)).astype(bf)
    mask = mx.contiguous(mask)
    mx.eval(q, k, v, mask)
    r = {}
    r["array_mask"] = timed(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=mask))
    try:
        r["array_mask_force_fused"] = timed(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=mask, force_fused=True))
    except Exception as e:  # noqa
        r["array_mask_force_fused"] = str(e)[:120]
    r["bool_mask"] = timed(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=(mask > -1e29)))
    r["none"] = timed(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=None))
    if LQ > 1:
        r["causal_str"] = timed(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask="causal"))
    if LQ * S <= 2048 * 8192:
        r["unfused_reference"] = timed(lambda: unfused(q, k, v, mask, 1 / 128), warm=1, reps=3)
    # numerical check fused vs unfused (small sizes only)
    if LQ * S <= 2048 * 2048:
        f = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=mask)
        u = unfused(q, k, v, mask, 1 / 128)
        r["max_abs_diff_vs_unfused"] = float(mx.max(mx.abs(f.astype(mx.float32) - u.astype(mx.float32))).item())
    res[f"LQ={LQ},S={S}"] = r
    print(f"LQ={LQ:5d} S={S:6d}: " + ", ".join(f"{kk}={vv}" if isinstance(vv, str) else (f"{kk}=({vv[0]:.3f} ms, +{vv[1]:.2f} GB)" if isinstance(vv, tuple) else f"{kk}={vv:.3g}") for kk, vv in r.items()), flush=True)
    del q, k, v, mask
    mx.clear_cache()

with open(os.path.join(OUT, "sdpa_probe.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("wrote sdpa_probe.json")
