"""Decode cost of the sliding-layer attention state at Inkling-Small shapes for three cache
designs, chained over the 35 sliding layers in one eval (B=1, 8 kv heads x 128, 32 query
heads, window 512, bf16):

  shifting : ShiftingWindowCache + reversed-slice bias (the mlx-lm model)
  rotating : RotatingKVCache(512) + one shared slot->distance index + a gather per layer
  full     : KVCache (full history) sliced to the window + the banded mask kernel (mlx-vlm style)

Each step: cache update, bias construction, fused SDPA; the per-step time is the median of
--iters synchronous evals after warm-up. Small arrays only; safe to run without the GPU lock.
"""
import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache, ShiftingWindowCache
from mlx_lm.models.inkling_mm_model import banded_mask

B, HKV, H, D, W, E = 1, 8, 32, 128, 512, 512


def step_shifting(caches, k, v, q, rl):
    outs = []
    for c in caches:
        kk, vv = c.update_and_fetch(k, v)
        S = kk.shape[2]
        bias = rl[..., ::-1][..., E - S :] if S <= E else mx.pad(rl[..., ::-1], [(0, 0)] * 3 + [(S - E, 0)])
        outs.append(mx.fast.scaled_dot_product_attention(q, kk, vv, scale=1 / D, mask=bias))
    return outs


def step_rotating(caches, k, v, q, rl):
    outs = []
    c0 = caches[0]
    S = min(c0.offset + 1, W)
    idx = c0._idx if c0.offset < W else (c0._idx if c0._idx < W else 0)
    dist = (idx - mx.arange(S)) % W  # slot s holds the key (idx - s) mod W positions back, before the write
    dist = mx.broadcast_to(dist[None, None, None], (B, H, 1, S))
    for c in caches:
        kk, vv = c.update_and_fetch(k, v)
        bias = mx.take_along_axis(rl, dist, axis=-1)
        outs.append(mx.fast.scaled_dot_product_attention(q, kk, vv, scale=1 / D, mask=bias))
    return outs


def step_full(caches, k, v, q, rl):
    outs = []
    for c in caches:
        kk, vv = c.update_and_fetch(k, v)
        S = kk.shape[2]
        if S > W:
            kk, vv = kk[:, :, S - W :], vv[:, :, S - W :]
            S = W
        mask = banded_mask(rl, S, W, E)
        outs.append(mx.fast.scaled_dot_product_attention(q, kk, vv, scale=1 / D, mask=mask))
    return outs


def bench(name, make, step, context, iters, layers=35):
    caches = [make() for _ in range(layers)]
    kp = mx.random.normal((B, HKV, context, D)).astype(mx.bfloat16)
    for c in caches:
        c.update_and_fetch(kp, kp)
    mx.eval([c.state for c in caches])
    k = mx.random.normal((B, HKV, 1, D)).astype(mx.bfloat16)
    q = mx.random.normal((B, H, 1, D)).astype(mx.bfloat16)
    rl = mx.random.normal((B, H, 1, E)).astype(mx.bfloat16)
    times = []
    for i in range(iters + 5):
        t0 = time.perf_counter()
        outs = step(caches, k, k, q, rl)
        mx.eval(outs, [c.state for c in caches])
        t = time.perf_counter() - t0
        if i >= 5:
            times.append(t)
    med = statistics.median(times) * 1e3
    print(f"  {name:9s} ctx {context:5d}: {med:.3f} ms/step for {layers} layers "
          f"({med / layers * 1e3:.1f} us/layer, best {min(times) * 1e3:.3f} ms)")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    for ctx in (128, 600, 4096):
        print(f"context {ctx} (+1 decode token), window {W}")
        bench("shifting", lambda: ShiftingWindowCache(W), step_shifting, ctx, args.iters)
        bench("rotating", lambda: RotatingKVCache(W), step_rotating, ctx, args.iters)
        bench("full", lambda: KVCache(), step_full, ctx, args.iters)


if __name__ == "__main__":
    main()
