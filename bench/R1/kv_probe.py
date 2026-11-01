"""Does KVCache.update_and_fetch (cache.py L508-530) copy the whole K/V buffer on
every decode token? It does `self.keys[..., prev:offset, :] = keys` (slice_update) and
returns views; MLX can only update in place when nothing else references the old
buffer. Measures ms per one-token update at ctx 2K/8K/32K for:
  A. sync: one update per mx.eval (old buffer unreferenced -> in-place possible)
  B. pipelined like generate_step: next update built+dispatched before waiting on the
     previous one (previous graph may still hold the old buffer)
  C. chained: 8 updates in one graph (old buffers definitely referenced)
plus the sliding-layer K/V slice `k[:, :, j0:, :]` followed by SDPA, and a
RotatingKVCache(max_size=512) alternative for the sliding layers.
Results -> results/kv_probe.json"""
import json
import os
import statistics
import time

import mlx.core as mx

from mlx_vlm.models.cache import KVCache, RotatingKVCache

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {}
bf = mx.bfloat16
k1 = mx.random.normal((1, 8, 1, 128)).astype(bf)
v1 = mx.random.normal((1, 8, 1, 128)).astype(bf)
mx.eval(k1, v1)

# warm clocks
a = mx.random.normal((4096, 4096)).astype(bf)
mx.eval(a)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 3:
    a = (a @ a) * 1e-3
    mx.eval(a)


def fresh(ctx, rotating=False):
    c = RotatingKVCache(max_size=512) if rotating else KVCache()
    if ctx:
        k = mx.random.normal((1, 8, ctx, 128)).astype(bf)
        v = mx.random.normal((1, 8, ctx, 128)).astype(bf)
        c.update_and_fetch(k, v)
        mx.eval(c.keys, c.values)
    return c


for ctx in (2048, 8192, 32768):
    r = {"kv_bytes_per_layer": 2 * 8 * ctx * 128 * 2}
    # A. sync
    c = fresh(ctx)
    ts = []
    for i in range(40):
        t = time.perf_counter()
        k, v = c.update_and_fetch(k1, v1)
        mx.eval(k)
        ts.append(time.perf_counter() - t)
    r["A_sync_ms"] = statistics.median(ts[10:]) * 1e3
    # B. pipelined (two graphs in flight), each step also runs a tiny consumer of k
    c = fresh(ctx)
    k, v = c.update_and_fetch(k1, v1)
    s = mx.sum(k[..., -1:, :])
    mx.async_eval(s)
    ts = []
    for i in range(40):
        t = time.perf_counter()
        k2, v2 = c.update_and_fetch(k1, v1)
        s2 = mx.sum(k2[..., -1:, :])
        mx.async_eval(s2)
        mx.eval(s)
        ts.append(time.perf_counter() - t)
        s = s2
    r["B_pipelined_ms"] = statistics.median(ts[10:]) * 1e3
    # C. chained 8 per eval
    c = fresh(ctx)

    def chain():
        out = None
        for _ in range(8):
            k, v = c.update_and_fetch(k1, v1)
            out = mx.sum(k[..., -1:, :]) if out is None else out + mx.sum(k[..., -1:, :])
        return out

    for _ in range(3):
        mx.eval(chain())
    ts = []
    for _ in range(8):
        t = time.perf_counter()
        mx.eval(chain())
        ts.append(time.perf_counter() - t)
    r["C_chained_ms_per_update"] = statistics.median(ts) * 1e3 / 8
    # D. the sliding-layer slice + SDPA on the sliced views (A-style sync)
    c = fresh(ctx)
    q = mx.random.normal((1, 32, 1, 128)).astype(bf)
    mx.eval(q)
    ts = []
    for i in range(30):
        t = time.perf_counter()
        k, v = c.update_and_fetch(k1, v1)
        S = k.shape[2]
        j0 = S - 1 - 511
        ks, vs = k[:, :, j0:, :], v[:, :, j0:, :]
        o = mx.fast.scaled_dot_product_attention(q, ks, vs, scale=1 / 128, mask=None)
        mx.eval(o)
        ts.append(time.perf_counter() - t)
    r["D_update+slice+sdpa_sync_ms"] = statistics.median(ts[10:]) * 1e3
    # E. same with a RotatingKVCache(512) (bounded) for comparison
    c = fresh(min(ctx, 4096), rotating=True)
    ts = []
    for i in range(30):
        t = time.perf_counter()
        k, v = c.update_and_fetch(k1, v1)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=None)
        mx.eval(o)
        ts.append(time.perf_counter() - t)
    r["E_rotating512_update+sdpa_sync_ms"] = statistics.median(ts[10:]) * 1e3
    res[f"ctx{ctx}"] = r
    print(f"ctx {ctx}: " + ", ".join(f"{kk}={vv:.4f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in r.items()), flush=True)

with open(os.path.join(OUT, "kv_probe.json"), "w") as f:
    json.dump(res, f, indent=1)
print("wrote kv_probe.json")
