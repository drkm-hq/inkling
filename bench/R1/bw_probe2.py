"""Bandwidth probe v2: dispatch-amortized timing.

v1 timed one op per mx.eval, so every number carried the ~0.3-0.4 ms
command-buffer/sync latency. Here each timed eval runs a chain of `chain`
dependent ops over distinct weight copies (total footprint > the SLC so the
weights stream from DRAM); per-op time = chain time / chain.
Results -> results/bw_probe2.json
"""
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
results = {}


def chain_time(make_step, n_weights, chain, warm=3, reps=15):
    """make_step(i) -> fn(y) using weight copy i % n_weights. Returns ms per step (median)."""
    steps = [make_step(i) for i in range(n_weights)]
    x0 = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    mx.eval(x0)

    def run():
        y = x0
        for j in range(chain):
            y = steps[j % n_weights](y)
        return y

    for _ in range(warm):
        mx.eval(run())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(run())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3 / chain


def rec(name, ms, nbytes, extra=None):
    gbs = nbytes / (ms * 1e-3) / 1e9
    results[name] = {"ms_per_op": ms, "bytes": nbytes, "GB/s": gbs, **(extra or {})}
    print(f"{name:58s} {ms:8.4f} ms/op {nbytes/1e6:9.1f} MB {gbs:7.1f} GB/s", flush=True)


def qbytes(nparams, bits, group=64):
    return nparams * bits / 8 + nparams / group * 4


# square-ish projections that map 4096 -> 4096 so the chain composes
def make_linear(dout, din, bits):
    lin = nn.Linear(din, dout, bias=False)
    lin.weight = (mx.random.normal((dout, din)) * 0.02).astype(mx.bfloat16)
    if bits is None:
        mx.eval(lin.weight)
        return lin
    ql = lin.to_quantized(group_size=64, bits=bits)
    mx.eval(ql.parameters())
    return ql


# 1) o_proj-shaped 4096x4096: 16 distinct copies (bf16: 537 MB total) chained 32x
for bits in (None, 8, 4, 2):
    n = 16
    mods = [make_linear(4096, 4096, bits) for _ in range(n)]
    ms = chain_time(lambda i: (lambda y: mods[i](y)), n, 32)
    nb = 4096 * 4096 * 2 if bits is None else qbytes(4096 * 4096, bits)
    rec(f"gemv M=1 4096x4096 {'bf16' if bits is None else f'q{bits}'} (16 copies, chain 32)", ms, nb)
    del mods

# 2) qkvr-shaped 6656x4096 followed by a slice back to 4096
for bits in (None, 8, 4):
    n = 12
    mods = [make_linear(6656, 4096, bits) for _ in range(n)]
    ms = chain_time(lambda i: (lambda y: mods[i](y)[..., :4096]), n, 24)
    nb = 6656 * 4096 * 2 if bits is None else qbytes(6656 * 4096, bits)
    rec(f"gemv M=1 6656x4096 {'bf16' if bits is None else f'q{bits}'} (+slice) (12 copies, chain 24)", ms, nb)
    del mods

# 3) dense MLP 16384x4096 then 4096x16384 (gate only + down) chained
for bits in (None, 8, 4, 2):
    n = 4
    ups = [make_linear(16384, 4096, bits) for _ in range(n)]
    downs = [make_linear(4096, 16384, bits) for _ in range(n)]
    ms = chain_time(lambda i: (lambda y: downs[i](ups[i](y))), n, 12)
    nb = 2 * 16384 * 4096 * (2 if bits is None else 0) + (0 if bits is None else 2 * qbytes(16384 * 4096, bits))
    rec(f"gemv M=1 dense up+down 16384x4096 x2 {'bf16' if bits is None else f'q{bits}'} (4 copies, chain 12)", ms, nb)
    del ups, downs

# 4) lm_head 201024x4096: one copy (1.65 GB bf16 > SLC), chain 6, slice back to 4096 input
for bits in (None, 8, 4, 2):
    head = make_linear(201024, 4096, bits)
    ms = chain_time(lambda i: (lambda y: head(y)[..., :4096]), 1, 6)
    nb = 201024 * 4096 * 2 if bits is None else qbytes(201024 * 4096, bits)
    rec(f"gemv M=1 lm_head 201024x4096 {'bf16' if bits is None else f'q{bits}'} (chain 6)", ms, nb)
    del head

# 5) routed experts gather_qmm: SwitchGLU at 2/4 bits, N=1 K=6, chained over distinct index sets
from mlx_vlm.models.switch_layers import SwitchGLU

for bits in (2, 4):
    sw = SwitchGLU(4096, 2048, 256)
    for p in ("gate_proj", "up_proj", "down_proj"):
        setattr(sw, p, getattr(sw, p).to_quantized(group_size=64, bits=bits))
    mx.eval(sw.parameters())
    idxs = [mx.array([[(i * 37 + j * 41) % 256 for j in range(6)]], dtype=mx.uint32) for i in range(16)]
    mx.eval(idxs)
    w6 = mx.full((1, 6, 1), 1.0 / 6, dtype=mx.bfloat16)
    mx.eval(w6)

    def step_factory(i):
        def step(y):
            out = sw(y.reshape(1, 4096), idxs[i])  # [1, 6, 4096]
            return (out * w6).sum(axis=1).reshape(1, 1, 4096)
        return step

    ms = chain_time(step_factory, 16, 16, warm=3, reps=10)
    rec(f"SwitchGLU gather_qmm q{bits} N=1 K=6 up+gate+down (chain 16)", ms, 6 * qbytes(3 * 2048 * 4096, bits))
    # down only
    acts = [mx.random.normal((1, 6, 1, 2048)).astype(mx.bfloat16) for _ in range(16)]
    mx.eval(acts)

    def down_factory(i):
        def step(y):
            out = sw.down_proj(acts[i] + y[..., :1, :1, :1].reshape(1, 1, 1, 1), idxs[i])  # [1,6,1,4096]
            return out.sum(axis=1).reshape(1, 1, 4096)
        return step

    ms = chain_time(down_factory, 16, 16, warm=3, reps=10)
    rec(f"gather_qmm q{bits} down_proj only N=1 K=6 (chain 16)", ms, 6 * qbytes(2048 * 4096, bits))

    def up_factory(i):
        def step(y):
            out = sw.up_proj(y.reshape(1, 1, 1, 4096), idxs[i])  # [1,6,1,2048]
            return mx.concatenate([out.sum(axis=1).reshape(1, 1, 2048)] * 2, axis=-1)
        return step

    ms = chain_time(up_factory, 16, 16, warm=3, reps=10)
    rec(f"gather_qmm q{bits} up_proj only N=1 K=6 (chain 16)", ms, 6 * qbytes(2048 * 4096, bits))
    del sw

with open(os.path.join(OUT, "bw_probe2.json"), "w") as f:
    json.dump({"mlx": mx.__version__, "results": results}, f, indent=1)
print("wrote", os.path.join(OUT, "bw_probe2.json"), flush=True)
