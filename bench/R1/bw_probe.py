"""Achievable memory bandwidth per op type on this machine, at Inkling-Small shapes.

Decode (M=1) is weight-bandwidth bound, so what matters is the GB/s each
primitive reaches when streaming weights once per token:
  * bf16 gemv (nn.Linear) at the attention / lm_head shapes
  * quantized_matmul (nn.QuantizedLinear) at 2/4/8 bits, group 64
  * gather_qmm for the routed experts (N=1, K=6) at 2/4 bits
  * a plain bf16 copy as the memcpy ceiling

Each number is the median of `reps` timed runs after `warm` warmups, with
mx.eval() after every call (synchronous timing). Results -> results/bw_probe.json
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


def timeit(fn, warm=5, reps=25):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3, min(ts) * 1e3


def qbytes(nparams, bits, group=64):
    # packed weight + bf16 scale + bf16 bias per group
    return nparams * bits / 8 + nparams / group * 4


results = {}


def rec(name, ms, nbytes, extra=None):
    gbs = nbytes / (ms * 1e-3) / 1e9
    results[name] = {"ms": ms, "bytes": nbytes, "GB/s": gbs, **(extra or {})}
    print(f"{name:52s} {ms:8.3f} ms  {nbytes/1e6:9.1f} MB  {gbs:7.1f} GB/s")


# --- memcpy ceiling: copy a 1 GB bf16 buffer into a preallocated destination ---
src = mx.random.normal((512, 1024, 1024)).astype(mx.bfloat16)  # 1 GB
mx.eval(src)
ms, _ = timeit(lambda: src + mx.array(0, dtype=mx.bfloat16), warm=3, reps=10)
rec("copy bf16 1GB read + 1GB write", ms, 2 * src.nbytes)
ms, _ = timeit(lambda: mx.sum(src, axis=(1, 2)), warm=3, reps=10)
rec("reduce bf16 1GB read (sum axis 1,2)", ms, src.nbytes)
del src

x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
mx.eval(x)

# --- dense projections at Inkling shapes ---
shapes = {
    "qkvr_proj 6656x4096": (4096, 6656),
    "o_proj 4096x4096": (4096, 4096),
    "shared gate+up 4096x4096 (2 experts x 2048)": (4096, 4096),
    "dense_mlp gate 16384x4096": (4096, 16384),
    "lm_head 201024x4096": (4096, 201024),
}
for name, (din, dout) in shapes.items():
    lin = nn.Linear(din, dout, bias=False)
    lin.weight = mx.random.normal((dout, din)).astype(mx.bfloat16) * 0.02
    mx.eval(lin.weight)
    ms, _ = timeit(lambda: lin(x))
    rec(f"bf16 gemv {name}", ms, dout * din * 2)
    for bits in (8, 4, 2):
        ql = lin.to_quantized(group_size=64, bits=bits)
        mx.eval(ql.parameters())
        ms, _ = timeit(lambda: ql(x))
        rec(f"q{bits} gemv {name}", ms, qbytes(dout * din, bits))
        del ql
    del lin

# --- M=8 (MTP verify block) at the same shapes: bytes are the same, time should be ~same ---
x8 = mx.random.normal((1, 8, 4096)).astype(mx.bfloat16)
mx.eval(x8)
lin = nn.Linear(4096, 6656, bias=False)
lin.weight = mx.random.normal((6656, 4096)).astype(mx.bfloat16) * 0.02
mx.eval(lin.weight)
ms, _ = timeit(lambda: lin(x8))
rec("bf16 gemm M=8 qkvr_proj 6656x4096", ms, 6656 * 4096 * 2)
ql = lin.to_quantized(group_size=64, bits=8)
mx.eval(ql.parameters())
ms, _ = timeit(lambda: ql(x8))
rec("q8 gemm M=8 qkvr_proj 6656x4096", ms, qbytes(6656 * 4096, 8))
del lin, ql

# --- routed experts: gather_qmm N=1, K=6 at 2 and 4 bits ---
from mlx_vlm.models.switch_layers import SwitchGLU

for bits in (2, 4):
    sw = SwitchGLU(4096, 2048, 256)
    for p in ("gate_proj", "up_proj", "down_proj"):
        sub = getattr(sw, p)
        setattr(sw, p, sub.to_quantized(group_size=64, bits=bits))
    mx.eval(sw.parameters())
    idx = mx.array([[3, 50, 77, 120, 200, 255]], dtype=mx.uint32)
    xe = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    mx.eval(idx, xe)
    nb = 6 * qbytes(3 * 2048 * 4096, bits)
    ms, _ = timeit(lambda: sw(xe, idx))
    rec(f"gather_qmm SwitchGLU q{bits} N=1 K=6 (up+gate+down)", ms, nb)
    # individual projections
    xq = mx.expand_dims(xe, (-2, -3))
    ms, _ = timeit(lambda: sw.up_proj(xq, idx))
    rec(f"gather_qmm q{bits} up_proj [256,2048,4096] N=1 K=6", ms, 6 * qbytes(2048 * 4096, bits))
    act = mx.random.normal((1, 6, 1, 2048)).astype(mx.bfloat16)
    mx.eval(act)
    ms, _ = timeit(lambda: sw.down_proj(act, idx))
    rec(f"gather_qmm q{bits} down_proj [256,4096,2048] N=1 K=6", ms, 6 * qbytes(2048 * 4096, bits))
    # N=8 tokens (MTP verify): 48 expert-slots, distinct experts
    idx8 = mx.array([[i * 5 + j for j in range(6)] for i in range(8)], dtype=mx.uint32)
    xe8 = mx.random.normal((8, 4096)).astype(mx.bfloat16)
    mx.eval(idx8, xe8)
    ms, _ = timeit(lambda: sw(xe8, idx8))
    rec(f"gather_qmm SwitchGLU q{bits} N=8 K=6 distinct (48 slots)", ms, 48 * qbytes(3 * 2048 * 4096, bits))
    # N=2048 prefill (sorted path): all experts touched
    idx2k = mx.random.randint(0, 256, (2048, 6)).astype(mx.uint32)
    xe2k = mx.random.normal((2048, 4096)).astype(mx.bfloat16)
    mx.eval(idx2k, xe2k)
    ms, _ = timeit(lambda: sw(xe2k, idx2k), warm=3, reps=10)
    rec(f"gather_qmm SwitchGLU q{bits} N=2048 K=6 (sorted path)", ms, 256 * qbytes(3 * 2048 * 4096, bits),
        {"note": "bytes = all 256 experts once; compute 2048*6*3*2048*4096*2 flops"})
    del sw

with open(os.path.join(OUT, "bw_probe.json"), "w") as f:
    json.dump({"mlx": mx.__version__, "device": mx.device_info(), "results": results}, f, indent=1, default=str)
print("wrote", os.path.join(OUT, "bw_probe.json"))
