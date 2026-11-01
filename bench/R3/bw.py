"""Memory-bandwidth probe on the GPU: streaming elementwise ops and a bf16 GEMV.

Reports GB/s (bytes moved / median time). The elementwise ops count read+write
traffic; the reduce counts reads only; the GEMV counts the weight bytes.
"""

import mlx.core as mx

from benchutil import Recorder

rec = Recorder("bw")
rec.note(section="bandwidth probe")

for gib in (1, 4):
    nbytes = gib << 30
    rows = nbytes // 2 // 4096
    a = mx.random.normal((rows, 4096), dtype=mx.float16)
    b = mx.random.normal((rows, 4096), dtype=mx.float16)
    mx.eval(a, b)
    rec.row(f"a+1  fp16 {gib} GiB (read+write)", lambda: a + 1, nbytes=2 * nbytes, reps=(1, 4), iters=20)
    rec.row(f"a*b  fp16 {gib} GiB (2 reads+write)", lambda: a * b, nbytes=3 * nbytes, reps=(1, 4), iters=20)
    rec.row(f"sum(a) fp16 {gib} GiB (read)", lambda: mx.sum(a), nbytes=nbytes, reps=(1, 4), iters=20)
    del a, b
    mx.clear_cache()

for m, k in ((16384, 4096), (16384, 16384), (32768, 16384)):
    w = mx.random.normal((m, k), dtype=mx.bfloat16)
    x = mx.random.normal((1, k), dtype=mx.bfloat16)
    mx.eval(w, x)
    rec.row(f"gemv bf16 {m}x{k} ({m*k*2/2**30:.2f} GiB)", lambda: x @ w.T, nbytes=m * k * 2, reps=(1, 8), iters=30)
    del w, x
    mx.clear_cache()
