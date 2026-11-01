"""Shared timing helpers for the R3 micro-benchmarks (MLX, Apple Silicon).

Timing model: ``fn`` builds a lazy MLX graph for one op (or a short op chain).
``timeit`` evaluates ``reps`` independent copies per iteration so that
``reps=1`` measures a synchronous single dispatch (what a decode loop pays per
op when nothing else is queued) and ``reps=8`` amortises the launch/eval
overhead (closer to the per-op kernel time when many ops are queued, as in a
42-layer forward). Results are medians over ``iters`` iterations plus the best.
"""

import json
import os
import statistics
import time

import mlx.core as mx

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sysinfo():
    load = os.getloadavg()
    info = mx.device_info()
    return {
        "mlx": mx.__version__,
        "device": info["device_name"],
        "arch": info.get("architecture"),
        "load1": round(load[0], 1),
        "load5": round(load[1], 1),
    }


def timeit(fn, *, reps=1, iters=30, warm=5):
    """Return (median_ms, best_ms) per single invocation of ``fn``."""
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        outs = [fn() for _ in range(reps)]
        t0 = time.perf_counter()
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) / reps)
        del outs
    return statistics.median(ts) * 1e3, min(ts) * 1e3


class Recorder:
    def __init__(self, name):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.path = os.path.join(RESULTS_DIR, f"{name}.jsonl")
        self.f = open(self.path, "a")
        self.f.write(json.dumps({"_sysinfo": sysinfo(), "_t": time.time()}) + "\n")

    def row(self, label, fn, *, nbytes=None, reps=(1, 8), iters=30, **extra):
        rec = {"label": label, **extra}
        line = f"{label:58s}"
        for r in reps:
            med, best = timeit(fn, reps=r, iters=iters)
            rec[f"med_ms_r{r}"] = round(med, 4)
            rec[f"best_ms_r{r}"] = round(best, 4)
            line += f" | r{r}: {med:8.3f} ms (best {best:8.3f})"
            if nbytes:
                rec[f"gbps_r{r}"] = round(nbytes / (med * 1e-3) / 1e9, 1)
                line += f" {nbytes / (med * 1e-3) / 1e9:6.0f} GB/s"
        if nbytes:
            rec["nbytes"] = nbytes
        print(line, flush=True)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        return rec

    def note(self, **kw):
        print(json.dumps(kw), flush=True)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()


def affine_bytes(n_out, n_in, bits, group_size=64):
    """Bytes of a packed affine-quantized [n_out, n_in] matrix incl. bf16 scales+biases."""
    return n_out * n_in * bits // 8 + n_out * (n_in // group_size) * 2 * 2


def qmatrix(shape_out, shape_in, bits, group_size=64, lead=()):
    """Random packed affine-quantized matrix with the exact dtypes/shapes MLX expects."""
    packed = shape_in * bits // 32
    w = mx.random.randint(-(2**31), 2**31 - 1, (*lead, shape_out, packed), dtype=mx.int32).view(
        mx.uint32
    )
    s = mx.random.uniform(0.001, 0.01, (*lead, shape_out, shape_in // group_size)).astype(mx.bfloat16)
    b = mx.random.uniform(-0.01, 0.01, (*lead, shape_out, shape_in // group_size)).astype(mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b
