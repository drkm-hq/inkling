"""(a) MoE expert matmuls at Inkling-Small shapes with gather_qmm.

Shapes: hidden H=4096, expert intermediate I=2048, E=256 routed experts,
top-k K=6, 2 shared experts. Affine quantization, group size 64, bits in
{2,3,4}. Weights are random bit patterns of the exact packed layout MLX uses
(timing-exact; values are garbage).

Paths measured
  decode  (T=1): gather_qmm with rhs_indices of 6 experts (mlx-lm SwitchGLU,
          indices.size < 64 so no sort); up, gate, down separately and the
          fused gate+up [E, 2I, H] variant.
  prefill (T=512, 2048): mlx-lm's sorted path (_gather_sort + sorted_indices=True),
          which on MLX 0.32.4 takes the gather_qmm_rhs kernel when
          M == 1 and B >= 16 and B/E >= 4.
  shared experts at decode: gather_qmm over 2 experts vs one dense
          quantized_matmul with the two experts concatenated (2I = 4096).
  router: fp32 [258, 4096] gemv + sigmoid + argpartition + softmax (graph).
"""

import argparse

import mlx.core as mx
import mlx.nn as nn

from benchutil import Recorder, affine_bytes, qmatrix

H, I, E, K = 4096, 2048, 256, 6
GS = 64

p = argparse.ArgumentParser()
p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
p.add_argument("--prefill", type=int, nargs="+", default=[512, 2048])
args = p.parse_args()

rec = Recorder("moe")


def gqmm(x, w, s, b, idx, bits, sorted_indices=False):
    return mx.gather_qmm(
        x, w, s, b, rhs_indices=idx, transpose=True, group_size=GS, bits=bits,
        sorted_indices=sorted_indices,
    )


def gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


for bits in args.bits:
    rec.note(section=f"routed experts bits={bits}")
    w_up, s_up, b_up = qmatrix(I, H, bits, lead=(E,))
    w_gate, s_gate, b_gate = qmatrix(I, H, bits, lead=(E,))
    w_down, s_down, b_down = qmatrix(H, I, bits, lead=(E,))
    per_expert_up = affine_bytes(I, H, bits)
    per_expert_down = affine_bytes(H, I, bits)

    # ---------------- decode, T = 1 ----------------
    x = mx.random.normal((1, 1, 1, H)).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (1, K))
    mx.eval(x, idx)
    rec.row(f"decode gather_qmm up    4096->2048 x6 {bits}b", lambda: gqmm(x, w_up, s_up, b_up, idx, bits),
            nbytes=K * per_expert_up, bits=bits, T=1)
    act = gqmm(x, w_up, s_up, b_up, idx, bits)
    mx.eval(act)
    rec.row(f"decode gather_qmm down  2048->4096 x6 {bits}b", lambda: gqmm(act, w_down, s_down, b_down, idx, bits),
            nbytes=K * per_expert_down, bits=bits, T=1)

    def glu_decode():
        u = gqmm(x, w_up, s_up, b_up, idx, bits)
        g = gqmm(x, w_gate, s_gate, b_gate, idx, bits)
        return gqmm(nn.silu(g) * u, w_down, s_down, b_down, idx, bits)

    rec.row(f"decode SwitchGLU (3 gather_qmm) x6 {bits}b", glu_decode,
            nbytes=K * (2 * per_expert_up + per_expert_down), bits=bits, T=1)

    # fused gate+up: one gather_qmm of 4096 -> 4096 then split
    w_13, s_13, b_13 = qmatrix(2 * I, H, bits, lead=(E,))

    def glu_decode_fused():
        gu = gqmm(x, w_13, s_13, b_13, idx, bits)
        g, u = mx.split(gu, 2, axis=-1)
        return gqmm(nn.silu(g) * u, w_down, s_down, b_down, idx, bits)

    rec.row(f"decode SwitchGLU fused gate+up (2 gather_qmm) x6 {bits}b", glu_decode_fused,
            nbytes=K * (2 * per_expert_up + per_expert_down), bits=bits, T=1)
    del w_13, s_13, b_13
    mx.clear_cache()

    # ---------------- prefill, sorted path ----------------
    for T in args.prefill:
        xT = mx.random.normal((T, 1, 1, H)).astype(mx.bfloat16)
        idxT = mx.random.randint(0, E, (T, K))
        mx.eval(xT, idxT)

        def glu_prefill():
            xs, ids, inv = gather_sort(xT, idxT)
            u = gqmm(xs, w_up, s_up, b_up, ids, bits, sorted_indices=True)
            g = gqmm(xs, w_gate, s_gate, b_gate, ids, bits, sorted_indices=True)
            y = gqmm(nn.silu(g) * u, w_down, s_down, b_down, ids, bits, sorted_indices=True)
            return mx.unflatten(y[inv], 0, idxT.shape)

        n_touch = min(E, T * K)  # with random routing nearly every expert is touched
        r = rec.row(f"prefill T={T} SwitchGLU sorted x6 {bits}b (per call)", glu_prefill,
                    nbytes=n_touch * (2 * per_expert_up + per_expert_down), bits=bits, T=T,
                    reps=(1,), iters=10)
        flops = T * K * 3 * 2 * H * I
        rec.note(label=f"prefill T={T} {bits}b per-token us", us_per_token=round(r["med_ms_r1"] * 1e3 / T, 1),
                 tflops=round(flops / (r["med_ms_r1"] * 1e-3) / 1e12, 2))
        if T == 512:
            def glu_prefill_unsorted():
                u = gqmm(xT, w_up, s_up, b_up, idxT, bits)
                g = gqmm(xT, w_gate, s_gate, b_gate, idxT, bits)
                return gqmm(nn.silu(g) * u, w_down, s_down, b_down, idxT, bits)
            rec.row(f"prefill T={T} SwitchGLU UNsorted x6 {bits}b (per call)", glu_prefill_unsorted,
                    bits=bits, T=T, reps=(1,), iters=10)
        del xT, idxT
        mx.clear_cache()

    del w_up, s_up, b_up, w_gate, s_gate, b_gate, w_down, s_down, b_down, act
    mx.clear_cache()

# ---------------- shared experts (2, always on) at decode ----------------
for bits in args.bits:
    rec.note(section=f"shared experts bits={bits}")
    w_up, s_up, b_up = qmatrix(I, H, bits, lead=(2,))
    w_gate, s_gate, b_gate = qmatrix(I, H, bits, lead=(2,))
    w_down, s_down, b_down = qmatrix(H, I, bits, lead=(2,))
    x = mx.random.normal((1, 1, 1, H)).astype(mx.bfloat16)
    idx2 = mx.array([[0, 1]])
    mx.eval(x, idx2)

    def shared_gather():
        u = gqmm(x, w_up, s_up, b_up, idx2, bits)
        g = gqmm(x, w_gate, s_gate, b_gate, idx2, bits)
        return gqmm(nn.silu(g) * u, w_down, s_down, b_down, idx2, bits).sum(axis=1)

    nb = 2 * (2 * affine_bytes(I, H, bits) + affine_bytes(H, I, bits))
    rec.row(f"decode shared experts via gather_qmm (E=2) {bits}b", shared_gather, nbytes=nb, bits=bits)

    # dense fused: concatenate the two experts -> up/gate [2I, H], down [H, 2I]
    wd_up, sd_up, bd_up = qmatrix(2 * I, H, bits)
    wd_gate, sd_gate, bd_gate = qmatrix(2 * I, H, bits)
    wd_down, sd_down, bd_down = qmatrix(H, 2 * I, bits)
    x2 = mx.random.normal((1, H)).astype(mx.bfloat16)
    mx.eval(x2)

    def shared_dense():
        u = mx.quantized_matmul(x2, wd_up, sd_up, bd_up, transpose=True, group_size=GS, bits=bits)
        g = mx.quantized_matmul(x2, wd_gate, sd_gate, bd_gate, transpose=True, group_size=GS, bits=bits)
        return mx.quantized_matmul(nn.silu(g) * u, wd_down, sd_down, bd_down, transpose=True, group_size=GS, bits=bits)

    rec.row(f"decode shared experts as dense GLU (2I=4096) {bits}b", shared_dense, nbytes=nb, bits=bits)
    del w_up, s_up, b_up, w_gate, s_gate, b_gate, w_down, s_down, b_down, wd_up, sd_up, bd_up, wd_gate, sd_gate, bd_gate, wd_down, sd_down, bd_down
    mx.clear_cache()

# ---------------- router ----------------
rec.note(section="router")
Wr = mx.random.normal((E + 2, H))
br = mx.random.normal((E,)) * 0.01
x = mx.random.normal((1, H)).astype(mx.bfloat16)
mx.eval(Wr, br, x)


def router():
    flat = x.astype(mx.float32)
    logits = flat @ Wr.T
    scores = mx.sigmoid(logits)[:, :E] + br
    top = mx.argpartition(-scores, kth=K - 1, axis=-1)[:, :K]
    sel = mx.concatenate([mx.take_along_axis(logits[:, :E], top, axis=-1), logits[:, E:]], axis=-1)
    w = mx.softmax(-mx.logaddexp(mx.zeros_like(sel), -sel), axis=-1) * 8.0
    return top, w


rec.row("decode router fp32 (gemv+sigmoid+argpartition+softmax)", lambda: router()[1], nbytes=(E + 2) * H * 4)
