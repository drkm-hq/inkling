#!/usr/bin/env python
"""Like-for-like reduced-precision check on the R5 tiny random model: K2's
mlx-lm Inkling and mlx-vlm 0.7.11's LanguageModel, both cast to bf16 (and
fp16) the way the real checkpoints are loaded (router, global scales and conv
taps kept fp32), decoded through their Metal kernel paths (prefill 7 + 9 single
tokens), against the float64 oracle and against each other. Routing flips vs the
fp32 model are counted per implementation so the MoE amplification of near
ties is separated from arithmetic differences.

Run with the base venv (mlx_vlm 0.7.11 wheel) and mlx_lm pointed at the reviewed tree:
  PYTHONPATH=~/code/inkling-contrib/upstream/mlx-lm-a2 ~/.venvs/inkling-base/bin/python a2_tiny_bf16.py
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path(os.environ.get("A2_TREE", HERE.parents[2] / "upstream" / "mlx-lm-a2"))
sys.path.insert(0, str(W / "tests"))
sys.path.insert(0, str(HERE.parents[2] / "research" / "oracle"))

import test_inkling_a2 as A  # noqa: E402  (sets MLX_ENABLE_TF32=0 before importing mlx)
import mlx.core as mx  # noqa: E402
import test_oracle_vs_mlx_vlm as H  # noqa: E402
from mlx.utils import tree_map_with_path  # noqa: E402
from mlx_vlm.models.inkling import language as L  # noqa: E402

import mlx_lm  # noqa: E402

KEEP = ("gate", "global_scale", "sconv", "e_score_correction_bias")


def cast_like_checkpoint(model, dtype):
    def cast(path, a):
        if mx.issubdtype(a.dtype, mx.floating) and not any(k in path for k in KEEP):
            return a.astype(dtype)
        return a

    model.update(tree_map_with_path(cast, model.parameters()))
    mx.eval(model.parameters())
    return model


def vlm_forward_with_routes(lm, ids, chunks):
    sparse = [i for i, t in enumerate(A.TINY["mlp_layer_types"]) if t == "sparse"]
    calls = []
    orig = L.InklingSparseMoE._route

    def hooked(self, logits):
        idx, tw, gamma = orig(self, logits)
        calls.append(np.array(mx.sort(idx.astype(mx.int32), axis=-1)))
        return idx, tw, gamma

    L.InklingSparseMoE._route = hooked
    try:
        cache = lm.make_cache()
        x = mx.array(ids)
        outs, pos = [], 0
        for n in chunks:
            outs.append(lm(x[:, pos : pos + n], cache=cache).logits)
            mx.eval(outs[-1])
            pos += n
        out = np.array(mx.concatenate(outs, 1).astype(mx.float32), np.float64)
    finally:
        L.InklingSparseMoE._route = orig
    routes, pos, k = {}, 0, 0
    for n in chunks:
        for i in sparse:
            inds = calls[k]
            k += 1
            for row in range(inds.shape[0]):
                routes[(i, row // n, pos + row % n)] = tuple(inds[row].tolist())
        pos += n
    assert k == len(calls)
    return out, routes


def main():
    mx.set_default_device(mx.gpu)
    ids, chunks = A.IDS40[:, :16], [7] + [1] * 9
    ref = A.oracle_forward(ids)
    print(f"mlx_lm from {mlx_lm.__file__}")
    print(f"mlx_vlm from {L.__file__}")
    lm32 = H.run_lm(mx.gpu)[0]
    out_v32, routes_v32 = vlm_forward_with_routes(lm32, ids, chunks)
    out_k32, routes_k32 = A.forward_with_routes(A.build_model(), ids, chunks)
    print(f"fp32 (TF32 off): mlx-vlm vs oracle {A.rel(out_v32, ref):.2e}, mlx-lm vs oracle {A.rel(out_k32, ref):.2e}, "
          f"routes identical: {routes_v32 == routes_k32}")
    for dtype in (mx.bfloat16, mx.float16):
        lm = cast_like_checkpoint(H.run_lm(mx.gpu)[0], dtype)
        k2 = cast_like_checkpoint(A.build_model(), dtype)
        out_v, routes_v = vlm_forward_with_routes(lm, ids, chunks)
        out_k, routes_k = A.forward_with_routes(k2, ids, chunks)
        flips_v = {(b, p) for (i, b, p), r in routes_v.items() if routes_v32[(i, b, p)] != r}
        flips_k = {(b, p) for (i, b, p), r in routes_k.items() if routes_k32[(i, b, p)] != r}
        keep = np.ones(ids.shape, bool)
        for b, p in flips_v | flips_k:
            keep[b, p] = False
        print(f"\n{dtype}: tokens whose routed experts differ from fp32: mlx-vlm {len(flips_v)}/{ids.size}, mlx-lm {len(flips_k)}/{ids.size}; "
              f"{int(keep.sum())} tokens routed identically by both")
        for name, a, b in [
            ("mlx-vlm vs oracle, all tokens", out_v, ref),
            ("mlx-lm  vs oracle, all tokens", out_k, ref),
            ("mlx-lm  vs mlx-vlm, all tokens", out_k, out_v),
            ("mlx-vlm vs oracle, identically routed tokens", out_v[keep], ref[keep]),
            ("mlx-lm  vs oracle, identically routed tokens", out_k[keep], ref[keep]),
            ("mlx-lm  vs mlx-vlm, identically routed tokens", out_k[keep], out_v[keep]),
        ]:
            print(f"  {name:48s} max|d|/max|ref| = {A.rel(a, b):.3e}")
        # greedy tokens and per-position top-1 agreement with the oracle
        am_r, am_v, am_k = ref.argmax(-1), out_v.argmax(-1), out_k.argmax(-1)
        print(f"  top-1 agreement with the oracle over {ids.size} positions: mlx-vlm {np.mean(am_v == am_r):.3f}, mlx-lm {np.mean(am_k == am_r):.3f}; "
              f"mlx-lm == mlx-vlm {np.mean(am_k == am_v):.3f}")


if __name__ == "__main__":
    main()
