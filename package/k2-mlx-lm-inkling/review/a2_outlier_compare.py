#!/usr/bin/env python
"""Compare the per-layer dumps written by a2_forced.py for the same decode step
of the two implementations: where the residual stream diverges, whether a
routed expert set differs there, and how close to a tie that decision was.

  python a2_outlier_compare.py review/a2-dumps p_2k_raw 45 46 47 48
"""

import sys
from pathlib import Path

import numpy as np

LOCAL = [i for i in range(42) if (i + 1) % 6 != 0]


def kl(ref_logits, logits):
    ref = ref_logits - np.logaddexp.reduce(ref_logits)
    mine = logits - np.logaddexp.reduce(logits)
    p = np.exp(ref)
    return float((p * (ref - mine)).sum()), ref, mine


def main(d, run, steps):
    d = Path(d)
    for s in steps:
        a = np.load(d / f"mlx_vlm_{run}_step{int(s):03d}.npz")
        b = np.load(d / f"mlx_lm_{run}_step{int(s):03d}.npz")
        assert int(a["token"]) == int(b["token"]) and int(a["position"]) == int(b["position"])
        k, ref, mine = kl(a["logits"].astype(np.float64), b["logits"].astype(np.float64))
        ta, tb = np.argsort(-ref)[:5], np.argsort(-mine)[:5]
        print(f"\n== {run} step {s}: fed token {int(a['token'])} at position {int(a['position'])}: "
              f"KL(mlx-vlm || mlx-lm) = {k:.4f} nats; mlx-vlm top-5 {ta.tolist()} (lp {ref[ta].round(2).tolist()}); "
              f"mlx-lm top-5 {tb.tolist()} (lp {mine[tb].round(2).tolist()})")
        print("| layer | type | max|d|/max|ref| | mean|d|/mean|ref| | routed set differs | gap mlx-vlm (bf16 logits) | gap mlx-lm (fp32 logits) | weights |")
        print("|---|---|---|---|---|---|---|---|")
        moe = 0
        first_diff = None
        for i in range(a["layers"].shape[0]):
            ra, rb = a["layers"][i].astype(np.float64), b["layers"][i].astype(np.float64)
            dev = np.abs(ra - rb)
            route = "dense"
            if i >= 2:
                sa, sb = a["routes"][moe], b["routes"][moe]
                diff = set(sa.tolist()) != set(sb.tolist())
                if diff and first_diff is None:
                    first_diff = i
                route = (f"{'YES' if diff else 'no'} | {a['gaps'][moe]:.5f} | {b['gaps'][moe]:.5f} | "
                         f"vlm {np.round(np.sort(a['route_weights'][moe])[::-1], 3).tolist()} / lm {np.round(np.sort(b['route_weights'][moe])[::-1], 3).tolist()}")
                if diff:
                    route += f" ; sets vlm {sa.tolist()} lm {sb.tolist()}"
                moe += 1
            else:
                route = "dense | | | "
            print(f"| {i} | {'local' if i in LOCAL else 'GLOBAL'} | {dev.max() / np.abs(ra).max():.2e} | "
                  f"{dev.mean() / np.abs(ra).mean():.2e} | {route} |")
        print(f"first layer whose routed expert set differs: {first_diff}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
