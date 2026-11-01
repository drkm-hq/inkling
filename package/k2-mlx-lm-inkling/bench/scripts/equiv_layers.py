"""Per-layer comparison of the cache-free forward over p_short dumped by equiv.py (--extras
hidden): relative deviation of the residual stream after the embedding norm and after every
layer, and the fraction of tokens whose routed expert set differs, mlx-lm vs mlx-vlm.

  python equiv_layers.py ../results/K2-hidden
"""
import glob
import os
import sys

import numpy as np


def main(d):
    a_files = sorted(glob.glob(os.path.join(d, "mlx_vlm_layer*.npy")))
    print(f"{len(a_files)} layers; residual stream [tokens, hidden] after each layer (bf16 model, dumped as fp32)")
    print("| layer | max|d| / max|ref| | mean|d| / mean|ref| | tokens with a different routed expert set |")
    print("|---|---|---|---|")
    ref = np.load(os.path.join(d, "mlx_vlm_embed.npy")); mine = np.load(os.path.join(d, "mlx_lm_embed.npy"))
    dev = np.abs(ref - mine)
    print(f"| embed+norm | {dev.max() / np.abs(ref).max():.2e} | {dev.mean() / np.abs(ref).mean():.2e} | - |")
    moe = 0
    for i in range(len(a_files)):
        ref = np.load(os.path.join(d, f"mlx_vlm_layer{i:02d}.npy")); mine = np.load(os.path.join(d, f"mlx_lm_layer{i:02d}.npy"))
        dev = np.abs(ref - mine)
        ra = os.path.join(d, f"mlx_vlm_route{moe:02d}.npy"); rb = os.path.join(d, f"mlx_lm_route{moe:02d}.npy")
        route = "dense"
        if i >= 2 and os.path.exists(ra) and os.path.exists(rb):
            A, B = np.load(ra), np.load(rb)
            n_diff = int(np.any(A != B, axis=-1).sum())
            route = f"{n_diff}/{A.shape[0]} ({100 * n_diff / A.shape[0]:.1f}%)"
            moe += 1
        print(f"| {i} | {dev.max() / np.abs(ref).max():.2e} | {dev.mean() / np.abs(ref).mean():.2e} | {route} |")


if __name__ == "__main__":
    main(sys.argv[1])
