"""Markdown summary of the equivalence evidence: the mlx-vlm self-noise floor, the mlx-lm
teacher-forced comparison (KL, logprob deltas, top-1 agreement, router audit), the greedy
divergence steps, and the per-layer table.

  python equiv_report.py > ../results/K2-equivalence.md
"""
import json
import os
import re
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ref = load("K2-equiv-mlx_vlm.json")
    forced = load("K2-equiv-mlx_lm-forced.json")
    greedy = load("K2-equiv-mlx_lm.json")
    print("# Equivalence: mlx-lm Inkling vs mlx-vlm 0.7.11 on mlx-community/Inkling-Small-mlx-2bit\n")
    if ref:
        print(f"Reference: {ref['impl']} {ref['version']} from `{ref['python']}` (mlx {ref['mlx']}); "
              f"mlx-lm from `{greedy['python'] if greedy else '?'}`. Greedy, 128 new tokens, prefill chunk {ref['prefill_step_size']}, "
              f"wired limit {ref['wired_limit_gib']:.0f} GiB, identical prompt token ids on both sides.\n")
    print("## Teacher-forced comparison (same prefix fed to both; per-step distributions compared)\n")
    print("| run | prompt tok | noise floor: mlx-vlm vs itself (prefill 2048 vs 128) KL mean / median / max, top-1 agree | "
          "mlx-lm vs mlx-vlm KL mean / median / max, top-1 agree | mlx-lm |dlogprob| of ref top-1 mean / max |")
    print("|---|---|---|---|---|")
    fr = {r["name"]: r for r in forced["runs"]} if forced else {}
    for r in (ref["runs"] if ref else []):
        nf = r.get("noise_floor")
        vs = fr.get(r["name"], {}).get("vs_reference")
        nf_s = (f"{nf['kl_mean']:.4f} / {nf['kl_median']:.4f} / {nf['kl_max']:.3f}, {100 * nf['top1_agreement']:.1f}%" if nf else "-")
        vs_s = (f"{vs['kl_mean']:.4f} / {vs['kl_median']:.4f} / {vs['kl_max']:.3f}, {100 * vs['top1_agreement']:.1f}%" if vs else "pending")
        d_s = f"{vs['d_top1_mean']:.3f} / {vs['d_top1_max']:.3f}" if vs else "pending"
        print(f"| {r['name']} | {r['prompt_tokens']} | {nf_s} | {vs_s} | {d_s} |")
    if forced:
        for r in forced["runs"]:
            if "router_audit" in r:
                print(f"\nRouter audit ({r['name']}, prompt + 128 tokens, fp32 vs bf16 router logits inside mlx-lm): {r['router_audit']['summary']}")
    print("\n## Greedy runs (first divergence; the top-2 logit gap at that step on each side)\n")
    diff = os.path.join(R, "K2-equiv-diff.txt")
    if os.path.exists(diff):
        lines = open(diff).read().splitlines()
        for i, l in enumerate(lines):
            if "DIVERGE" in l or "IDENTICAL" in l:
                print("- " + l.strip())
                for m in lines[i + 1 : i + 5]:
                    mm = re.match(r"\s+([AB]): chosen (\d+) logit ([\d.]+), the other's token (\d+) logit ([\d.]+), gap ([\d.]+)", m)
                    if mm:
                        side, c, lc, o, lo, gap = mm.groups()
                        print(f"  - {'mlx-vlm' if side == 'A' else 'mlx-lm'}: chose {c} ({lc}) over {o} ({lo}), gap {gap}")
    layers = os.path.join(R, "K2-equiv-layers.md")
    if os.path.exists(layers):
        print("\n## Per-layer residual stream and routing on p_short (cache-free forward, 226 tokens)\n")
        print(open(layers).read())


if __name__ == "__main__":
    main()
