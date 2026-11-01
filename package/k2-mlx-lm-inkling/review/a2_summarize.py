#!/usr/bin/env python
"""Markdown summary of A2's hold: the teacher-forced statistics next to K2's hold-2 numbers
(bench/results/K2-equiv-mlx_lm-forced.json), the mlx-vlm determinism check, the p_32k answer,
the back-to-back benchmarks (a2-bench-*.jsonl summary rows) and the attn8 variant.

  python a2_summarize.py > a2-hold-summary.md
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "bench" / "results"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def fmt(st):
    if not st:
        return "-"
    return (f"{st['kl_mean']:.4f} / {st['kl_median']:.4f} / {st['kl_max']:.3f} (step {st.get('kl_argmax', '?')}), "
            f"top-1 {100 * st['top1_agreement']:.1f} %, |dlp| top-1 max {st['d_top1_max']:.3f}")


def main():
    k2 = load(RESULTS / "K2-equiv-mlx_lm-forced.json")
    k2_runs = {r["name"]: r for r in k2["runs"]} if k2 else {}
    ref = load(RESULTS / "K2-equiv-mlx_vlm.json")
    ref_runs = {r["name"]: r for r in ref["runs"]} if ref else {}
    mine = load(HERE / "a2-forced-mlx_lm.json")
    vlm = load(HERE / "a2-forced-mlx_vlm.json")
    print("# A2 hold summary\n")
    if mine:
        i = mine["info"]
        print(f"mlx-lm {i['version']} at `{i.get('git_head')}` (dirty: {i.get('git_dirty')}), {i['module']}; load {i['load_lazy_s']:.1f} + {i['load_eval_s']:.1f} s, "
              f"{i['param_gib']:.1f} GiB, load1 at start {i['load1']:.1f}, peak {mine.get('peak_gib', 0):.1f} GiB\n")
    print("## Teacher-forced on the mlx-vlm 0.7.11 tokens of K2's hold 2, vs the saved reference logprobs\n")
    print("| run | prompt tok | noise floor (mlx-vlm vs itself, chunk 2048 vs 128) | K2 hold 2: KL mean / median / max, top-1 | A2 re-run (same statistic) | A2 decode tok/s (forced loop) | argmax == fed |")
    print("|---|---|---|---|---|---|---|")
    for r in (mine["runs"] if mine else []):
        n = r["name"]
        nf = ref_runs.get(n, {}).get("noise_floor")
        print(f"| {n} | {r['prompt_tokens']} | {fmt(nf)} | {fmt(k2_runs.get(n, {}).get('vs_reference'))} | {fmt(r.get('vs_reference'))} | "
              f"{r['decode_tps']:.1f} | {r['teacher_agree']}/{len(r['fed'])} |")
    if vlm:
        print("\n## mlx-vlm 0.7.11 re-run on its own tokens (determinism of the reference logprobs)\n")
        for r in vlm["runs"]:
            print(f"- {r['name']}: {fmt(r.get('vs_reference'))}; argmax == fed {r['teacher_agree']}/{len(r['fed'])}; decode {r['decode_tps']:.1f} tok/s")
    if mine and "p32k" in mine:
        p = mine["p32k"]
        print(f"\n## p_32k (chat template, reasoning effort none)\n\n{p['prompt_tokens']} prompt tokens, TTFT {p['ttft_s']:.1f} s "
              f"({p['prompt_tps']:.0f} tok/s), {p['n_tokens']} tokens at {p['generation_tps']:.1f} tok/s, peak {p['peak_gib']:.1f} GiB "
              f"(active {p['active_gib']:.1f}), answer_ok={p['answer_ok']}, finish={p['finish_reason']}, load1 {p['load1']:.1f}\n\n"
              f"text: {p['text'][:400]!r}")
    print("\n## Back-to-back benchmarks in A2's hold (median of 3, after one warm-up)\n")
    print("| engine | prompt | prompt tok | prefill tok/s (api) | TTFT s | decode tok/s (wall) | peak GiB (mx) | load1 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, path in [("mlx-lm (a2)", HERE / "a2-bench-mlxlm.jsonl"), ("mlx-vlm 0.7.11 (a2)", HERE / "a2-bench-mlxvlm.jsonl"),
                      ("mlx-lm attn8 (a2)", HERE / "a2-bench-attn8.jsonl")]:
        rows = jsonl(path)
        runs = [r for r in rows if r.get("kind") == "run"]
        for s in [r for r in rows if r.get("kind") == "summary"]:
            loads = [r.get("load1", 0) for r in runs if r.get("prompt") == s["prompt"]]
            pk = s.get("peak_mem_gib_mx_median")
            if pk is None and "peak_mem_gib_mx" in s:
                pk = s["peak_mem_gib_mx"].get("median") if isinstance(s["peak_mem_gib_mx"], dict) else s["peak_mem_gib_mx"]
            def med(key):
                v = s.get(key + "_median")
                if v is None and isinstance(s.get(key), dict):
                    v = s[key].get("median")
                return v
            print(f"| {tag} | {s['prompt']} | {s.get('prompt_tokens')} | {med('prompt_tps_api') or 0:.0f} | {med('ttft_wall_s') or 0:.2f} | "
                  f"{med('decode_tps_wall') or 0:.1f} | {pk or 0:.1f} | {max(loads) if loads else 0:.1f} |")
    attn8 = load(HERE / "a2-forced-attn8.json")
    if attn8:
        print("\n## 8-bit-attention variant (p_short, teacher forced on the same tokens, vs the bf16-attention mlx-lm logprobs)\n")
        for r in attn8["runs"]:
            print(f"- {r['name']}: {fmt(r.get('vs_reference'))}; decode {r['decode_tps']:.1f} tok/s (forced loop); peak {attn8.get('peak_gib', 0):.1f} GiB")
    for name in ("a2-cli-generate.log", "a2-server-smoke.log"):
        p = HERE / name
        if p.exists():
            txt = p.read_text()
            print(f"\n## {name} (tail)\n\n```\n{txt[-1800:]}\n```")


if __name__ == "__main__":
    main()
