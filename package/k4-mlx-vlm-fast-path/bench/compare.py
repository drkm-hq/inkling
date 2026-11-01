#!/usr/bin/env python
"""Compare equiv_bench.py runs: equivalence verdict and a before/after perf table.

Usage: python compare.py results/before.jsonl results/after.jsonl [more.jsonl ...]
The first file is the reference; every other file is compared with it per prompt:
identical token ids, or the first divergent index with the top-2 logit margins
of both runs at that step (a small margin means a near-tie, not a code change).
"""
import json
import sys
from pathlib import Path


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    meta = next((r for r in rows if r["kind"] == "meta"), {})
    summaries = {r["prompt"]: r for r in rows if r["kind"] == "summary"}
    tf = {r["prompt"]: r for r in rows if r["kind"] == "tf"}
    if tf and not summaries:
        meta["_tf"] = tf
    runs = {}
    for r in rows:
        if r["kind"] == "run" and not r.get("aborted") and not r.get("error"):
            runs.setdefault(r["prompt"], []).append(r)
    loads = [r for r in rows if r["kind"] == "load"]
    return meta, summaries, runs, loads


def fmt(stat, nd=1):
    if stat is None:
        return "-"
    return f"{stat['median']:.{nd}f} ({stat['min']:.{nd}f}-{stat['max']:.{nd}f})"


def main(paths):
    data = [(Path(p).stem, *load(p)) for p in paths]
    ref_tag, ref_meta, ref_sum, ref_runs, _ = data[0]
    print("## Runs\n")
    for tag, meta, _, _, loads in data:
        ld = loads[0] if loads else {}
        print(
            f"- `{tag}`: mlx_vlm {meta.get('mlx_vlm')} from `{meta.get('mlx_vlm_file')}` "
            f"(git {meta.get('git')}), mlx {meta.get('mlx')}, load1 at start {meta.get('load1', 0):.1f}, "
            f"load {'eager' if ld.get('eager') else 'lazy+chunked'} {ld.get('load_s', 0) + ld.get('eval_s', 0):.1f}s, "
            f"{ld.get('param_gib', 0):.1f} GiB"
        )
    print("\n## Equivalence (greedy token ids vs `%s`)\n" % ref_tag)
    for tag, meta_o, summaries, runs, _ in data[1:]:
        if meta_o.get("_tf"):
            continue
        for prompt, s in ref_sum.items():
            o = summaries.get(prompt)
            if o is None:
                print(f"- {tag} / {prompt}: no run")
                continue
            a, b = s["token_ids"], o["token_ids"]
            same_within = s["identical_outputs"] and o["identical_outputs"]
            n = min(len(a), len(b))
            first = next((i for i in range(n) if a[i] != b[i]), None)
            if first is None and len(a) == len(b):
                same_margins = s["margins"] == o["margins"]
                ties = sum(1 for m in s["margins"] if m == 0)
                print(
                    f"- {tag} / {prompt}: IDENTICAL {len(a)} tokens "
                    f"(runs self-consistent: {same_within}); top-2 logit margins "
                    f"{'identical at every step' if same_margins else 'DIFFER'}; "
                    f"{ties} exact top-2 ties on the path, smallest non-zero margin "
                    f"{min(m for m in s['margins'] if m):.3f}"
                )
            else:
                i = first if first is not None else n
                ma = s["margins"][i] if i < len(s["margins"]) else None
                mb = o["margins"][i] if i < len(o["margins"]) else None
                print(
                    f"- {tag} / {prompt}: DIVERGES at token {i}: {a[i] if i < len(a) else None} vs "
                    f"{b[i] if i < len(b) else None}; top-2 margin before {ma} / after {mb}; "
                    f"identical prefix {i} tokens"
                )
    tf_runs = [(tag, meta["_tf"]) for tag, meta, *_ in data if meta.get("_tf")]
    if tf_runs:
        print("\n## Teacher-forced (same 128 tokens fed to every run; KL(reference || run) in nats)\n")
        print("| run | prompt | chunk | reference | identical logprobs | KL mean | KL max (step) | mean abs dlogprob of teacher token | max | top-1 agree |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for tag, tf in tf_runs:
            for prompt, r in tf.items():
                if "kl_mean" in r:
                    print(
                        f"| {tag} | {prompt} | {r['chunk']} | {r['reference']} | {r['identical']} | "
                        f"{r['kl_mean']:.2e} | {r['kl_max']:.2e} ({r['kl_argmax']}) | {r['dlogprob_mean_abs']:.2e} | "
                        f"{r['dlogprob_max_abs']:.2e} | {r['top1_agree']}/{r['n']} |"
                    )
                else:
                    print(f"| {tag} | {prompt} | {r['chunk']} | (reference) | md5 {r['md5'][:12]} | | | | | teacher top-1 {r['top1_matches_teacher']}/{r['n']} |")
    print("\n## Performance (median of timed runs, min-max in parentheses)\n")
    print("| run | prompt | prompt tok | prefill tok/s | TTFT s | decode tok/s | peak GiB (mx) | total s |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, _, summaries, runs, _ in data:
        for prompt, s in summaries.items():
            print(
                f"| {tag} | {prompt} | {s['prompt_tokens']} | {fmt(s['prompt_tps_api'])} | "
                f"{fmt(s['ttft_wall_s'], 2)} | {fmt(s['decode_tps_wall'], 2)} | "
                f"{s['peak_mem_gib_mx']['max']:.1f} | {fmt(s['total_wall_s'])} |"
            )
    print("\n## Ratios vs `%s` (medians)\n" % ref_tag)
    print("| run | prompt | prefill | TTFT | decode | peak GiB delta |")
    print("|---|---|---|---|---|---|")
    for tag, _, summaries, _, _ in data[1:]:
        for prompt, s in summaries.items():
            r = ref_sum.get(prompt)
            if r is None:
                continue
            print(
                f"| {tag} | {prompt} | {s['prompt_tps_api']['median'] / r['prompt_tps_api']['median']:.3f}x | "
                f"{s['ttft_wall_s']['median'] / r['ttft_wall_s']['median']:.3f}x | "
                f"{s['decode_tps_wall']['median'] / r['decode_tps_wall']['median']:.3f}x | "
                f"{s['peak_mem_gib_mx']['max'] - r['peak_mem_gib_mx']['max']:+.1f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
