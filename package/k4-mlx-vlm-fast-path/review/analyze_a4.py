#!/usr/bin/env python3
"""Summarise the A4 hold: before/after generation (min and median of reps, load),
token/margin identity, teacher-forced comparisons, kv8 and the eager load.
Usage: python analyze_a4.py <results_dir>"""
import glob
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
rows = []
for f in sorted(glob.glob(str(d / "*.jsonl"))):
    for line in open(f):
        rows.append(json.loads(line))

by = lambda kind: [r for r in rows if r.get("kind") == kind]
print("# meta")
for r in by("meta"):
    print(f"- {r['tag']}: {r['mlx_vlm_file']} git={r.get('git')} load1@start={r['load1']:.1f} tf32={r.get('env_tf32')}")
for r in by("load"):
    print(f"- load {r['tag']}: eager={r['eager']} load_s={r['load_s']:.1f} eval_s={r['eval_s']:.1f} params={r['param_gib']:.1f} GiB peak={r['peak_gib']:.1f} GiB")
for r in by("cache_kinds"):
    print(f"- cache kinds {r['tag']}: {r['kinds']}")

print("\n# generation runs (per rep: prefill tok/s | TTFT s | decode tok/s | peak GiB | load1)")
runs = {}
for r in by("run"):
    if r.get("error") or r.get("aborted"):
        print(f"  ! {r['tag']} {r['prompt']} rep{r.get('rep')}: error={r.get('error')} aborted={r.get('aborted')}")
        continue
    runs.setdefault((r["tag"], r["prompt"]), []).append(r)
table = []
for (tag, prompt), rs in sorted(runs.items()):
    pf = [r["prompt_tps_api"] for r in rs]; tt = [r["ttft_wall_s"] for r in rs]
    dc = [r["decode_tps_wall"] or 0 for r in rs]; pk = [r["peak_mem_gib_mx"] for r in rs]; ld = [r["load1"] for r in rs]
    print(f"- {tag:16s} {prompt:8s} n={len(rs)} gen={rs[0]['gen_tokens']} "
          + " ; ".join(f"{a:.0f}|{b:.2f}|{c:.1f}|{p:.1f}|{l:.0f}" for a, b, c, p, l in zip(pf, tt, dc, pk, ld)))
    table.append((tag, prompt, len(rs), max(pf), statistics.median(pf), min(tt), statistics.median(tt), max(dc), statistics.median(dc), max(pk), min(ld), max(ld)))
print("\n| run | prompt | n | prefill tok/s best / median | TTFT s best / median | decode tok/s best / median | peak GiB | load1 range |")
print("|---|---|---|---|---|---|---|---|")
for t in table:
    print(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]:.0f} / {t[4]:.0f} | {t[5]:.2f} / {t[6]:.2f} | {t[7]:.1f} / {t[8]:.1f} | {t[9]:.1f} | {t[10]:.0f}-{t[11]:.0f} |")

print("\n# token identity (first n-1 entries; the last entry is the stream's closing duplicate)")
first = {k: v[0] for k, v in runs.items()}
for (tag, prompt), r in sorted(first.items()):
    ref = first.get(("a4-before-gen", prompt)) or first.get(("a4-before-32k", prompt))
    self_ok = len({tuple(x["token_ids"]) for x in runs[(tag, prompt)]}) == 1
    if ref is None or ref is r:
        print(f"- {tag:16s} {prompt:8s} reps self-identical={self_ok} text64={r['text64'][:90]!r}")
        continue
    n = min(len(r["token_ids"]), len(ref["token_ids"])) - 1
    ids_ok = r["token_ids"][:n] == ref["token_ids"][:n]
    m_ok = r["margins"][:n] == ref["margins"][:n]
    first_diff = next((i for i in range(n) if r["token_ids"][i] != ref["token_ids"][i]), None)
    print(f"- {tag:16s} {prompt:8s} vs {ref['tag']}: ids identical={ids_ok} (n={n}, first diff {first_diff}) margins identical={m_ok} reps self-identical={self_ok} text64={r['text64'][:70]!r}")

print("\n# teacher-forced (64 teacher tokens from a4-before-gen)")
for r in by("tf"):
    base = f"- {r['tag']:18s} {r['prompt']:8s} chunk={r['chunk']} kv_bits={r.get('kv_bits')} prompt_tok={r['prompt_tokens']} teacher-top1={r['top1_matches_teacher']}/{r['n']} md5={r['md5'][:12]} peak={r['peak_mem_gib_mx']:.1f} GiB"
    if "kl_mean" in r:
        base += (f"\n    vs {r['reference']}: identical={r['identical']} KL mean={r['kl_mean']:.3e} max={r['kl_max']:.3e} (step {r['kl_argmax']})"
                 f" |dlogprob| mean={r['dlogprob_mean_abs']:.3e} max={r['dlogprob_max_abs']:.3e} top-1 agree={r['top1_agree']}/{r['n']}")
    print(base)
