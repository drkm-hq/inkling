#!/usr/bin/env python
"""Print markdown tables from the baseline result files (bench/results/*.jsonl, llama-bench-*.jsonl)."""
import glob
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def med_range(vals, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.median(vals)
    if len(vals) == 1:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} [{min(vals):.{nd}f}-{max(vals):.{nd}f}]"


def mlx_table(tag):
    path = RESULTS / f"mlx-{tag}.jsonl"
    if not path.exists():
        return f"(no {path.name})\n"
    rows = load(path)
    runs = [r for r in rows if r.get("kind") == "run"]
    out = [f"| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    # group by prompt in order of first appearance, but keep repeated p_short blocks separate
    groups = []
    for r in runs:
        key = r["prompt"]
        if groups and groups[-1][0] == key and r["rep"] > groups[-1][1][-1]["rep"]:
            groups[-1][1].append(r)
        elif groups and groups[-1][0] == key and r["rep"] == 1 and len(groups[-1][1]) >= 1 and False:
            pass
        else:
            if groups and groups[-1][0] == key and r["rep"] == 1:
                groups.append((key, [r]))
            elif groups and groups[-1][0] == key:
                groups[-1][1].append(r)
            else:
                groups.append((key, [r]))
    for key, g in groups:
        comp = sum(1 for r in g if any(c.get("gpu_suspect") for c in (r.get("competing") or []) + (r.get("competing_end") or [])))
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            key, g[0]["prompt_tokens"],
            med_range([r["prompt_tps_api"] for r in g]),
            med_range([r["ttft_wall_s"] for r in g], 2),
            med_range([r["decode_tps_wall"] for r in g], 2),
            med_range([r["generation_tps_api"] for r in g], 2),
            fmt(max(r["peak_mem_gb_api"] for r in g)),
            med_range([r["load1"] for r in g]),
            f"{comp}/{len(g)} runs" if any("competing" in r for r in g) else "n/r",
            ",".join(str(r.get("first_eos_pos")) for r in g),
            "ok" if all(r["answer_ok"] for r in g) else "NO (" + ",".join("y" if r["answer_ok"] else "n" for r in g) + ")",
            "yes" if len({tuple(r["first64_ids"]) for r in g}) == 1 else "no",
        ))
    extra = [r for r in rows if r.get("kind") in ("aborted", "warning", "skipped")]
    for r in extra:
        out.append(f"\n{r['kind']}: {r.get('reason') or r.get('msg') or r.get('prompt')}")
    load_rows = [r for r in rows if r.get("kind") == "load"]
    if load_rows:
        lr = load_rows[-1]
        out.append(f"\nload: lazy {lr['lazy_s']:.1f} s + chunked eval {lr['eval_s']:.1f} s, {lr['param_gib']:.1f} GiB parameters")
    end = [r for r in rows if r.get("kind") == "end"]
    if end:
        out.append(f"process maxrss {end[-1]['maxrss_gib']:.1f} GiB (ru_maxrss; excludes most Metal buffers)")
    cache = [r.get("cache_types") for r in runs if r.get("cache_types")]
    if cache:
        out.append(f"cache after last run: {cache[-1]}")
    return "\n".join(out) + "\n"


def mlx_first64(tag):
    path = RESULTS / f"mlx-{tag}.jsonl"
    if not path.exists():
        return ""
    out = []
    seen = set()
    for r in load(path):
        if r.get("kind") == "run" and r["prompt"] not in seen:
            seen.add(r["prompt"])
            out.append(f"- {r['prompt']} (rep {r['rep']}): `{r['first64']!r}`")
    return "\n".join(out) + "\n"


def server_table(tag):
    path = RESULTS / f"llama-server-{tag}.jsonl"
    if not path.exists():
        return f"(no {path.name})\n"
    rows = load(path)
    runs = [r for r in rows if r.get("kind") in ("run", "run_single")]
    out = ["| prompt | prompt_n | prefill tok/s (server) | prompt_ms | TTFT s (wall) | decode tok/s (server) | total s | RSS GiB | load1 | answer | identical |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    names = []
    for r in runs:
        if r["prompt"] not in names:
            names.append(r["prompt"])
    for n in names:
        g = [r for r in runs if r["prompt"] == n and not r.get("aborted")]
        bad = [r for r in runs if r["prompt"] == n and r.get("aborted")]
        if not g:
            out.append(f"| {n} | - | aborted: {bad[0]['aborted'] if bad else '?'} | | | | | | | | |")
            continue
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            n, g[0].get("prompt_n"),
            med_range([r.get("prompt_tps") for r in g]),
            med_range([r.get("prompt_ms") for r in g], 0),
            med_range([r.get("ttft_wall_s") for r in g], 2),
            med_range([r.get("decode_tps") for r in g], 2),
            med_range([r.get("total_wall_s") for r in g]),
            med_range([r.get("rss_gib") for r in g]),
            med_range([r.get("load1") for r in g]),
            "ok" if all(r["answer_ok"] for r in g) else "NO",
            "yes" if len({r["content"][:200] for r in g}) == 1 else "no",
        ) + (f" aborted {len(bad)}" if bad else ""))
    for r in rows:
        if r.get("kind") in ("load", "template_check", "tokenize_check", "prefix_summary", "end"):
            out.append(f"\n{r['kind']}: " + json.dumps({k: v for k, v in r.items() if k not in ('kind', '_t', 'tag', 'fa')}))
    return "\n".join(out) + "\n"


def server_first(tag):
    path = RESULTS / f"llama-server-{tag}.jsonl"
    if not path.exists():
        return ""
    out, seen = [], set()
    for r in load(path):
        if r.get("kind") in ("run", "run_single", "prefix_a_first", "prefix_b_extended") and (r["prompt"], r["kind"]) not in seen:
            seen.add((r["prompt"], r["kind"]))
            out.append(f"- {r['prompt']} ({r['kind']}): `{r.get('content','')[:300]!r}`")
    return "\n".join(out) + "\n"


def bench_table(tag):
    path = RESULTS / f"llama-bench-{tag}.jsonl"
    if not path.exists():
        return f"(no {path.name})\n"
    rows = load(path)
    if not rows:
        return f"({path.name} is empty)\n"
    out = ["| test | depth | fa | t/s avg | t/s stddev | samples |", "|---|---|---|---|---|---|"]
    for r in rows:
        test = f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}"
        out.append(f"| {test} | {r.get('n_depth')} | {r.get('flash_attn')} | {r.get('avg_ts', 0):.2f} | {r.get('stddev_ts', 0):.2f} | {len(r.get('samples_ns', []))} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for tag in ("baseline", "rerun", "kv8", "kv8-rerun", "p32k-clean"):
        if "all" in which or "mlx" in which:
            print(f"\n### MLX {tag}\n")
            print(mlx_table(tag))
            print(mlx_first64(tag))
    for tag in ("fa0", "fa1", "fa1-d16k"):
        if "all" in which or "bench" in which:
            print(f"\n### llama-bench {tag}\n")
            print(bench_table(tag))
    for tag in ("fa0", "fa1", "fa0-c16k", "fa1-c16k"):
        if "all" in which or "server" in which:
            print(f"\n### llama-server {tag}\n")
            print(server_table(tag))
            print(server_first(tag))
