"""Markdown tables from the benchmark jsonl files: mlx-vlm baseline (bench/baseline/bench_mlx.py)
and the mlx-lm runs (bench_mlx_lm.py). Medians of the timed runs per prompt.

  python summarize.py ../results/mlx-baseline.jsonl ../results/K2-mlxlm-2bit.jsonl ../results/K2-mlxlm-attn8.jsonl
"""
import json
import statistics
import sys


def load_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    runs = [r for r in rows if r.get("kind") == "run"]
    meta = next((r for r in rows if r.get("kind") == "meta"), {})
    load = next((r for r in rows if r.get("kind") == "load"), {})
    return meta, load, runs


def med(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def spread(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    if len(vals) < 2:
        return ""
    return f" ({min(vals):.1f}-{max(vals):.1f})"


def main(paths):
    engines = []
    for p in paths:
        meta, load, runs = load_rows(p)
        tag = meta.get("tag", p)
        engine = "mlx-vlm " + meta["mlx_vlm"] if "mlx_vlm" in meta else "mlx-lm " + meta.get("mlx_lm", "?")
        engines.append((tag, engine, load, runs))
    print("| engine | tag | load (lazy + eval) | params GiB |")
    print("|---|---|---|---|")
    for tag, engine, load, runs in engines:
        print(f"| {engine} | {tag} | {load.get('lazy_s', 0):.1f} + {load.get('eval_s', 0):.1f} s | {load.get('param_gib', 0):.1f} |")
    prompts = []
    for _, _, _, runs in engines:
        for r in runs:
            if r["prompt"] not in prompts:
                prompts.append(r["prompt"])
    print()
    print("| prompt | tokens | engine (tag) | prefill tok/s | TTFT s | decode tok/s (wall) | decode tok/s (api) | peak GiB | load1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for prompt in prompts:
        for tag, engine, load, runs in engines:
            rs = [r for r in runs if r["prompt"] == prompt]
            if not rs:
                continue
            print(
                f"| {prompt} | {rs[0]['prompt_tokens']} | {engine} ({tag}) | "
                f"{med(rs, 'prompt_tps_api'):.0f}{spread(rs, 'prompt_tps_api')} | "
                f"{med(rs, 'ttft_wall_s'):.2f} | "
                f"{med(rs, 'decode_tps_wall'):.2f}{spread(rs, 'decode_tps_wall')} | "
                f"{med(rs, 'generation_tps_api'):.2f} | "
                f"{med(rs, 'peak_mem_gib_mx'):.1f} | {med(rs, 'load1'):.0f} |"
            )
    print()
    for tag, engine, load, runs in engines:
        for prompt in prompts:
            rs = [r for r in runs if r["prompt"] == prompt]
            if rs:
                print(f"{engine} ({tag}) {prompt} first64: {rs[0]['first64']!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
