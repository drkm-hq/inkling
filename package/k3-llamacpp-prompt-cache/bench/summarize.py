#!/usr/bin/env python3
"""Tabulate the before/after prompt-cache runs (results/<tag>.jsonl) as markdown."""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TAGS = sys.argv[1:] or ["before-ub512", "after-ub512", "before-ub256", "after-ub256"]


def load(tag):
    rows = []
    p = RESULTS / f"{tag}.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def last(rows, kind, name=None, **kw):
    out = None
    for r in rows:
        if r.get("kind") != kind:
            continue
        if name is not None and r.get("name") != name:
            continue
        if any(r.get(k) != v for k, v in kw.items()):
            continue
        out = r
    return out


def fmt_ms(v):
    return "-" if v is None else f"{v/1000:.2f} s"


def main():
    data = {t: load(t) for t in TAGS}
    tags = [t for t in TAGS if data[t]]
    if not tags:
        print("no results yet")
        return

    for t in tags:
        meta = last(data[t], "meta") or {}
        load_row = last(data[t], "load") or {}
        pr = last(data[t], "prompts") or {}
        head = meta.get("worktree_head") or meta.get("commit") or "?"
        print(f"- `{t}`: worktree `{head[:9]}`, binary `{meta.get('binary_build', 'see the server log')}`, "
              f"-ub {meta.get('ub')}, n_predict {meta.get('n_predict')}, load1 at start {meta.get('load1', 0):.1f}, "
              f"model load {load_row.get('load_s', '?')} s, rss {load_row.get('rss_gib', 0):.1f} GiB, "
              f"p_8k = {pr.get('n_A')} tokens, extension = {pr.get('n_extra')} tokens, p_2k = {pr.get('n_p_2k')} tokens")
    print()

    names = [("A_ref", "p_8k, no cache (reference)"),
             ("A_cached", "p_8k, cache on, slot held p_8k+200+gen"),
             ("B1_cached", "(a) p_8k + 200 tokens, cache on"),
             ("B1_ref", "(a) same, no cache"),
             ("B2_cached", "(a2) p_8k + generated + 200, cache on"),
             ("B2_ref", "(a2) same, no cache"),
             ("C_cached", "(c) p_8k, last 100 tokens replaced, cache on"),
             ("C_ref", "(c) same, no cache")]
    print("| request | " + " | ".join(f"{t} prompt_n / cache_n / prompt_ms / TTFT" for t in tags) + " |")
    print("|---|" + "---|" * len(tags))
    for name, desc in names:
        cells = []
        for t in tags:
            r = last(data[t], "completion", name)
            if not r:
                cells.append("-")
            else:
                cells.append(f"{r['prompt_n']} / {r['cache_n']} / {fmt_ms(r['prompt_ms'])} / {r['ttft_s']} s")
        print(f"| {desc} | " + " | ".join(cells) + " |")
    print()

    print("| greedy tokens, cached vs uncached | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for cmp in ["B1_cached_vs_B1_ref", "B2_cached_vs_B2_ref", "C_cached_vs_C_ref"]:
        cells = []
        for t in tags:
            r = last(data[t], "compare", cmp)
            if not r:
                cells.append("-")
            else:
                fd = r["first_diff"]
                cells.append(f"identical ({r['n_cached']} tokens)" if fd is None else f"differ at token {fd} of {r['n_cached']}/{r['n_ref']}")
        print(f"| {cmp} | " + " | ".join(cells) + " |")
    print()

    print("| chat turn | " + " | ".join(f"{t} prompt_n / cache_n / prompt_ms" for t in tags) + " |")
    print("|---|" + "---|" * len(tags))
    for turn in range(1, 5):
        cells = []
        for t in tags:
            r = last(data[t], "chat", turn=turn)
            cells.append("-" if not r else f"{r['prompt_n']} / {r['cache_n']} / {fmt_ms(r['prompt_ms'])}")
        print(f"| turn {turn} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
