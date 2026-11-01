"""Diff two equiv.py outputs: identical token sequences per run, or the first divergence
with both implementations' top-8 logits at that step (near-tie evidence).

  python equiv_diff.py ../results/K2-equiv-mlx_vlm.json ../results/K2-equiv-mlx_lm.json
"""
import json
import sys


def main(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A = {a['impl']} {a['version']} | B = {b['impl']} {b['version']} | mlx {a['mlx']}")
    all_same = True
    for ra, rb in zip(a["runs"], b["runs"]):
        assert ra["name"] == rb["name"], (ra["name"], rb["name"])
        assert ra["prompt_tokens"] == rb["prompt_tokens"], "prompt tokenization differs"
        assert ra["prompt_tail_ids"] == rb["prompt_tail_ids"], "prompt tokenization differs"
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        if div is None:
            print(f"{ra['name']:20s} prompt {ra['prompt_tokens']:6d} tok: IDENTICAL {n} tokens; "
                  f"min top-2 margin A {min(ra['margins']):.3f} B {min(rb['margins']):.3f}; "
                  f"max |top1 logit diff| {max(abs(x[0][1] - y[0][1]) for x, y in zip(ra['top8'], rb['top8'])):.4f}")
            continue
        all_same = False
        print(f"{ra['name']:20s} prompt {ra['prompt_tokens']:6d} tok: DIVERGE at step {div} "
              f"(A {ta[div]} vs B {tb[div]}); agree on {div} tokens before")
        for label, r, other_tok in (("A", ra, tb[div]), ("B", rb, ta[div])):
            top = r["top8"][div]
            mine = top[0]
            theirs = next((p for p in top if p[0] == other_tok), None)
            print(f"   {label} top-8: {[(i, round(v, 3)) for i, v in top]}")
            if theirs is not None:
                print(f"   {label}: chosen {mine[0]} logit {mine[1]:.4f}, the other's token {theirs[0]} "
                      f"logit {theirs[1]:.4f}, gap {mine[1] - theirs[1]:.4f}")
            else:
                print(f"   {label}: the other's token {other_tok} is not in {label}'s top-8")
        print(f"   text A: {ra['text'][:160]!r}")
        print(f"   text B: {rb['text'][:160]!r}")
    print("RESULT:", "all runs identical" if all_same else "divergence found")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
