#!/usr/bin/env python3
# A1 review: compare two greedy runs (a1_greedy.py JSON) token by token; at the first divergence print both sides'
# top-5 with the logprob margin between the chosen token and the runner-up. usage: a1_compare.py <A.json> <B.json> [labelA] [labelB]
import json, math, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
la = sys.argv[3] if len(sys.argv) > 3 else "A"; lb = sys.argv[4] if len(sys.argv) > 4 else "B"

def top(step):
    cands = step.get("top_logprobs") or step.get("top_probs") or []
    out = []
    for c in cands:
        if "logprob" in c:
            lp = c["logprob"]
        else:
            lp = math.log(max(c.get("prob", 0.0), 1e-30))
        out.append((c.get("id"), c.get("token"), lp))
    return out

for k in a:
    if k not in b:
        continue
    ta, tb = a[k]["tokens"], b[k]["tokens"]
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    print(f"== {k}: prompt_n={a[k].get('prompt_n')} {la}:{len(ta)} tok {lb}:{len(tb)} tok; "
          f"{la} decode {a[k].get('predicted_n')}/{a[k].get('predicted_ms'):.0f}ms "
          f"({1000.0*a[k]['predicted_n']/a[k]['predicted_ms']:.2f} tok/s), {lb} {b[k].get('predicted_n')}/{b[k].get('predicted_ms'):.0f}ms "
          f"({1000.0*b[k]['predicted_n']/b[k]['predicted_ms']:.2f} tok/s); prefill {la} {1000.0*a[k]['prompt_n']/a[k]['prompt_ms']:.1f} tok/s, {lb} {1000.0*b[k]['prompt_n']/b[k]['prompt_ms']:.1f} tok/s")
    if div is None:
        print(f"   IDENTICAL for {n} tokens" + ("" if len(ta) == len(tb) else f" (lengths differ: {len(ta)} vs {len(tb)})"))
    else:
        print(f"   DIVERGE at token {div}")
    # per-step margin statistics on the common prefix, from each side's own top-5
    pa, pb = a[k].get("completion_probabilities") or [], b[k].get("completion_probabilities") or []
    m = min(len(pa), len(pb), n if div is None else div + 1)
    worst = []
    for i in range(m):
        ca, cb = top(pa[i]), top(pb[i])
        if len(ca) >= 2 and len(cb) >= 2:
            worst.append((min(ca[0][2] - ca[1][2], cb[0][2] - cb[1][2]), i))
        # cross-check: does side B rank A's chosen token first, and vice versa?
    worst.sort()
    print("   smallest top1-top2 logprob margins on the common prefix (margin, step):", [(round(w, 4), i) for w, i in worst[:5]])
    # the logprob the other side assigns to this side's token at every common step: max abs diff of the chosen token's logprob
    dmax = (0.0, -1)
    for i in range(m):
        ca, cb = top(pa[i]), top(pb[i])
        da = {c[0]: c[2] for c in ca}; db = {c[0]: c[2] for c in cb}
        if ta[i] in da and ta[i] in db:
            d = abs(da[ta[i]] - db[ta[i]])
            if d > dmax[0]:
                dmax = (d, i)
    print(f"   max |logprob diff| of the chosen token over {m} steps: {dmax[0]:.4f} at step {dmax[1]}")
    if div is not None:
        for lab, p, t in ((la, pa, ta), (lb, pb, tb)):
            if div < len(p):
                c = top(p[div])
                print(f"   {lab} step {div}: chosen id={t[div]} top5=" + ", ".join(f"({i},{repr(s)},{lp:.4f})" for i, s, lp in c[:5]))
        ca, cb = top(pa[div]) if div < len(pa) else [], top(pb[div]) if div < len(pb) else []
        da = {c[0]: c[2] for c in ca}; db = {c[0]: c[2] for c in cb}
        if ta[div] in da and tb[div] in da and ta[div] in db and tb[div] in db:
            print(f"   margins at the divergence: {la} prefers {ta[div]} over {tb[div]} by {da[ta[div]] - da[tb[div]]:.4f}; "
                  f"{lb} prefers {tb[div]} over {ta[div]} by {db[tb[div]] - db[ta[div]]:.4f}")
        print(f"   {la} text: {a[k]['content'][:300]!r}")
        print(f"   {lb} text: {b[k]['content'][:300]!r}")
