#!/usr/bin/env python3
"""Pair the completion rows of a3_gpu_run.py / a3_gpu_noise.py and print divergence statistics.

For every pair (a, b) of rows: the first differing greedy token, the margins at that step (in each path's own
top-5: top-1 minus the other path's token), the step-0 max |dlogprob| on the common top-5 ids, the mean/max
|dlogprob| over the steps before the divergence, and a KL(a||b) estimate over the common top-5 ids (renormalised)
averaged over the steps before the divergence. Usage: a3_gpu_compare.py main.jsonl [noise.jsonl]
"""
import json
import math
import sys


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "completion" and r.get("probs"):
            rows[r["name"]] = r
    return rows


def stats(a, b, label):
    ta, tb = a["tokens"], b["tokens"]
    fd = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    n = fd if fd is not None else min(len(ta), len(tb))
    devs, kls = [], []
    for sa, sb in zip(a["probs"][:n], b["probs"][:n]):
        da, db = dict(sa), dict(sb)
        common = set(da) & set(db)
        if not common:
            continue
        devs.append(max(abs(da[t] - db[t]) for t in common))
        za = sum(math.exp(da[t]) for t in common)
        zb = sum(math.exp(db[t]) for t in common)
        kls.append(sum(math.exp(da[t]) / za * ((da[t] - math.log(za)) - (db[t] - math.log(zb))) for t in common))
    d0 = devs[0] if devs else None
    line = (f"{label:34s} first_diff={str(fd):4s} ({len(ta)}/{len(tb)} tok) step0_dev={d0 if d0 is None else round(d0, 3)} "
            f"pre-div mean|d|={round(sum(devs) / len(devs), 3) if devs else None} max|d|={round(max(devs), 3) if devs else None} "
            f"KL5={round(sum(kls) / len(kls), 4) if kls else None}")
    if fd is not None and fd < min(len(a["probs"]), len(b["probs"])):
        da, db = dict(a["probs"][fd]), dict(b["probs"][fd])
        tc, tr = ta[fd], tb[fd]
        mc = da[tc] - da[tr] if tr in da else None
        mr = db[tr] - db[tc] if tc in db else None
        line += (f"\n{'':34s} at diff: a={tc} b={tr} margin_in_a={None if mc is None else round(mc, 3)} "
                 f"margin_in_b={None if mr is None else round(mr, 3)}  a_top5={a['probs'][fd][:3]} b_top5={b['probs'][fd][:3]}")
    print(line)


def main():
    main_rows = load(sys.argv[1])
    noise_rows = load(sys.argv[2]) if len(sys.argv) > 2 else {}
    R = main_rows
    pairs = [
        ("X_ref2", "X_ref", "run-to-run, identical splits"),
        ("Xp_cached_ckpt", "Xp_ref", "checkpoint 7876, identical splits"),
        ("B4_cached_ckpt", "B4_cached_live", "same request: ckpt vs live"),
        ("C_cached", "C_ref", "rewind via checkpoint (K3 c)"),
        ("B1_cached", "B1_ref", "+200 via checkpoint 8214 (K3 a)"),
        ("B2_cached", "B2_ref", "strict extension, live state (K3 a2)"),
        ("B4_cached_live", "B4_ref_like", None),
        ("A_cached", "A_ref", "same prompt A via checkpoint 7902"),
        ("A_cached_2", "A_ref", "same prompt A via checkpoint 8214"),
        ("A_cached_3", "A_ref", "same prompt A via checkpoint 7965"),
        ("A_cached_4", "A_ref", "same prompt A via a checkpoint"),
    ]
    print("== main run (fixed build, -ub 512): cached vs uncached and exactness pairs")
    for a, b, label in pairs:
        if label and a in R and b in R:
            stats(R[a], R[b], label)
    if noise_rows:
        print("== noise floor: uncached at another ubatch vs uncached -ub 512 (no cache involved)")
        for a, b, label in [("A_ref", "A_ref", "A: -ub noise vs -ub 512"), ("B1_ref", "B1_ref", "B1: -ub noise vs -ub 512"),
                            ("X_ref", "X_ref", "X: -ub noise vs -ub 512")]:
            if a in noise_rows and b in R:
                stats(noise_rows[a], R[b], label)
        if "X4_ref" in noise_rows and "X_ref" in noise_rows:
            # X4 = X + the first 4 generated tokens of X (main run); compare its generation with X_ref's tokens[4:]
            x4, x = noise_rows["X4_ref"], noise_rows["X_ref"]
            shifted = {"tokens": x["tokens"][4:], "probs": x["probs"][4:]}
            stats(x4, shifted, "X+4 vs X[4:] (same -ub, other breaks)")
            if "X_ref" in R:
                shifted_main = {"tokens": R["X_ref"]["tokens"][4:], "probs": R["X_ref"]["probs"][4:]}
                stats(x4, shifted_main, "X+4 (-ub noise) vs X[4:] (-ub 512)")


if __name__ == "__main__":
    main()
