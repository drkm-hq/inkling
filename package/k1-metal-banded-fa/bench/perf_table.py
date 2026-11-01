#!/usr/bin/env python3
# Build the op-level perf table from test-backend-ops perf logs.
# usage: perf_table.py <banded.log[,..]> <plain-fa.log[,..]> [<banded-exp.log[,..]>] [<cpu-banded.log>]  (min over repeated logs)
import re, sys

ansi = re.compile(r"\x1b\[[0-9;]*m")

def parse(paths, kind):
    # comma-separated logs of the same kind are min-merged per shape (contention only adds time)
    res = {}
    if not paths:
        return res
    lines = []
    for path in paths.split(","):
        lines += open(path, errors="replace").readlines()
    for line in lines:
        line = ansi.sub("", line)
        m = re.search(r"(\d+(?:\.\d+)?) us/run", line)
        if not m:
            continue
        us = float(m.group(1))
        if kind == "banded":
            k = re.search(r"n_q=(\d+),n_kv=(\d+)", line)
        else:
            k = re.search(r"kv=(\d+),nb=(\d+)", line)
            if k:
                key = (int(k.group(2)), int(k.group(1)))
                res[key] = min(us, res.get(key, us))
                continue
        if k:
            key = (int(k.group(1)), int(k.group(2)))
            res[key] = min(us, res.get(key, us))
    return res

args = sys.argv[1:] + [None]*4
banded = parse(args[0], "banded")
plain  = parse(args[1], "plain")
exp    = parse(args[2], "banded") if args[2] else {}
cpu    = parse(args[3], "banded") if args[3] else {}

def fmt(us):
    return "-" if us is None else (f"{us/1000:.2f} ms" if us >= 1000 else f"{us:.1f} us")

hdr = ["shape (n_q x n_kv)", "banded (Metal)", "plain FA (Metal)", "banded / plain"]
if exp:
    hdr += ["banded single-ptr", "exp / banded"]
if cpu:
    hdr += ["CPU banded", "speedup vs CPU"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] + ["---:"]*(len(hdr)-1)) + "|")
for key in sorted(banded, key=lambda k: (k[0] != 512, k[1])):
    b = banded[key]; p = plain.get(key); e = exp.get(key); c = cpu.get(key)
    row = [f"{key[0]} x {key[1]}", fmt(b), fmt(p), "-" if p is None else f"{b/p:.2f}x"]
    if exp:
        row += [fmt(e), "-" if e is None else f"{e/b:.2f}x"]
    if cpu:
        row += [fmt(c), "-" if c is None else f"{c/b:.0f}x"]
    print("| " + " | ".join(row) + " |")
