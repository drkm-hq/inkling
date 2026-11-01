#!/usr/bin/env python3
# A1 review: compare llama-eval-callback dumps (first/last 3 values per dim + sum of every graph node) across runs.
# usage: a1_evalcb_compare.py <ref.log> <a.log> <b.log> [name-regex]
import re, sys, math
def parse(path):
    tensors = {}; name = None; vals = []
    hdr = re.compile(r'^common_debug_cb_eval:\s+(\S+) = \((\w+)\)\s+(\S+)\(')
    for line in open(path, errors='replace'):
        m = hdr.match(line)
        if m:
            name = m.group(1); vals = []; tensors[name] = {'op': m.group(3), 'vals': vals, 'sum': None}
            continue
        if name is None: continue
        if line.strip().startswith('sum ='):
            tensors[name]['sum'] = float(line.split('=')[1]); continue
        for tok in re.findall(r'-?\d+\.\d+', line):
            vals.append(float(tok))
    return tensors
ref = parse(sys.argv[1]); A = parse(sys.argv[2]); B = parse(sys.argv[3])
pat = re.compile(sys.argv[4]) if len(sys.argv) > 4 else re.compile(r'^(kqv_out|inkling_attn_q_norm|inkling_rel_logits|inkling_attn_k_sconv|attn_out|ffn_out|l_out|result_output)-?\d*$')
def rel(a, b):
    if not a or not b or len(a) != len(b): return None
    den = max(1e-6, max(abs(x) for x in b))
    return max(abs(x - y) for x, y in zip(a, b)) / den
print(f"{'tensor':28s} {'op':14s} {'kernel/ref':>11s} {'dense/ref':>11s} {'kernel/dense':>13s} {'sum ref':>14s} {'sum kernel':>14s} {'sum dense':>14s}")
for name in ref:
    if not pat.search(name) or name not in A or name not in B: continue
    r, a, b = ref[name], A[name], B[name]
    ra, rb, rab = rel(a['vals'], r['vals']), rel(b['vals'], r['vals']), rel(a['vals'], b['vals'])
    f = lambda x: '-' if x is None else f"{x:.2e}"
    g = lambda s: '-' if s is None else f"{s:.4f}"
    print(f"{name:28s} {r['op']:14s} {f(ra):>11s} {f(rb):>11s} {f(rab):>13s} {g(r['sum']):>14s} {g(a['sum']):>14s} {g(b['sum']):>14s}")
