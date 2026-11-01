#!/usr/bin/env python3
# Logit-margin view of a greedy divergence: re-run one prompt with n_probs and print the top-2 candidates
# (probability and token) at a given generated position. usage: greedy_probs.py <port> <prompt-file> <position> [n_predict]
import json, sys, urllib.request
port, path, pos = sys.argv[1], sys.argv[2], int(sys.argv[3])
n_predict = int(sys.argv[4]) if len(sys.argv) > 4 else pos + 1
body = {"prompt": open(path, encoding="utf-8").read(), "n_predict": n_predict, "temperature": 0, "top_k": 1,
        "seed": 1, "n_probs": 5, "return_tokens": True, "cache_prompt": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=3600))
probs = d.get("completion_probabilities", [])
for i in range(max(0, pos - 2), min(len(probs), pos + 3)):
    top = probs[i].get("top_probs") or probs[i].get("probs") or []
    cand = [(c.get("token") or c.get("tok_str"), c.get("prob")) for c in top[:3]]
    print(f"pos {i}: chosen {d['tokens'][i]!r} top3 {cand}")
