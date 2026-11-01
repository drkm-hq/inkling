#!/usr/bin/env python3
# Greedy-output equivalence: run the prompts through a llama-server /completion endpoint and store the token ids.
# usage: greedy_check.py <port> <out-json> <prompt-file>...
import json, sys, time, urllib.request

port, out = sys.argv[1], sys.argv[2]
res = {}
for path in sys.argv[3:]:
    name = path.split("/")[-1].rsplit(".", 1)[0]
    prompt = open(path, encoding="utf-8").read()
    body = {"prompt": prompt, "n_predict": 128, "temperature": 0, "top_k": 1, "seed": 1,
            "return_tokens": True, "cache_prompt": False}
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    dt = time.time() - t0
    tm = d.get("timings", {})
    res[name] = {"tokens": d.get("tokens"), "content": d.get("content"), "wall_s": dt,
                 "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                 "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms")}
    print(f"{name}: prompt_n={tm.get('prompt_n')} prompt_ms={tm.get('prompt_ms'):.0f} "
          f"predicted_n={tm.get('predicted_n')} predicted_ms={tm.get('predicted_ms'):.0f} wall={dt:.1f}s", flush=True)
json.dump(res, open(out, "w"), indent=1)
