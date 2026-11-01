#!/usr/bin/env python3
# A1 review: greedy completion through llama-server /completion with the top-5 probabilities per step.
# usage: a1_greedy.py <port> <out-json> <n_predict> <prompt-file>...
import json, sys, time, urllib.request
port, out, n_predict = sys.argv[1], sys.argv[2], int(sys.argv[3])
res = {}
for path in sys.argv[4:]:
    name = path.split("/")[-1].rsplit(".", 1)[0]
    prompt = open(path, encoding="utf-8").read()
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 1,
            "n_probs": 5, "return_tokens": True, "cache_prompt": False}
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    dt = time.time() - t0
    tm = d.get("timings", {})
    res[name] = {"tokens": d.get("tokens"), "content": d.get("content"), "wall_s": dt,
                 "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                 "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                 "completion_probabilities": d.get("completion_probabilities")}
    print(f"{name}: prompt_n={tm.get('prompt_n')} prompt_ms={tm.get('prompt_ms'):.0f} "
          f"predicted_n={tm.get('predicted_n')} predicted_ms={tm.get('predicted_ms'):.0f} wall={dt:.1f}s", flush=True)
    print("  text:", repr((d.get("content") or "")[:400]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
