#!/usr/bin/env python3
"""A3 review: real-model (Inkling-Small UD-IQ3_XXS) prompt-cache checks on the FIXED llama-server build.

Starts one llama-server (-np 1 --no-kv-unified -ngl 99 -fa 0 -c 70000 --jinja --temp 0 -ub 512 -lv 4), then:
  (1) K3's cases on p_8k: prime, +200 extension after a prime (checkpoint path), strict extension of the generated
      tokens (live-state path), rewind of the last 100 tokens; each against the same prompt without the cache,
      64 greedy tokens with top-5 logprobs per step, the margins at the first divergence
  (2) exactness with identical ubatch splits: X = A[:8192] + 200 uncached (twice: run-to-run noise), then
      X' = X[:7876] + 516 unrelated tokens cached (restores the checkpoint at 7876 that the uncached X left, and
      evaluates the same [7876, 8388) + [8388, 8392) batches as the uncached X') -> must match bit for bit;
      and the same strict-extension request once from the live state and once through its own checkpoint
  (3) a 4-turn chat on p_2k where turn 3 edits turn 2's user text (rewind + extension), prompt_n / cache_n / TTFT
      per turn, RSS before and after
Writes <out>/<tag>.jsonl, <out>/<tag>-server.log; kills only the server it started (by PID).
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path.home() / "code/inkling-contrib/bench"
PROMPTS = BENCH / "prompts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return round(int(out) / 2**20, 3) if out else None


class Server:
    def __init__(self, base):
        self.base = base

    def post(self, path, payload, timeout=1800):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        return body, time.perf_counter() - t0

    def stream(self, path, payload, timeout=1800):
        """streamed POST; returns (ttft_s, wall_s, last_obj, tokens, content, probs[step] = [(id, logprob)]*5)"""
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        t_first = None
        tokens, content, last, probs = [], "", None, []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if path == "/completion":
                    if t_first is None and (obj.get("content") or obj.get("tokens")):
                        t_first = time.perf_counter()
                    tokens += obj.get("tokens", [])
                    content += obj.get("content", "")
                    for cp in obj.get("completion_probabilities", []):
                        probs.append([(x["id"], round(x["logprob"], 5)) for x in cp.get("top_logprobs", [])[:5]])
                    if obj.get("timings"):
                        last = obj
                    if obj.get("stop"):
                        break
                else:
                    ch = (obj.get("choices") or [{}])[0]
                    delta = ch.get("delta", {}).get("content")
                    if t_first is None and delta:
                        t_first = time.perf_counter()
                    content += delta or ""
                    if obj.get("timings"):
                        last = obj
        return (t_first - t0 if t_first else None), time.perf_counter() - t0, last, tokens, content, probs


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def margins(step_c, step_r, tok_c, tok_r):
    """margins at a divergent step: in each path's own distribution, top-1 minus the other path's token"""
    dc = dict(step_c)
    dr = dict(step_r)
    m_c = (dc[tok_c] - dc[tok_r]) if tok_r in dc else None   # how far the ref's choice is below the cached top-1
    m_r = (dr[tok_r] - dr[tok_c]) if tok_c in dr else None   # how far the cached choice is below the ref top-1
    return m_c, m_r


def max_dev(probs_a, probs_b, n):
    """max |dlogprob| over the common top-5 ids of the first n steps, and the number of steps with a different id set"""
    dev, n_idsets = 0.0, 0
    for sa, sb in zip(probs_a[:n], probs_b[:n]):
        da, db = dict(sa), dict(sb)
        common = set(da) & set(db)
        if set(da) != set(db):
            n_idsets += 1
        for t in common:
            dev = max(dev, abs(da[t] - db[t]))
    return dev, n_idsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--tag", default="a3-fixed")
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--port", type=int, default=18650)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--health-timeout", type=float, default=900.0)
    ap.add_argument("--model", default=None, help="gguf path (default: the Inkling-Small UD-IQ3_XXS first shard)")
    ap.add_argument("--ctx", type=int, default=70000)
    args = ap.parse_args()

    def on_signal(signum, frame):
        raise SystemExit(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)

    model = args.model or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/*/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf")))[0]
    out_dir = Path(args.out)
    rows = open(out_dir / f"{args.tag}.jsonl", "a")
    server_log = open(out_dir / f"{args.tag}-server.log", "a")

    def emit(row):
        row["_t"] = time.time()
        row["tag"] = args.tag
        rows.write(json.dumps(row) + "\n")
        rows.flush()
        log(json.dumps({k: v for k, v in row.items() if k not in ("tokens", "content", "probs", "probs_ref")}))

    bin_dir = Path(args.bin_dir)
    commit = subprocess.run(["git", "-C", str(bin_dir.parent.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    cmd = [str(bin_dir / "llama-server"), "-m", model, "-np", "1", "--no-kv-unified", "-ngl", "99", "-fa", "0",
           "-c", str(args.ctx), "--jinja", "--temp", "0", "-ub", str(args.ub),
           "--host", "127.0.0.1", "--port", str(args.port), "--no-webui", "-to", "3600", "-lv", "4"]
    emit({"kind": "meta", "model": model, "commit": commit, "cmd": cmd, "ub": args.ub, "n_predict": args.n_predict,
          "load1": os.getloadavg()[0]})
    server_log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd)}\n")
    server_log.flush()
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
    log(f"llama-server pid {proc.pid}")
    base = f"http://127.0.0.1:{args.port}"
    srv = Server(base)

    def shutdown():
        if proc.poll() is None:
            log("stopping llama-server")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        log(f"llama-server exit code {proc.returncode}")

    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited early with {proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if time.perf_counter() - t0 > args.health_timeout:
                raise SystemExit("llama-server did not become healthy in time")
            time.sleep(2)
        emit({"kind": "load", "load_s": round(time.perf_counter() - t0, 1), "rss_gib": rss_gib(proc.pid)})

        def tokenize(text, add_special):
            body, _ = srv.post("/tokenize", {"content": text, "add_special": add_special})
            return body["tokens"]

        texts = {n: (PROMPTS / f"{n}.txt").read_text() for n in ["p_short", "p_2k", "p_8k"]}
        src = (PROMPTS / "src/moby-dick.txt").read_text(encoding="utf-8", errors="replace")
        excerpt = texts["p_8k"].split("\n\nQuestion:")[0]
        i = src.find(excerpt[-200:])
        assert i > 0
        continuation = src[i + 200:i + 200 + 4000]
        unrelated = src[400000:404000]                 # K3's rewind text
        unrelated2 = src[i + 200 + 60000:i + 200 + 64000]   # well past the excerpt and its continuation

        A = tokenize(texts["p_8k"], True)
        if len(A) < 8192:
            # another tokenizer (dry runs on a small model): pad with later text so the 8192-based cases apply
            A = A + tokenize(src[i + 4200:i + 4200 + 80000], False)[:8192 - len(A)]
        E = tokenize("\n" + continuation, False)[:200]
        other = tokenize(unrelated, False)[:100]
        Z = tokenize(unrelated2, False)[:516]
        emit({"kind": "prompts", "n_A": len(A), "n_E": len(E), "n_other": len(other), "n_Z": len(Z),
              "n_p_2k": len(tokenize(texts["p_2k"], True))})
        assert len(A) >= 8192 and len(E) == 200 and len(Z) == 516

        def completion(name, prompt, cache_prompt, n_predict=args.n_predict, note=""):
            ttft, wall, last, tokens, content, probs = srv.stream("/completion", {
                "prompt": prompt, "cache_prompt": cache_prompt, "n_predict": n_predict, "temperature": 0,
                "n_probs": 5, "stream": True, "return_tokens": True})
            t = (last or {}).get("timings", {})
            row = {"kind": "completion", "name": name, "cache_prompt": cache_prompt, "n_prompt": len(prompt),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 3), "note": note,
                   "rss_gib": rss_gib(proc.pid), "tokens": tokens, "content": content[:300], "probs": probs}
            emit(row)
            return row

        def compare(name, a, b):
            fd = first_diff(a["tokens"], b["tokens"])
            n_cmp = fd if fd is not None else min(len(a["tokens"]), len(b["tokens"]))
            dev, n_idsets = max_dev(a["probs"], b["probs"], n_cmp)
            row = {"kind": "compare", "name": name, "first_diff": fd, "n_cached": len(a["tokens"]),
                   "n_ref": len(b["tokens"]), "max_dev_before_diff": round(dev, 5), "idset_diffs_before_diff": n_idsets}
            if fd is not None and fd < min(len(a["probs"]), len(b["probs"])):
                m_c, m_r = margins(a["probs"][fd], b["probs"][fd], a["tokens"][fd], b["tokens"][fd])
                row.update({"at_diff_cached_top5": a["probs"][fd], "at_diff_ref_top5": b["probs"][fd],
                            "tok_cached": a["tokens"][fd], "tok_ref": b["tokens"][fd],
                            "margin_in_cached": None if m_c is None else round(m_c, 4),
                            "margin_in_ref": None if m_r is None else round(m_r, 4),
                            "defect_gt_0p7": (m_c is not None and m_c > 0.7) or (m_r is not None and m_r > 0.7)})
            # the step-0 distribution is the cleanest state check: no sampling history yet
            if a["probs"] and b["probs"]:
                d0, _ = max_dev(a["probs"][:1], b["probs"][:1], 1)
                row["step0_top5_cached"] = a["probs"][0]
                row["step0_top5_ref"] = b["probs"][0]
                row["step0_max_dev"] = round(d0, 5)
            emit(row)
            return row

        # warm-up
        completion("warmup", tokenize(texts["p_short"], True), False, n_predict=16)

        # ---- (1) K3's cases on p_8k
        B1 = A + E
        r_A_ref = completion("A_ref", A, False, note="p_8k, no cache")
        r_B1_ref = completion("B1_ref", B1, False, note="p_8k + 200, no cache")
        r_A = completion("A_cached", A, True, note="prime (slot held B1_ref)")
        r_B1 = completion("B1_cached", B1, True, note="p_8k + 200 after the prime: checkpoint path")
        compare("B1_cached_vs_B1_ref", r_B1, r_B1_ref)
        r_A2 = completion("A_cached_2", A, True, note="prime for the strict extension")
        gen = r_A2["tokens"][:-1]
        B2 = A + gen + E
        r_B2 = completion("B2_cached", B2, True, note="strict extension of the live state")
        r_B2_ref = completion("B2_ref", B2, False, note="same, no cache")
        compare("B2_cached_vs_B2_ref", r_B2, r_B2_ref)
        r_A3 = completion("A_cached_3", A, True, note="prime for the rewind")
        C = A[:-100] + other
        r_C = completion("C_cached", C, True, note="rewind: last 100 tokens replaced")
        r_C_ref = completion("C_ref", C, False, note="same, no cache")
        compare("C_cached_vs_C_ref", r_C, r_C_ref)

        # ---- (2) exactness with identical ubatch splits
        # the server breaks the prompt batches 516 and 4 tokens before the end of each prompt, so two prompts of the
        # same length share their splits. X_ref2 leaves checkpoints at 7876 and 8388; X' = X[:7876] + 516 unrelated
        # tokens restores the one at 7876 and evaluates [7876, 8388) + [8388, 8392) exactly as the uncached X' does.
        A8192 = A[:8192]
        X = A8192 + E
        Xp = X[:7876] + Z
        assert len(Xp) == len(X) and Xp[7876] != X[7876]
        # the Xp reference first: the cached Xp below must find the slot holding X (rewind at 7876), not Xp
        r_Xp_ref = completion("Xp_ref", Xp, False, note="X[:7876] + 516 unrelated, no cache")
        r_X_ref = completion("X_ref", X, False, note="A[:8192] + 200, no cache")
        r_X_ref2 = completion("X_ref2", X, False, note="repeat: backend run-to-run noise; leaves checkpoints at 7876 and 8388")
        compare("X_ref2_vs_X_ref", r_X_ref2, r_X_ref)
        r_Xp = completion("Xp_cached_ckpt", Xp, True, note="rewind at 7876 -> checkpoint 7876, identical splits")
        compare("Xp_ckpt_vs_Xp_ref", r_Xp, r_Xp_ref)
        # live state vs checkpoint restore for the same request: B2 again goes through the checkpoint 4 before its end
        r_A4 = completion("A_cached_4", A, True, note="prime for the live-vs-checkpoint comparison")
        gen4 = r_A4["tokens"][:-1]
        B4 = A + gen4 + E
        r_B4 = completion("B4_cached_live", B4, True, note="strict extension from the live state")
        r_B4b = completion("B4_cached_ckpt", B4, True, note="same prompt again: the checkpoint 4 before the end")
        compare("B4_live_vs_B4_ckpt", r_B4, r_B4b)

        # ---- (3) 4-turn chat on p_2k, turn 3 edits turn 2's user text
        rss_before = rss_gib(proc.pid)
        messages = [{"role": "user", "content": texts["p_2k"]}]
        q2 = "Name two other characters mentioned in the text, in one sentence."
        q2_edit = "Which chapter title in the list sounds most ominous, and why, in one sentence?"
        q4 = "Summarize the narrator's mood in one sentence."
        chat_rows = []

        def chat(turn, note):
            payload = {"messages": messages, "max_tokens": args.n_predict, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}, "timings_per_token": True,
                       "chat_template_kwargs": {"reasoning_effort": "none"}}
            ttft, wall, last, _, content, _ = srv.stream("/v1/chat/completions", payload)
            t = (last or {}).get("timings", {})
            row = {"kind": "chat", "turn": turn, "note": note, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                   "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
                   "predicted_ms": t.get("predicted_ms"), "ttft_s": round(ttft, 3) if ttft else None,
                   "wall_s": round(wall, 3), "rss_gib": rss_gib(proc.pid), "content": content[:200]}
            emit(row)
            chat_rows.append(row)
            return content

        a1 = chat(1, "p_2k")
        messages += [{"role": "assistant", "content": a1}, {"role": "user", "content": q2}]
        a2 = chat(2, "+q2 (strict extension)")
        # turn 3: the user edits q2 -> rewind to the start of user message 2, then extension
        messages[-1] = {"role": "user", "content": q2_edit}
        a3 = chat(3, "q2 edited (rewind + extension)")
        messages += [{"role": "assistant", "content": a3}, {"role": "user", "content": q4}]
        chat(4, "+q4 (strict extension)")
        emit({"kind": "chat_rss", "rss_before_gib": rss_before, "rss_after_gib": rss_gib(proc.pid)})

        emit({"kind": "done", "rss_gib": rss_gib(proc.pid), "load1": os.getloadavg()[0]})
    finally:
        shutdown()
        server_log.close()


if __name__ == "__main__":
    main()
