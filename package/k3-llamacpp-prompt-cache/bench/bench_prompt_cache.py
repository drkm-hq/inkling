#!/usr/bin/env python3
"""Prompt-cache reuse measurement for llama-server on Inkling-Small (K3, server-hybrid-prompt-cache).

Starts one llama-server from --bin-dir with the campaign flags
  -np 1 --no-kv-unified -ngl 99 -fa 0 -c 70000 --jinja --temp 0 [-ub UB]
and, with the fixed prompts from bench/prompts/, measures:
  (a) p_8k then p_8k + 200 tokens with cache_prompt=true   -> prompt_n / prompt_ms / TTFT of the second request
      a2: the same extension on top of the tokens the model generated for p_8k (the agent-loop case)
  (b) a 4-turn chat on /v1/chat/completions, each turn appending the previous answer
  (c) p_8k with its last 100 tokens replaced (a rewind)
  (d) the greedy tokens of every cached request against an uncached request with the same prompt
Writes <out>/<tag>.jsonl (one row per request), <out>/<tag>-server.log and <out>/<tag>-curl.sh (the
equivalent curl commands), kills only the server it started.

Usage: bench_prompt_cache.py --bin-dir <.../build-k3/bin> --tag after-ub512 [--ub 512] [--n-predict 64]
"""
import argparse
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH = Path.home() / "code/inkling-contrib/bench"
PROMPTS = BENCH / "prompts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 2**20 if out else None


class Server:
    def __init__(self, base, curl_file):
        self.base = base
        self.curl = open(curl_file, "a")

    def _record_curl(self, path, payload, stream):
        body = json.dumps(payload)
        if isinstance(payload.get("prompt"), list) and len(payload["prompt"]) > 40:
            body = json.dumps(dict(payload, prompt=f"<{len(payload['prompt'])} token ids>"))
        flag = " -N" if stream else ""
        self.curl.write(f"curl -s{flag} {self.base}{path} -H 'Content-Type: application/json' -d {shlex.quote(body)}\n")
        self.curl.flush()

    def post(self, path, payload, timeout=1800):
        self._record_curl(path, payload, False)
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        return body, time.perf_counter() - t0

    def stream_completion(self, payload, timeout=1800):
        """POST /completion with stream=true. Returns (ttft_s, wall_s, final_chunk, tokens, content)."""
        payload = dict(payload, stream=True, return_tokens=True)
        self._record_curl("/completion", payload, True)
        req = urllib.request.Request(self.base + "/completion", data=json.dumps(payload).encode(),
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
                if t_first is None and (obj.get("content") or obj.get("tokens")):
                    t_first = time.perf_counter()
                tokens += obj.get("tokens", [])
                content += obj.get("content", "")
                # top logprobs per generated token (n_probs), to judge a greedy divergence
                for cp in obj.get("completion_probabilities", []):
                    probs.append([(x["id"], round(x["logprob"], 4)) for x in cp.get("top_logprobs", [])[:3]])
                last = obj
                if obj.get("stop"):
                    break
        return t_first - t0 if t_first else None, time.perf_counter() - t0, last, tokens, content, probs


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=70000)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--port", type=int, default=18640)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--health-timeout", type=float, default=1800.0)
    ap.add_argument("--ctx-fallback", type=int, default=36000, help="retry with this -c if the server dies during load (0 = none)")
    ap.add_argument("--quick", action="store_true", help="skip the uncached reference requests (cache behaviour only)")
    args = ap.parse_args()

    # a signal to this driver must still stop the server it started (by PID, in shutdown() below)
    def on_signal(signum, frame):
        raise SystemExit(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)

    model = args.model or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/*/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf")))[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = open(out_dir / f"{args.tag}.jsonl", "a")
    curl_file = out_dir / f"{args.tag}-curl.sh"
    server_log_path = out_dir / f"{args.tag}-server.log"

    def emit(row):
        row["_t"] = time.time()
        row["tag"] = args.tag
        rows.write(json.dumps(row) + "\n")
        rows.flush()
        log(json.dumps({k: v for k, v in row.items() if k not in ("tokens", "content", "probs")}))

    bin_dir = Path(args.bin_dir)
    # the worktree HEAD is not necessarily what the binary was built from: record both
    commit = subprocess.run(["git", "-C", str(bin_dir.parent.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    ver = subprocess.run([str(bin_dir / "llama-server"), "--version"], capture_output=True, text=True)
    build = (ver.stdout + ver.stderr).strip().splitlines()[0] if (ver.stdout + ver.stderr).strip() else "?"
    cmd = [str(bin_dir / "llama-server"), "-m", model, "-np", "1", "--no-kv-unified", "-ngl", "99", "-fa", "0",
           "-c", str(args.ctx), "--jinja", "--temp", "0", "-ub", str(args.ub),
           "--host", "127.0.0.1", "--port", str(args.port), "--no-webui", "-to", "3600", "-lv", "4"]
    with open(curl_file, "a") as f:
        f.write(f"# {time.strftime('%FT%T')} tag={args.tag} worktree_head={commit} binary={build}\n# server: {' '.join(cmd)}\n")
    emit({"kind": "meta", "model": model, "worktree_head": commit, "binary_build": build, "cmd": cmd, "ub": args.ub,
          "n_predict": args.n_predict, "load1": os.getloadavg()[0]})

    server_log = open(server_log_path, "a")

    def start(cmd):
        server_log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd)}\n")
        server_log.flush()
        p = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
        log(f"llama-server pid {p.pid}: {' '.join(cmd)}")
        return p

    proc = start(cmd)
    base = f"http://127.0.0.1:{args.port}"
    srv = Server(base, curl_file)

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
                emit({"kind": "server_exit", "returncode": proc.returncode, "ctx": args.ctx})
                if args.ctx_fallback and args.ctx != args.ctx_fallback:
                    log(f"llama-server exited early with {proc.returncode} at -c {args.ctx}; retrying with -c {args.ctx_fallback}")
                    cmd[cmd.index("-c") + 1] = str(args.ctx_fallback)
                    args.ctx = args.ctx_fallback
                    proc = start(cmd)
                    t0 = time.perf_counter()
                    continue
                raise SystemExit(f"llama-server exited early with {proc.returncode}; see {server_log_path}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if time.perf_counter() - t0 > args.health_timeout:
                raise SystemExit("llama-server did not become healthy in time")
            time.sleep(3)
        load_s = time.perf_counter() - t0
        emit({"kind": "load", "load_s": round(load_s, 1), "rss_gib": rss_gib(proc.pid)})

        def tokenize(text, add_special):
            body, _ = srv.post("/tokenize", {"content": text, "add_special": add_special})
            return body["tokens"]

        texts = {n: (PROMPTS / f"{n}.txt").read_text() for n in ["p_short", "p_2k", "p_8k"]}
        src = (PROMPTS / "src/moby-dick.txt").read_text(encoding="utf-8", errors="replace")
        # the text that follows the p_8k excerpt in the source, and an unrelated slice for the rewind
        excerpt = texts["p_8k"].split("\n\nQuestion:")[0]
        i = src.find(excerpt[-200:])
        assert i > 0, "p_8k excerpt not found in the source text"
        continuation = src[i + 200:i + 200 + 4000]
        unrelated = src[400000:404000]

        A = tokenize(texts["p_8k"], True)
        extra = tokenize("\n" + continuation, False)[:200]
        other = tokenize(unrelated, False)[:100]
        emit({"kind": "prompts", "n_A": len(A), "n_extra": len(extra), "n_other": len(other),
              "n_p_2k": len(tokenize(texts["p_2k"], True)), "n_short": len(tokenize(texts["p_short"], True))})

        def completion(name, prompt, cache_prompt, n_predict=args.n_predict, note=""):
            ttft, wall, last, tokens, content, probs = srv.stream_completion({
                "prompt": prompt, "cache_prompt": cache_prompt, "n_predict": n_predict, "temperature": 0, "n_probs": 3,
            })
            t = (last or {}).get("timings", {})
            row = {"kind": "completion", "name": name, "cache_prompt": cache_prompt, "n_prompt": len(prompt),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 3), "note": note,
                   "tokens": tokens, "content": content[:400], "probs": probs}
            emit(row)
            return row

        def compare(name, a, b):
            fd = first_diff(a["tokens"], b["tokens"])
            row = {"kind": "compare", "name": name, "first_diff": fd, "n_cached": len(a["tokens"]), "n_ref": len(b["tokens"])}
            if fd is not None and fd < min(len(a.get("probs", [])), len(b.get("probs", []))):
                row["at_diff_cached_top"] = a["probs"][fd]
                row["at_diff_ref_top"] = b["probs"][fd]
            emit(row)

        # warm-up (Metal pipelines, page cache)
        completion("warmup_p_short", tokenize(texts["p_short"], True), False, n_predict=16)

        # uncached references
        B1 = A + extra
        if not args.quick:
            r_A_ref = completion("A_ref", A, False, note="p_8k, no cache")
            r_B1_ref = completion("B1_ref", B1, False, note="p_8k + 200 tokens, no cache")

        # (a) p_8k cached, then p_8k + 200 tokens cached. The slot holds p_8k plus the tokens it generated, so
        #     the extension rewinds past them: the reuse goes through the checkpoint 4 tokens before the end of p_8k
        r_A = completion("A_cached", A, True, note="p_8k, cache_prompt=true (prime)")
        r_B1 = completion("B1_cached", B1, True, note="p_8k + 200 tokens, cache_prompt=true")
        if not args.quick:
            compare("B1_cached_vs_B1_ref", r_B1, r_B1_ref)

        # (a2) the agent-loop extension: p_8k + what the model generated for it + 200 tokens; the slot holds exactly that
        #      prefix after the prime request below (the last sampled token is not decoded, hence [:-1])
        r_A2 = completion("A_cached_2", A, True, note="p_8k, cache_prompt=true (prime for a2)")
        gen = r_A2["tokens"][:-1]
        B2 = A + gen + extra
        r_B2 = completion("B2_cached", B2, True, note="p_8k + generated + 200 tokens, cache_prompt=true (strict extension)")
        if not args.quick:
            r_B2_ref = completion("B2_ref", B2, False, note="same prompt, no cache")
            compare("B2_cached_vs_B2_ref", r_B2, r_B2_ref)

        # (c) rewind: p_8k with the last 100 tokens replaced, on top of a slot that holds p_8k + generated tokens
        r_A3 = completion("A_cached_3", A, True, note="p_8k, cache_prompt=true (prime for c)")
        C = A[:-100] + other
        r_C = completion("C_cached", C, True, note="p_8k with the last 100 tokens replaced, cache_prompt=true (rewind)")
        if not args.quick:
            r_C_ref = completion("C_ref", C, False, note="same prompt, no cache")
            compare("C_cached_vs_C_ref", r_C, r_C_ref)

        # (b) 4-turn chat, each turn appends the previous answer
        messages = [{"role": "user", "content": texts["p_2k"]}]
        follow_ups = ["Name two other characters mentioned in the text, in one sentence.",
                      "Which chapter title in the list sounds most ominous, and why, in one sentence?",
                      "Summarize the narrator's mood in one sentence."]
        for turn in range(4):
            payload = {"messages": messages, "max_tokens": args.n_predict, "temperature": 0,
                       "chat_template_kwargs": {"reasoning_effort": "none"}}
            body, wall = srv.post("/v1/chat/completions", payload)
            t = body.get("timings", {})
            answer = body["choices"][0]["message"].get("content") or ""
            emit({"kind": "chat", "turn": turn + 1, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                  "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
                  "predicted_ms": t.get("predicted_ms"), "wall_s": round(wall, 3),
                  "usage": body.get("usage"), "content": answer[:300]})
            messages.append({"role": "assistant", "content": answer})
            if turn < 3:
                messages.append({"role": "user", "content": follow_ups[turn]})

        emit({"kind": "done", "rss_gib": rss_gib(proc.pid)})
    finally:
        shutdown()
        server_log.close()


if __name__ == "__main__":
    main()
