#!/usr/bin/env python
"""llama-server /completion benchmark for Inkling-Small GGUF on Metal (bench/PROTOCOL.md).

Starts llama-server (PR #25731 build), waits for /health, sends the fixed prompts rendered
with the Inkling chat template (reasoning_effort=none, identical bytes to the MLX run) to
/completion with stream=true (TTFT = first SSE chunk), n_predict small, temperature 0,
ignore_eos, cache_prompt=false; 1 warm-up + N timed runs per prompt, then a prefix-cache
test; records the server's own timings (prompt_ms / predicted_ms) and wall clock, the
first 64 generated tokens, resident memory, and kills the server it started.

Usage (from bench/):
  python baseline/bench_llama_server.py --fa 0 --tag fa0 --prompts p_short,p_2k,p_8k,p_32k --repeats 3
  python baseline/bench_llama_server.py --fa 1 --tag fa1 --prompts p_short,p_2k,p_8k --repeats 3 --single p_32k
"""
import argparse
import glob
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
RESULTS = BENCH / "results"
PROMPTS = BENCH / "prompts"
SRC = Path(os.environ.get("LLAMA_SRC", str(Path.home() / "code/inkling-contrib/upstream/llama.cpp-base")))
BIN = Path(os.environ.get("LLAMA_BIN", str(SRC / "build/bin")))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 2**20 if out else None


def http_json(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_completion(url, payload, timeout):
    """POST with stream=true; returns (first_chunk_wall, last_json, content, aborted)."""
    payload = dict(payload, stream=True)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    content = ""
    last = None
    aborted = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if t_first is None and obj.get("content"):
                    t_first = time.perf_counter()
                content += obj.get("content", "")
                last = obj
                if time.perf_counter() - t0 > timeout:
                    aborted = f"timeout {timeout}s"
                    break
    except (socket.timeout, TimeoutError) as e:
        aborted = f"socket timeout: {e}"
    except urllib.error.HTTPError as e:
        aborted = f"http {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        aborted = f"{type(e).__name__}: {e}"
    return t0, t_first, time.perf_counter(), last, content, aborted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", required=True, choices=["0", "1"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="p_short,p_2k,p_8k,p_32k")
    ap.add_argument("--single", default="", help="comma list of prompts to run once (no warm-up, 1 timed run)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=70000)
    ap.add_argument("--ctx-fallback", type=int, default=36000, help="retry with this -c if the server dies during load (0 = no retry)")
    ap.add_argument("--port", type=int, default=18611)
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-request bound (s)")
    ap.add_argument("--health-timeout", type=float, default=1500.0)
    ap.add_argument("--no-prefix-test", action="store_true")
    ap.add_argument("--extra", default="", help="extra llama-server args (space separated)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokenizer", default=None, help="HF dir with the Inkling chat template (default: the MLX snapshot)")
    args = ap.parse_args()

    model = args.model or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/*/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf")))[0]
    tok_dir = args.tokenizer or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    RESULTS.mkdir(exist_ok=True)
    out = open(RESULTS / f"llama-server-{args.tag}.jsonl", "a")

    def emit(row):
        row["_t"] = time.time()
        out.write(json.dumps(row) + "\n")
        out.flush()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)

    def render(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort="none")

    names = [p for p in args.prompts.split(",") if p]
    singles = [p for p in args.single.split(",") if p]
    texts = {n: (PROMPTS / f"{n}.txt").read_text() for n in dict.fromkeys(names + singles + ["p_8k"])}
    prompts = {n: render(t) for n, t in texts.items()}
    counts = {n: len(tok.encode(f, add_special_tokens=False)) for n, f in prompts.items()}

    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    version = subprocess.run([str(BIN / "llama-server"), "--version"], capture_output=True, text=True)
    version = (version.stdout + version.stderr).strip().splitlines()[-2:]
    cmd = [str(BIN / "llama-server"), "-m", model, "-np", "1", "--no-kv-unified", "-ngl", "99", "-c", str(args.ctx),
           "--jinja", "--temp", "0", "--chat-template-kwargs", '{"reasoning_effort":"none"}', "-fa", args.fa,
           "--host", "127.0.0.1", "--port", str(args.port), "--no-webui", "-to", "3600", "--metrics"] + args.extra.split()
    meta = {"kind": "meta", "tag": args.tag, "fa": args.fa, "model": model, "commit": commit, "version": version, "src": str(SRC), "cmd": cmd,
            "prompt_tokens_hf": counts, "load1": os.getloadavg()[0]}
    log(json.dumps(meta))
    emit(meta)

    server_log = open(RESULTS / f"llama-server-{args.tag}.log", "a")
    base = f"http://127.0.0.1:{args.port}"

    def start(cmd):
        server_log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd)}\n")
        server_log.flush()
        p = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
        log(f"llama-server pid {p.pid}: {' '.join(cmd)}")
        return p

    proc = start(cmd)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda *_: (_ for _ in ()).throw(SystemExit(143)))

    def shutdown():
        if proc.poll() is None:
            log("stopping llama-server")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(60)
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
                emit({"kind": "server_exit", "tag": args.tag, "returncode": proc.returncode, "ctx": args.ctx})
                if args.ctx_fallback and args.ctx != args.ctx_fallback:
                    log(f"llama-server exited early with {proc.returncode} at -c {args.ctx}; retrying with -c {args.ctx_fallback}")
                    cmd[cmd.index("-c") + 1] = str(args.ctx_fallback)
                    args.ctx = args.ctx_fallback
                    proc = start(cmd)
                    t0 = time.perf_counter()
                    continue
                raise SystemExit(f"llama-server exited early with {proc.returncode}; see the server log")
            try:
                h = http_json(base + "/health", timeout=5)
                if h.get("status") == "ok":
                    break
            except Exception:
                pass
            if time.perf_counter() - t0 > args.health_timeout:
                raise SystemExit("llama-server did not become healthy in time")
            time.sleep(3)
        load_s = time.perf_counter() - t0
        log(f"server healthy after {load_s:.1f}s; rss {rss_gib(proc.pid):.1f} GiB")
        emit({"kind": "load", "tag": args.tag, "load_s": load_s, "rss_gib": rss_gib(proc.pid)})

        # Does the server's own template render the same bytes? (/apply-template; informational)
        try:
            at = http_json(base + "/apply-template", {"messages": [{"role": "user", "content": texts["p_short"]}],
                                                       "chat_template_kwargs": {"reasoning_effort": "none"}}, timeout=30)
            same = at.get("prompt") == prompts["p_short"]
            log(f"/apply-template matches our rendering: {same}; server head={at.get('prompt','')[:100]!r}")
            emit({"kind": "template_check", "tag": args.tag, "same": same, "server_head": at.get("prompt", "")[:200]})
        except Exception as e:  # noqa: BLE001
            log(f"/apply-template check failed: {e}")

        # Tokenizer parity: server token count vs HF count for p_short
        try:
            tk = http_json(base + "/tokenize", {"content": prompts["p_short"], "add_special": True, "with_pieces": False}, timeout=30)
            log(f"/tokenize p_short: server {len(tk['tokens'])} tokens vs HF {counts['p_short']}")
            emit({"kind": "tokenize_check", "tag": args.tag, "server": len(tk["tokens"]), "hf": counts["p_short"]})
        except Exception as e:  # noqa: BLE001
            log(f"/tokenize check failed: {e}")

        def one(name, rep, kind, prompt=None, cache_prompt=False, n_predict=None):
            text = prompt if prompt is not None else prompts[name]
            payload = {"prompt": text, "n_predict": n_predict or args.n_predict, "temperature": 0.0,
                       "cache_prompt": cache_prompt, "ignore_eos": True, "n_probs": 0}
            t_start, t_first, t_end, last, content, aborted = stream_completion(base + "/completion", payload, args.timeout)
            row = {"kind": kind, "tag": args.tag, "fa": args.fa, "prompt": name, "rep": rep, "cache_prompt": cache_prompt,
                   "total_wall_s": t_end - t_start, "ttft_wall_s": (t_first - t_start) if t_first else None,
                   "aborted": aborted, "load1": os.getloadavg()[0], "rss_gib": rss_gib(proc.pid)}
            if last:
                tm = last.get("timings", {}) or {}
                row.update({
                    "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                    "prompt_tps": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
                    "predicted_ms": tm.get("predicted_ms"), "decode_tps": tm.get("predicted_per_second"),
                    "tokens_cached": last.get("tokens_cached"), "tokens_evaluated": last.get("tokens_evaluated"),
                    "tokens_predicted": last.get("tokens_predicted"), "stop_type": last.get("stop_type"),
                    "truncated": last.get("truncated"), "timings_raw": tm,
                })
            row["content"] = content
            row["answer_ok"] = ("Ishmael" in content) and ("Pequod" in content)
            emit(row)
            if last:
                def r2(v, nd=2):
                    return "-" if v is None else round(v, nd)
                log(f"{kind} {name} rep{rep} cache={cache_prompt}: prompt_n {row.get('prompt_n')} prompt {r2((row.get('prompt_ms') or 0)/1000)}s "
                    f"({r2(row.get('prompt_tps'), 1)} tok/s) TTFT wall {r2(row.get('ttft_wall_s'))}s | "
                    f"decode {row.get('predicted_n')} tok {r2(row.get('decode_tps'))} tok/s | cached {row.get('tokens_cached')} | "
                    f"total {row['total_wall_s']:.1f}s | rss {row['rss_gib']} GiB | ok={row['answer_ok']} aborted={aborted} | stop={row.get('stop_type')}")
            else:
                log(f"{kind} {name} rep{rep}: NO RESPONSE aborted={aborted} after {row['total_wall_s']:.1f}s")
            log(f"  first chars: {content[:300]!r}")
            return row

        def summarize(name, rows):
            good = [r for r in rows if not r.get("aborted") and r.get("prompt_n")]
            if not good:
                return
            def stat(k):
                v = [r[k] for r in good if r.get(k) is not None]
                if not v:
                    return None
                m = statistics.median(v)
                return {"median": m, "min": min(v), "max": max(v), "n": len(v), "spread_pct": (max(v) - min(v)) / m * 100 if m else None}
            s = {"kind": "summary", "tag": args.tag, "fa": args.fa, "prompt": name, "prompt_n": good[0]["prompt_n"],
                 "predicted_n": good[0]["predicted_n"]}
            for k in ("prompt_tps", "prompt_ms", "ttft_wall_s", "decode_tps", "predicted_ms", "total_wall_s", "rss_gib"):
                s[k] = stat(k)
            s["answer_ok"] = all(r["answer_ok"] for r in good)
            s["identical_outputs"] = len({r["content"][:200] for r in good}) == 1
            emit(s)
            log(f"SUMMARY fa{args.fa} {name}: prompt_n {s['prompt_n']} | prefill {s['prompt_tps']['median']:.1f} tok/s "
                f"(spread {s['prompt_tps']['spread_pct']:.1f}%) | TTFT {s['ttft_wall_s']['median'] if s['ttft_wall_s'] else None} s | "
                f"decode {s['decode_tps']['median']:.2f} tok/s (spread {s['decode_tps']['spread_pct']:.1f}%) | ok={s['answer_ok']}")

        for name in names:
            w = one(name, 0, "warmup")
            if w.get("aborted"):
                log(f"warm-up of {name} aborted ({w['aborted']}); skipping its timed runs")
                continue
            rows = []
            for rep in range(1, args.repeats + 1):
                r = one(name, rep, "run")
                rows.append(r)
                if r.get("aborted"):
                    break
            summarize(name, rows)
        for name in singles:
            r = one(name, 1, "run_single")
            summarize(name, [r])

        if not args.no_prefix_test:
            # Prefix-cache test: p_8k twice (exact repeat), then p_8k extended by ~200 tokens before the question.
            body, q = texts["p_8k"].rsplit("\n\nQuestion:", 1)
            src = (PROMPTS / "src/moby-dick.txt").read_text()
            anchor = body[-300:]
            i = src.find(anchor)
            ext = ""
            if i >= 0:
                tail = src[i + len(anchor):]
                ids = tok.encode(tail[:4000], add_special_tokens=False)[:200]
                ext = tok.decode(ids)
            ext_text = body + ext + "\n\nQuestion:" + q
            ext_prompt = render(ext_text)
            n_ext = len(tok.encode(ext_prompt, add_special_tokens=False))
            log(f"prefix test: p_8k {counts['p_8k']} tokens, extended {n_ext} tokens (+{n_ext - counts['p_8k']})")
            a1 = one("p_8k", 1, "prefix_a_first", cache_prompt=True, n_predict=16)
            a2 = one("p_8k", 2, "prefix_a_repeat", cache_prompt=True, n_predict=16)
            b = one("p_8k_ext", 1, "prefix_b_extended", prompt=ext_prompt, cache_prompt=True, n_predict=16)
            emit({"kind": "prefix_summary", "tag": args.tag, "fa": args.fa, "p8k_tokens": counts["p_8k"], "ext_tokens": n_ext,
                  "first": {k: a1.get(k) for k in ("prompt_n", "prompt_ms", "tokens_cached")},
                  "repeat": {k: a2.get(k) for k in ("prompt_n", "prompt_ms", "tokens_cached")},
                  "extended": {k: b.get(k) for k in ("prompt_n", "prompt_ms", "tokens_cached")}})
        log(f"final rss {rss_gib(proc.pid):.1f} GiB")
        emit({"kind": "end", "tag": args.tag, "rss_gib": rss_gib(proc.pid)})
    finally:
        shutdown()
        out.close()


if __name__ == "__main__":
    main()
