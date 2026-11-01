#!/usr/bin/env python3
"""A3 review: the accumulation-order noise floor on Inkling-Small, no cache involved.

Starts the fixed llama-server at a different ubatch (default -ub 256) and evaluates, without the cache, the same
prompts the main run (a3_gpu_run.py, -ub 512) evaluated without the cache: A (p_8k), B1 = A + 200, X = A[:8192] + 200,
plus X + the first 4 tokens the -ub 512 run generated for X (a prompt-length change moves the server's batch breaks,
so the same prefix is split differently even at one ubatch size). 64 greedy tokens with top-5 logprobs per step.
a3_gpu_compare.py pairs these rows with the main run's rows.
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("a3_gpu_run", HERE / "a3_gpu_run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

BENCH = Path.home() / "code/inkling-contrib/bench"
PROMPTS = BENCH / "prompts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--tag", default="a3-noise-ub256")
    ap.add_argument("--ub", type=int, default=256)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--port", type=int, default=18651)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--main-jsonl", default=str(HERE / "a3-fixed-ub512.jsonl"))
    ap.add_argument("--model", default=None)
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
        run.log(json.dumps({k: v for k, v in row.items() if k not in ("tokens", "content", "probs")}))

    # the first 4 generated tokens of X in the main run
    x_gen = None
    if os.path.exists(args.main_jsonl):
        for line in open(args.main_jsonl):
            r = json.loads(line)
            if r.get("kind") == "completion" and r.get("name") == "X_ref":
                x_gen = r["tokens"][:4]

    bin_dir = Path(args.bin_dir)
    cmd = [str(bin_dir / "llama-server"), "-m", model, "-np", "1", "--no-kv-unified", "-ngl", "99", "-fa", "0",
           "-c", str(args.ctx), "--jinja", "--temp", "0", "-ub", str(args.ub),
           "--host", "127.0.0.1", "--port", str(args.port), "--no-webui", "-to", "3600", "-lv", "4"]
    emit({"kind": "meta", "model": model, "cmd": cmd, "ub": args.ub, "n_predict": args.n_predict, "load1": os.getloadavg()[0]})
    server_log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd)}\n")
    server_log.flush()
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
    run.log(f"llama-server pid {proc.pid}")
    base = f"http://127.0.0.1:{args.port}"
    srv = run.Server(base)

    def shutdown():
        if proc.poll() is None:
            run.log("stopping llama-server")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        run.log(f"llama-server exit code {proc.returncode}")

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
            if time.perf_counter() - t0 > 900:
                raise SystemExit("llama-server did not become healthy in time")
            time.sleep(2)
        emit({"kind": "load", "load_s": round(time.perf_counter() - t0, 1), "rss_gib": run.rss_gib(proc.pid)})

        def tokenize(text, add_special):
            body, _ = srv.post("/tokenize", {"content": text, "add_special": add_special})
            return body["tokens"]

        texts = {n: (PROMPTS / f"{n}.txt").read_text() for n in ["p_short", "p_8k", "p_2k"]}
        src = (PROMPTS / "src/moby-dick.txt").read_text(encoding="utf-8", errors="replace")
        excerpt = texts["p_8k"].split("\n\nQuestion:")[0]
        i = src.find(excerpt[-200:])
        assert i > 0
        continuation = src[i + 200:i + 200 + 4000]
        A = tokenize(texts["p_8k"], True)
        E = tokenize("\n" + continuation, False)[:200]
        emit({"kind": "prompts", "n_A": len(A), "n_E": len(E), "x_gen4": x_gen})

        def completion(name, prompt, note=""):
            ttft, wall, last, tokens, content, probs = srv.stream("/completion", {
                "prompt": prompt, "cache_prompt": False, "n_predict": args.n_predict, "temperature": 0,
                "n_probs": 5, "stream": True, "return_tokens": True})
            t = (last or {}).get("timings", {})
            row = {"kind": "completion", "name": name, "cache_prompt": False, "n_prompt": len(prompt),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 3), "note": note,
                   "rss_gib": run.rss_gib(proc.pid), "tokens": tokens, "content": content[:300], "probs": probs}
            emit(row)
            return row

        completion("warmup", tokenize(texts["p_short"], True), note="warm-up")
        completion("A_ref", A, note=f"p_8k, no cache, -ub {args.ub}")
        completion("B1_ref", A + E, note=f"p_8k + 200, no cache, -ub {args.ub}")
        X = A[:8192] + E
        r_x = completion("X_ref", X, note=f"A[:8192] + 200, no cache, -ub {args.ub}")
        g4 = x_gen or r_x["tokens"][:4]
        completion("X4_ref", X + g4, note="X + the first 4 generated tokens of X: other batch breaks, same ubatch")
        emit({"kind": "phase1_done", "rss_gib": run.rss_gib(proc.pid), "load1": os.getloadavg()[0]})
    finally:
        shutdown()

    # ---- phase 2: the live-state path on p_2k, 8 different 100-token extensions, cached vs uncached, -ub 512
    cmd2 = list(cmd)
    cmd2[cmd2.index("-ub") + 1] = "512"
    emit({"kind": "meta2", "cmd": cmd2, "load1": os.getloadavg()[0]})
    server_log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd2)}\n")
    server_log.flush()
    proc = subprocess.Popen(cmd2, stdout=server_log, stderr=subprocess.STDOUT)
    run.log(f"llama-server pid {proc.pid} (phase 2)")
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
            if time.perf_counter() - t0 > 900:
                raise SystemExit("llama-server did not become healthy in time")
            time.sleep(2)
        emit({"kind": "load2", "load_s": round(time.perf_counter() - t0, 1), "rss_gib": run.rss_gib(proc.pid)})

        def tokenize(text, add_special):
            body, _ = srv.post("/tokenize", {"content": text, "add_special": add_special})
            return body["tokens"]

        def completion2(name, prompt, cache_prompt, n_predict=args.n_predict, note=""):
            ttft, wall, last, tokens, content, probs = srv.stream("/completion", {
                "prompt": prompt, "cache_prompt": cache_prompt, "n_predict": n_predict, "temperature": 0,
                "n_probs": 5, "stream": True, "return_tokens": True})
            t = (last or {}).get("timings", {})
            row = {"kind": "completion", "name": name, "cache_prompt": cache_prompt, "n_prompt": len(prompt),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 3), "note": note,
                   "rss_gib": run.rss_gib(proc.pid), "tokens": tokens, "content": content[:300], "probs": probs}
            emit(row)
            return row

        def has_loop(tokens, n=6):
            # a 6-gram that occurs twice within the 64 tokens
            seen = set()
            for k in range(len(tokens) - n + 1):
                g = tuple(tokens[k:k + n])
                if g in seen:
                    return True
                seen.add(g)
            return False

        P2 = tokenize((PROMPTS / "p_2k.txt").read_text(), True)
        excerpt2 = (PROMPTS / "p_2k.txt").read_text().split("\n\nQuestion:")[0]
        j = src.find(excerpt2[-200:])
        assert j > 0
        completion2("warmup2", tokenize(texts["p_short"], True), False, n_predict=8, note="warm-up")
        n_loop_c = n_loop_u = n_div = 0
        for k in range(8):
            ext = tokenize("\n" + src[j + 200 + k * 6000:j + 200 + k * 6000 + 3000], False)[:100]
            Pk = P2 + ext
            r_u = completion2(f"L{k}_ref", Pk, False, note="p_2k + 100 (variant %d), no cache" % k)
            completion2(f"L{k}_prime", P2, True, n_predict=1, note="prime p_2k (slot holds exactly p_2k)")
            r_c = completion2(f"L{k}_cached", Pk, True, note="strict extension from the live state")
            fd = run.first_diff(r_c["tokens"], r_u["tokens"])
            m_c = m_r = None
            if fd is not None and fd < min(len(r_c["probs"]), len(r_u["probs"])):
                m_c, m_r = run.margins(r_c["probs"][fd], r_u["probs"][fd], r_c["tokens"][fd], r_u["tokens"][fd])
            lc, lu = has_loop(r_c["tokens"]), has_loop(r_u["tokens"])
            n_loop_c += lc
            n_loop_u += lu
            n_div += fd is not None
            emit({"kind": "loop_compare", "k": k, "first_diff": fd, "cache_n": r_c["cache_n"], "prompt_n": r_c["prompt_n"],
                  "margin_in_cached": m_c, "margin_in_ref": m_r, "loop_cached": lc, "loop_ref": lu,
                  "step0_dev": run.max_dev(r_c["probs"][:1], r_u["probs"][:1], 1)[0] if r_c["probs"] and r_u["probs"] else None})
        emit({"kind": "loop_summary", "n": 8, "n_diverged": n_div, "loops_cached": n_loop_c, "loops_uncached": n_loop_u})
        emit({"kind": "done", "rss_gib": run.rss_gib(proc.pid), "load1": os.getloadavg()[0]})
    finally:
        shutdown()
        server_log.close()


if __name__ == "__main__":
    main()
