#!/usr/bin/env python
"""Baseline MLX (mlx-vlm) benchmark for Inkling-Small on Apple Silicon.

Follows bench/PROTOCOL.md: fixed prompts, greedy decoding, 128 new tokens, one
warm-up (discarded) then N timed runs per prompt. Loads once with lazy=True and
evaluates the parameters in ~4 GiB chunks (a one-shot mx.eval of ~82 GB trips the
Metal GPU watchdog on this machine). Writes one JSON line per run plus a summary
line per prompt to --out, and a readable log to stdout.

Usage (from bench/):
  python baseline/bench_mlx.py --model DIR --tag baseline \
      --prompts p_short,p_2k,p_8k,p_32k,p_64k --repeats 3 --max-tokens 128
  python baseline/bench_mlx.py ... --tag kv8 --kv-bits 8 --quantized-kv-start 0
  python baseline/bench_mlx.py ... --tag mtp --draft-model PATH --draft-kind mtp
  python baseline/bench_mlx.py ... --tag raw --no-template      # raw completion path
"""
import argparse
import atexit
import glob
import json
import os
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
LOCK = BENCH_DIR / "GPU_LOCK"
RESULTS = BENCH_DIR / "results"
PROMPTS = BENCH_DIR / "prompts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- lock
def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def big_processes(min_rss_kb=20_000_000):
    out = subprocess.run(["ps", "-axo", "pid,rss,command"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss = int(parts[0]), int(parts[1])
        if rss > min_rss_kb and pid != os.getpid():
            rows.append((pid, rss, parts[2][:120]))
    return rows


COMPETE_RE = re.compile(
    r"test-backend-ops|llama-(bench|server|cli|perplexity)|fa_bench|sched_bench|verify_dense|"
    r"bench_(sdpa|moe|layer|misc)|kernels_bench|layer_bench|bw_probe|kv_probe|sdpa_probe|mtp_probe|diag_loop|"
    r"mlx_vlm|mlx_lm|load_mlx_chunked")


def competing():
    """Other processes likely to touch the GPU (regex) or hog the CPU (>90%)."""
    out = subprocess.run(["ps", "-axo", "pid,pcpu,etime,command"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pcpu, et, cmd = int(parts[0]), float(parts[1]), parts[2], parts[3]
        if pid == os.getpid() or "bench_mlx.py" in cmd:
            continue
        gpu = bool(COMPETE_RE.search(cmd))
        if gpu or pcpu > 90:
            rows.append({"pid": pid, "pcpu": pcpu, "etime": et, "cmd": cmd[:100], "gpu_suspect": gpu})
    return rows


def wait_quiet(limit_s):
    override = BENCH_DIR / "baseline" / "QUIET_WAIT"
    if override.exists():
        try:
            limit_s = float(override.read_text().split()[0])
        except Exception:
            pass
    t0 = time.time()
    while limit_s > 0:
        c = [r for r in competing() if r["gpu_suspect"]]
        if not c:
            return
        if time.time() - t0 > limit_s:
            log(f"quiet-wait of {limit_s}s exhausted; proceeding with competing processes: {c}")
            return
        log(f"waiting for GPU-suspect processes: {[(r['pid'], r['cmd'][:70]) for r in c]}")
        time.sleep(15)


def acquire_lock(wait_s=3600):
    t0 = time.time()
    while True:
        busy = []
        if LOCK.exists():
            try:
                pid = int(LOCK.read_text().split()[0])
            except Exception:
                pid = -1
            if pid > 0 and pid_alive(pid):
                busy.append(f"GPU_LOCK held by pid {pid}")
            else:
                log(f"stale GPU_LOCK (pid {pid}); removing")
                LOCK.unlink(missing_ok=True)
        for pid, rss, cmd in big_processes():
            busy.append(f"pid {pid} rss {rss/2**20:.1f} GiB: {cmd}")
        if not busy:
            LOCK.write_text(f"{os.getpid()} {time.strftime('%FT%T')} bench_mlx\n")
            atexit.register(release_lock)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: sys.exit(1))
            log(f"GPU_LOCK acquired by pid {os.getpid()}")
            return
        if time.time() - t0 > wait_s:
            raise SystemExit(f"GPU busy for {wait_s}s: {busy}")
        log(f"GPU busy, waiting: {busy}")
        time.sleep(30)


def release_lock():
    try:
        if LOCK.exists() and LOCK.read_text().split()[0] == str(os.getpid()):
            LOCK.unlink()
            log("GPU_LOCK released")
    except Exception as e:  # pragma: no cover
        log(f"lock release failed: {e}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="p_short,p_2k,p_8k,p_32k")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-prompt", default="p_short")
    ap.add_argument("--warmup-tokens", type=int, default=128)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--no-template", action="store_true", help="raw completion: feed the prompt text with no chat template")
    ap.add_argument("--kv-bits", type=float, default=None)
    ap.add_argument("--kv-key-bits", type=float, default=None)
    ap.add_argument("--kv-value-bits", type=float, default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--kv-quant-scheme", default="uniform")
    ap.add_argument("--quantized-kv-start", type=int, default=0)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--draft-kind", default=None)
    ap.add_argument("--draft-block-size", type=int, default=None)
    ap.add_argument("--prefill-step-size", type=int, default=None, help="mlx-vlm default is 2048")
    ap.add_argument("--chunk-gib", type=float, default=4.0)
    ap.add_argument("--time-limit", type=float, default=1200.0, help="abort a run after this many seconds")
    ap.add_argument("--mem-limit-gb", type=float, default=110.0, help="abort a run if active memory exceeds this (GB)")
    ap.add_argument("--keep-eos", action="store_true", help="stop at EOS (default: ignore EOS so every run is exactly max-tokens)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="also time the smoke-test configuration (20-token prompt, default effort, 48 tokens, generate()) in this process")
    ap.add_argument("--quiet-wait", type=float, default=0.0, help="before each run wait up to N s for other GPU-suspect processes to finish")
    args = ap.parse_args()

    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    out_path = Path(args.out) if args.out else RESULTS / f"mlx-{args.tag}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    out = open(out_path, "a")

    def emit(row):
        row["_t"] = time.time()
        out.write(json.dumps(row) + "\n")
        out.flush()

    if not args.no_lock:
        acquire_lock()

    import mlx.core as mx
    import mlx_vlm
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.models import cache as mlx_cache
    from mlx_vlm.prompt_utils import apply_chat_template

    info = mx.device_info()
    meta = {
        "kind": "meta", "tag": args.tag, "model": args.model, "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__, "device": info.get("device_name"),
        "argv": sys.argv, "pid": os.getpid(), "load1": os.getloadavg()[0],
    }
    log(json.dumps(meta))
    emit(meta)

    # -------------------------------------------------------------- load (chunked)
    t0 = time.perf_counter()
    model, processor = load(args.model, lazy=True)
    t_lazy = time.perf_counter() - t0
    params = tree_flatten(model.parameters())
    chunk, nb, total = [], 0, 0
    t1 = time.perf_counter()
    for _, v in params:
        chunk.append(v)
        nb += v.nbytes
        if nb > args.chunk_gib * 2**30:
            mx.eval(chunk)
            total += nb
            chunk, nb = [], 0
    if chunk:
        mx.eval(chunk)
        total += nb
    t_eval = time.perf_counter() - t1
    log(f"load: lazy {t_lazy:.1f}s, chunked eval {t_eval:.1f}s, {total/2**30:.1f} GiB of parameters, "
        f"active {mx.get_active_memory()/2**30:.1f} GiB, peak {mx.get_peak_memory()/2**30:.1f} GiB")
    emit({"kind": "load", "tag": args.tag, "lazy_s": t_lazy, "eval_s": t_eval, "param_gib": total / 2**30,
          "active_gib": mx.get_active_memory() / 2**30, "peak_gib": mx.get_peak_memory() / 2**30})

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_id = model.config.eos_token_id
    eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]

    draft_model = None
    if args.draft_model:
        from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
        draft_model, kind = load_drafter(args.draft_model, kind=args.draft_kind)
        validate_drafter_compatibility(model, draft_model, kind)
        args.draft_kind = kind
        log(f"drafter loaded: {args.draft_model} kind={kind}")

    # -------------------------------------------------------------- prompts
    def build(name):
        text = (PROMPTS / f"{name}.txt").read_text()
        if args.no_template:
            return text
        formatted = apply_chat_template(processor, model.config, text, num_images=0,
                                        reasoning_effort=args.reasoning_effort)
        return formatted

    names = [p for p in args.prompts.split(",") if p]
    prompts = {n: build(n) for n in dict.fromkeys(names + [args.warmup_prompt])}
    for n, f in prompts.items():
        n_tok = len(tokenizer.encode(f, add_special_tokens=False)) if hasattr(tokenizer, "encode") else -1
        log(f"prompt {n}: {len(f)} chars, {n_tok} tokens; head={f[:140]!r} tail={f[-120:]!r}")
        emit({"kind": "prompt", "tag": args.tag, "prompt": n, "chars": len(f), "tokens": n_tok,
              "head": f[:200], "tail": f[-160:]})
    if not args.no_template:
        f0 = prompts[names[0]]
        if "Thinking effort level: 0<" not in f0:
            log("WARNING: reasoning effort 'none' NOT honoured by the template; header is: " + f0[:120])
            emit({"kind": "warning", "tag": args.tag, "msg": "reasoning effort not honoured", "head": f0[:200]})

    # Memory watchdog: the per-token guard below only runs between decode steps, so a prefill that
    # outgrows the machine would never be caught; this thread samples active memory during prefill too.
    import threading

    def _watchdog():
        while True:
            time.sleep(0.5)
            act = mx.get_active_memory()
            if act > args.mem_limit_gb * 1e9:
                msg = f"MEMORY GUARD: active {act/1e9:.1f} GB > {args.mem_limit_gb} GB (peak {mx.get_peak_memory()/1e9:.1f} GB); exiting"
                log(msg)
                emit({"kind": "aborted", "tag": args.tag, "reason": msg, "active_gb": act / 1e9,
                      "peak_gb": mx.get_peak_memory() / 1e9})
                out.flush()
                release_lock()
                os._exit(3)

    threading.Thread(target=_watchdog, daemon=True).start()

    gen_kwargs = dict(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=0, verbose=False)
    if args.prefill_step_size is not None:
        gen_kwargs["prefill_step_size"] = args.prefill_step_size
    if args.kv_bits is not None:
        gen_kwargs.update(kv_bits=args.kv_bits, kv_group_size=args.kv_group_size,
                          kv_quant_scheme=args.kv_quant_scheme, quantized_kv_start=args.quantized_kv_start)
        if args.kv_key_bits is not None:
            gen_kwargs["kv_key_bits"] = args.kv_key_bits
        if args.kv_value_bits is not None:
            gen_kwargs["kv_value_bits"] = args.kv_value_bits
    if draft_model is not None:
        gen_kwargs.update(draft_model=draft_model, draft_kind=args.draft_kind)
        if args.draft_block_size:
            gen_kwargs["draft_block_size"] = args.draft_block_size
    log(f"gen kwargs: { {k: v for k, v in gen_kwargs.items() if k != 'draft_model'} }")

    def cache_summary(pc):
        """Types/offsets of the per-layer caches (proves whether kv quantization took effect)."""
        from collections import Counter
        kinds = Counter()
        offsets = set()
        nbytes = 0
        for c in pc:
            subs = list(getattr(c, "caches", [c]))
            kinds[type(c).__name__ + "(" + ",".join(type(x).__name__ for x in subs) + ")"] += 1
            for x in subs:
                off = getattr(x, "offset", None)
                if isinstance(off, int):
                    offsets.add(off)
                st = getattr(x, "state", None)
                try:
                    for a in (st if isinstance(st, (list, tuple)) else [st]):
                        if a is not None and hasattr(a, "nbytes"):
                            nbytes += a.nbytes
                except Exception:
                    pass
        return {"kinds": dict(kinds), "offsets": sorted(offsets)[:4], "state_gib": nbytes / 2**30}

    never_stop = lambda tok: False  # noqa: E731

    def run(name, max_tokens, rep, timed=True):
        prompt = prompts[name]
        wait_quiet(args.quiet_wait)
        comp = competing()
        if args.keep_eos:
            tokenizer.stopping_criteria.reset(eos_id)
        else:
            tokenizer.stopping_criteria = never_stop
        mx.clear_cache()
        mx.reset_peak_memory()
        kw = dict(gen_kwargs, max_tokens=max_tokens)
        pc = mlx_cache.make_prompt_cache(model.language_model)
        kw["prompt_cache"] = pc
        t_start = time.perf_counter()
        t_first = None
        t_last = t_start
        last = None
        tokens = []
        aborted = None
        n_yield = 0
        for r in stream_generate(model, processor, prompt, **kw):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            last = r
            n_yield += 1
            if r.token is not None:
                tokens.append(int(r.token))
            if now - t_start > args.time_limit:
                aborted = f"time limit {args.time_limit}s"
                break
            if mx.get_active_memory() > args.mem_limit_gb * 1e9:
                aborted = f"memory limit {args.mem_limit_gb} GB (active {mx.get_active_memory()/1e9:.1f} GB)"
                break
        t_end = time.perf_counter()
        if last is None:
            row = {"kind": "run", "tag": args.tag, "prompt": name, "rep": rep, "error": "no output"}
            emit(row)
            return row
        token_ids = list(last.token_ids) if getattr(last, "token_ids", None) else tokens
        gen_tokens = last.generation_tokens
        ttft_wall = t_first - t_start
        decode_wall = t_last - t_first
        decode_tps_wall = (gen_tokens - 1) / decode_wall if gen_tokens > 1 and decode_wall > 0 else None
        text64 = tokenizer.decode(token_ids[:64]) if token_ids else ""
        text_all = tokenizer.decode(token_ids) if token_ids else ""
        first_eos = next((i for i, t in enumerate(token_ids) if t in eos_ids), None)
        row = {
            "kind": "run" if timed else "warmup", "tag": args.tag, "prompt": name, "rep": rep,
            "prompt_tokens": last.prompt_tokens,
            "prefill_s_api": last.prompt_tokens / last.prompt_tps if last.prompt_tps else None,
            "prompt_tps_api": last.prompt_tps,
            "ttft_wall_s": ttft_wall,
            "gen_tokens": gen_tokens,
            "generation_tps_api": last.generation_tps,
            "decode_tps_wall": decode_tps_wall,
            "decode_wall_s": decode_wall,
            "total_wall_s": t_end - t_start,
            "peak_mem_gb_api": last.peak_memory,
            "peak_mem_gib_mx": mx.get_peak_memory() / 2**30,
            "active_mem_gib_mx": mx.get_active_memory() / 2**30,
            "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
            "finish_reason": getattr(last, "finish_reason", None),
            "first_eos_pos": first_eos,
            "answer_ok": ("Ishmael" in text_all) and ("Pequod" in text_all),
            "first64": text64,
            "first64_ids": token_ids[:64],
            "n_yield": n_yield,
            "aborted": aborted,
            "load1": os.getloadavg()[0],
            "competing": comp,
            "competing_end": competing(),
            "cache_types": cache_summary(pc),
        }
        if draft_model is not None:
            try:
                from mlx_vlm.speculative.utils import format_speculative_stats
                row["speculative_stats"] = format_speculative_stats(draft_model)
            except Exception as e:  # pragma: no cover
                row["speculative_stats"] = f"unavailable: {e}"
            for attr in ("accepted_tokens", "draft_tokens", "rounds", "acceptance_rate"):
                if hasattr(draft_model, attr):
                    row[f"draft_{attr}"] = getattr(draft_model, attr)
        emit(row)
        log(f"{row['kind']} {name} rep{rep}: prompt {row['prompt_tokens']} tok, prefill {row['prefill_s_api']:.2f}s "
            f"({row['prompt_tps_api']:.1f} tok/s api), TTFT wall {ttft_wall:.2f}s, decode {gen_tokens} tok "
            f"{(decode_tps_wall or 0):.2f} tok/s wall ({last.generation_tps:.2f} api), total {t_end-t_start:.1f}s, "
            f"peak {last.peak_memory:.1f} GB api / {mx.get_peak_memory()/2**30:.1f} GiB, rss {row['maxrss_gib']:.1f} GiB, "
            f"eos@{first_eos} ok={row['answer_ok']} aborted={aborted} load1={row['load1']:.1f} competing={[(c['pid'], c['pcpu'], c['cmd'][:40]) for c in comp]}")
        log(f"  first64: {text64!r}")
        log(f"  cache: {row['cache_types']}")
        return row

    # -------------------------------------------------------------- warm-up
    log(f"warm-up with {args.warmup_prompt} ({args.warmup_tokens} tokens)")
    run(args.warmup_prompt, args.warmup_tokens, 0, timed=False)

    if args.smoke:
        from mlx_vlm import generate as vlm_generate
        smoke_prompt = apply_chat_template(processor, model.config, "The capital of France is", num_images=0)
        for rep in range(1, 4):
            wait_quiet(args.quiet_wait)
            comp = competing()
            tokenizer.stopping_criteria.reset(eos_id)
            mx.clear_cache()
            t0 = time.perf_counter()
            res = vlm_generate(model, processor, smoke_prompt, max_tokens=48, temperature=0.0, verbose=False)
            row = {"kind": "smoke", "tag": args.tag, "rep": rep, "prompt_tokens": res.prompt_tokens, "prompt_tps_api": res.prompt_tps,
                   "gen_tokens": res.generation_tokens, "generation_tps_api": res.generation_tps, "peak_mem_gb_api": res.peak_memory,
                   "total_wall_s": time.perf_counter() - t0, "text": res.text[:200], "load1": os.getloadavg()[0], "competing": comp}
            emit(row)
            log(f"smoke rep{rep}: prompt {res.prompt_tokens} tok {res.prompt_tps:.1f} tok/s, gen {res.generation_tokens} tok {res.generation_tps:.2f} tok/s api, "
                f"peak {res.peak_memory:.1f} GB, load1 {row['load1']:.1f}, competing={[(c['pid'], c['pcpu'], c['cmd'][:30]) for c in comp]}")

    # -------------------------------------------------------------- timed runs
    stop_all = False
    for name in names:
        if stop_all:
            log(f"skipping {name} after an aborted run")
            emit({"kind": "skipped", "tag": args.tag, "prompt": name})
            continue
        rows = []
        for rep in range(1, args.repeats + 1):
            row = run(name, args.max_tokens, rep)
            rows.append(row)
            if row.get("aborted") or row.get("error"):
                stop_all = True
                break
        good = [r for r in rows if not r.get("aborted") and not r.get("error")]
        if good:
            def stat(key):
                vals = [r[key] for r in good if r.get(key) is not None]
                if not vals:
                    return None
                med = statistics.median(vals)
                return {"median": med, "min": min(vals), "max": max(vals), "n": len(vals),
                        "spread_pct": (max(vals) - min(vals)) / med * 100 if med else None}
            summary = {"kind": "summary", "tag": args.tag, "prompt": name, "prompt_tokens": good[0]["prompt_tokens"],
                       "gen_tokens": good[0]["gen_tokens"]}
            for key in ("prompt_tps_api", "prefill_s_api", "ttft_wall_s", "generation_tps_api", "decode_tps_wall",
                        "total_wall_s", "peak_mem_gb_api", "peak_mem_gib_mx", "maxrss_gib"):
                summary[key] = stat(key)
            summary["answer_ok"] = all(r["answer_ok"] for r in good)
            summary["identical_outputs"] = len({tuple(r["first64_ids"]) for r in good}) == 1
            emit(summary)
            s = summary
            log(f"SUMMARY {name}: prompt {s['prompt_tokens']} tok | prefill {s['prompt_tps_api']['median']:.1f} tok/s "
                f"(spread {s['prompt_tps_api']['spread_pct']:.1f}%) | TTFT {s['ttft_wall_s']['median']:.2f}s | "
                f"decode {s['decode_tps_wall']['median']:.2f} tok/s wall (spread {s['decode_tps_wall']['spread_pct']:.1f}%) "
                f"| peak {s['peak_mem_gb_api']['max']:.1f} GB | answer_ok={s['answer_ok']} identical={s['identical_outputs']}")

    log(f"done; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GiB, "
        f"mx peak {mx.get_peak_memory()/2**30:.1f} GiB")
    emit({"kind": "end", "tag": args.tag, "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30})
    out.close()


if __name__ == "__main__":
    main()
