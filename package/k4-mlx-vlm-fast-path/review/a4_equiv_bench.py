#!/usr/bin/env python
"""Before/after equivalence and performance driver for Inkling-Small on mlx-vlm.

One process = one model load = one tag. Runs greedy generation (temperature 0,
EOS ignored so every run is exactly --max-tokens long) over the fixed prompts,
records every generated token id, the top-2 logit margin at every step (so a
divergence can be located and explained), prefill tok/s, TTFT, decode tok/s and
peak memory. Warm-up once, then --repeats timed runs; the token ids and margins
are taken from the first timed run (they are identical across runs, checked).

Usage:
  ~/.venvs/inkling-base/bin/python equiv_bench.py --tag before ...   (wheel 0.7.11)
  ~/.venvs/inkling/bin/python      equiv_bench.py --tag after  ...   (branch)
Options: --prompts p_short,p_2k,p_8k,p_32k[,p_64k]  --max-tokens 128  --repeats 3
         --eager-load (lazy=False: exercises load_model_weights itself)
         --kv-bits N --kv-group-size G --quantized-kv-start S
         --out-dir DIR (default: next to this script, results/)
"""
import argparse
import atexit
import glob
import hashlib
import json
import os
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRIB = Path(os.path.expanduser("~/code/inkling-contrib"))
LOCK = CONTRIB / "bench" / "GPU_LOCK"
PROMPTS = CONTRIB / "bench" / "prompts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def acquire_lock(wait_s):
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
            busy.append(f"pid {pid} rss {rss / 2**20:.1f} GiB: {cmd}")
        if not busy:
            LOCK.write_text(f"{os.getpid()} {time.strftime('%FT%T')} k4-equiv-bench\n")
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


def teacher_forced_logprobs(language_model, ids, continuation, chunk, quantize_fn=None):
    """Log-softmax rows (fp32, [len(continuation), V]) predicting each continuation
    token given the prompt ``ids`` and the preceding continuation tokens, computed
    with ``chunk``-token prefill steps exactly like generate_step does (the final
    chunk is the 128 positions whose logits predict the continuation)."""
    import mlx.core as mx
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(language_model)
    seq = list(ids) + list(continuation)
    n = len(continuation)
    head = seq[: len(ids) - 1]                # positions whose logits we discard
    tail = seq[len(ids) - 1 : len(seq) - 1]   # last prompt token + continuation[:-1]
    assert len(tail) == n
    for start in range(0, len(head), chunk):
        language_model(mx.array([head[start : start + chunk]]), cache=cache, logits_to_keep=1)
        if quantize_fn is not None:
            quantize_fn(cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    logits = language_model(mx.array([tail]), cache=cache).logits[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs)
    return logprobs


def teacher_forced(args, model, processor, prompts, names, emit, log):
    import numpy as np
    import mlx.core as mx

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    reference_tokens = {}
    for line in open(args.teacher_forced):
        row = json.loads(line)
        if row.get("kind") == "summary":
            reference_tokens[row["prompt"]] = row["token_ids"][: args.max_tokens]
    quantize_fn = None
    if args.kv_bits is not None:
        from mlx_vlm.generate.common import maybe_quantize_kv_cache
        quantize_fn = lambda c: maybe_quantize_kv_cache(  # noqa: E731
            c, args.quantized_kv_start, args.kv_group_size, args.kv_bits
        )
    chunk = args.prefill_step_size or 2048
    out_dir = Path(args.out_dir)
    for name in names:
        continuation = reference_tokens[name]
        ids = tokenizer.encode(prompts[name], add_special_tokens=False)
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        lp = teacher_forced_logprobs(model.language_model, ids, continuation, chunk, quantize_fn)
        elapsed = time.perf_counter() - t0
        lp_np = np.array(lp, dtype=np.float32)
        np.save(out_dir / f"{args.tag}-tf-{name}.npy", lp_np)
        teacher = np.array(continuation)
        token_lp = lp_np[np.arange(len(teacher)), teacher]
        top1 = lp_np.argmax(axis=-1)
        row = {
            "kind": "tf", "tag": args.tag, "prompt": name, "chunk": chunk, "prompt_tokens": len(ids),
            "n": len(teacher), "elapsed_s": elapsed, "peak_mem_gib_mx": mx.get_peak_memory() / 2**30,
            "teacher_ids": continuation, "token_logprob": [round(float(x), 6) for x in token_lp],
            "top1": top1.tolist(), "top1_matches_teacher": int((top1 == teacher).sum()),
            "kv_bits": args.kv_bits, "md5": hashlib.md5(lp_np.tobytes()).hexdigest(),
        }
        if args.tf_reference:
            ref_path = out_dir / f"{args.tf_reference}-tf-{name}.npy"
            if ref_path.exists():
                ref = np.load(ref_path)
                p_ref = np.exp(ref)
                kl = (p_ref * (ref - lp_np)).sum(axis=-1)          # KL(ref || this) per step, nats
                d = token_lp - ref[np.arange(len(teacher)), teacher]
                row.update({
                    "reference": args.tf_reference, "identical": bool(np.array_equal(ref, lp_np)),
                    "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "kl_argmax": int(kl.argmax()),
                    "dlogprob_mean_abs": float(np.abs(d).mean()), "dlogprob_max_abs": float(np.abs(d).max()),
                    "top1_agree": int((ref.argmax(axis=-1) == top1).sum()),
                })
        emit(row)
        log(f"tf {name}: chunk {chunk}, {len(ids)} prompt tokens + {len(teacher)} teacher tokens in {elapsed:.1f}s, "
            f"teacher top-1 {row['top1_matches_teacher']}/{len(teacher)}, md5 {row['md5'][:12]}"
            + (f", vs {args.tf_reference}: identical={row['identical']} KL mean {row['kl_mean']:.2e} max {row['kl_max']:.2e} "
               f"at {row['kl_argmax']}, |dlogprob| mean {row['dlogprob_mean_abs']:.2e} max {row['dlogprob_max_abs']:.2e}, "
               f"top-1 agree {row['top1_agree']}/{len(teacher)}" if "kl_mean" in row else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="p_short,p_2k,p_8k,p_32k")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-prompt", default="p_short")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--eager-load", action="store_true", help="load with lazy=False")
    ap.add_argument("--chunk-gib", type=float, default=4.0)
    ap.add_argument("--kv-bits", type=float, default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=0)
    ap.add_argument("--prefill-step-size", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--mem-limit-gb", type=float, default=112.0)
    ap.add_argument("--lock-wait", type=float, default=6 * 3600)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--teacher-forced", default=None, metavar="REF_JSONL",
                    help="instead of generating, feed each prompt's 128 tokens from this results file "
                         "and record per-token logprobs, top-1 and the full fp32 log-softmax matrix")
    ap.add_argument("--tf-reference", default=None, metavar="TAG",
                    help="tag whose teacher-forced matrices (<out-dir>/<TAG>-tf-<prompt>.npy) to compare against")
    args = ap.parse_args()

    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = open(out_dir / f"{args.tag}.jsonl", "a")

    def emit(row):
        row["_t"] = time.time()
        out.write(json.dumps(row) + "\n")
        out.flush()

    if not args.no_lock:
        acquire_lock(args.lock_wait)

    import mlx.core as mx
    import mlx_vlm
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    git = None
    src = Path(mlx_vlm.__file__).resolve().parent.parent
    if (src / ".git").exists():
        git = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    meta = {
        "kind": "meta", "tag": args.tag, "model": args.model, "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__, "mlx_vlm_file": mlx_vlm.__file__, "git": git,
        "python": sys.executable, "device": mx.device_info().get("device_name"),
        "argv": sys.argv, "pid": os.getpid(), "load1": os.getloadavg()[0],
        "env_tf32": os.environ.get("MLX_ENABLE_TF32"),
    }
    log(json.dumps(meta))
    emit(meta)

    t0 = time.perf_counter()
    if args.eager_load:
        model, processor = load(args.model, lazy=False)
        t_lazy, t_eval = time.perf_counter() - t0, 0.0
        total = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    else:
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
    log(f"load: {'eager' if args.eager_load else 'lazy'} {t_lazy:.1f}s, chunked eval {t_eval:.1f}s, "
        f"{total / 2**30:.1f} GiB of parameters, active {mx.get_active_memory() / 2**30:.1f} GiB, "
        f"peak {mx.get_peak_memory() / 2**30:.1f} GiB")
    emit({"kind": "load", "tag": args.tag, "eager": args.eager_load, "load_s": t_lazy, "eval_s": t_eval,
          "param_gib": total / 2**30, "active_gib": mx.get_active_memory() / 2**30,
          "peak_gib": mx.get_peak_memory() / 2**30})
    dtypes = {}
    for k, v in tree_flatten(model.parameters()):
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
    log(f"parameter dtypes: {dtypes}")
    try:
        caches = model.language_model.make_cache()
        cache_kinds = sorted({type(getattr(c, "caches", [c])[0]).__name__ for c in caches})
    except Exception as e:  # pragma: no cover
        cache_kinds = [f"unavailable: {e}"]
    log(f"make_cache kv kinds: {cache_kinds}")
    emit({"kind": "cache_kinds", "tag": args.tag, "kinds": cache_kinds, "dtypes": dtypes})

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def build(name):
        text = (PROMPTS / f"{name}.txt").read_text()
        return apply_chat_template(processor, model.config, text, num_images=0,
                                   reasoning_effort=args.reasoning_effort)

    names = [p for p in args.prompts.split(",") if p]
    prompts = {n: build(n) for n in dict.fromkeys(names + [args.warmup_prompt])}
    for n, f in prompts.items():
        n_tok = len(tokenizer.encode(f, add_special_tokens=False))
        log(f"prompt {n}: {len(f)} chars, {n_tok} tokens")
        emit({"kind": "prompt", "tag": args.tag, "prompt": n, "chars": len(f), "tokens": n_tok})

    gen_kwargs = dict(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=0, verbose=False)
    if args.prefill_step_size is not None:
        gen_kwargs["prefill_step_size"] = args.prefill_step_size
    if args.kv_bits is not None:
        gen_kwargs.update(kv_bits=args.kv_bits, kv_group_size=args.kv_group_size,
                          quantized_kv_start=args.quantized_kv_start)
    log(f"gen kwargs: {gen_kwargs}")
    tokenizer.stopping_criteria = lambda tok: False  # fixed-length runs

    def run(name, rep, timed=True):
        prompt = prompts[name]
        mx.clear_cache()
        mx.reset_peak_memory()
        t_start = time.perf_counter()
        t_first = None
        t_last = t_start
        last = None
        token_ids, margins, aborted = [], [], None
        for r in stream_generate(model, processor, prompt, **gen_kwargs):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            last = r
            if r.token is not None:
                token_ids.append(int(r.token))
                lp = r.logprobs
                if lp is not None:
                    top2 = mx.topk(lp.astype(mx.float32), 2)
                    margins.append(round(float((top2[1] - top2[0]).item()), 4))
            if now - t_start > args.time_limit:
                aborted = f"time limit {args.time_limit}s"
                break
            if mx.get_active_memory() > args.mem_limit_gb * 1e9:
                aborted = f"memory limit {args.mem_limit_gb} GB"
                break
        t_end = time.perf_counter()
        if last is None:
            row = {"kind": "run", "tag": args.tag, "prompt": name, "rep": rep, "error": "no output"}
            emit(row)
            return row
        gen_tokens = last.generation_tokens
        ttft = t_first - t_start
        decode_wall = t_last - t_first
        row = {
            "kind": "run" if timed else "warmup", "tag": args.tag, "prompt": name, "rep": rep,
            "prompt_tokens": last.prompt_tokens,
            "prefill_s_api": last.prompt_tokens / last.prompt_tps if last.prompt_tps else None,
            "prompt_tps_api": last.prompt_tps,
            "ttft_wall_s": ttft,
            "gen_tokens": gen_tokens,
            "generation_tps_api": last.generation_tps,
            "decode_tps_wall": (gen_tokens - 1) / decode_wall if gen_tokens > 1 and decode_wall > 0 else None,
            "total_wall_s": t_end - t_start,
            "peak_mem_gb_api": last.peak_memory,
            "peak_mem_gib_mx": mx.get_peak_memory() / 2**30,
            "active_mem_gib_mx": mx.get_active_memory() / 2**30,
            "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
            "token_ids": token_ids,
            "margins": margins,
            "text64": tokenizer.decode(token_ids[:64]),
            "aborted": aborted,
            "load1": os.getloadavg()[0],
        }
        emit(row)
        log(f"{row['kind']} {name} rep{rep}: prompt {row['prompt_tokens']} tok, prefill {row['prefill_s_api']:.2f}s "
            f"({row['prompt_tps_api']:.1f} tok/s), TTFT {ttft:.2f}s, decode {gen_tokens} tok "
            f"{(row['decode_tps_wall'] or 0):.2f} tok/s wall ({last.generation_tps:.2f} api), "
            f"peak {last.peak_memory:.1f} GB api / {row['peak_mem_gib_mx']:.1f} GiB, load1 {row['load1']:.1f}, "
            f"aborted={aborted}")
        log(f"  text64: {row['text64']!r}")
        return row

    if args.teacher_forced:
        teacher_forced(args, model, processor, prompts, names, emit, log)
        emit({"kind": "end", "tag": args.tag})
        out.close()
        return

    log(f"warm-up with {args.warmup_prompt}")
    run(args.warmup_prompt, 0, timed=False)
    for name in names:
        rows = []
        for rep in range(1, args.repeats + 1):
            rows.append(run(name, rep))
            if rows[-1].get("aborted") or rows[-1].get("error"):
                break
        good = [r for r in rows if not r.get("aborted") and not r.get("error")]
        if not good:
            continue

        def stat(key):
            vals = [r[key] for r in good if r.get(key) is not None]
            if not vals:
                return None
            med = statistics.median(vals)
            return {"median": med, "min": min(vals), "max": max(vals), "n": len(vals),
                    "spread_pct": (max(vals) - min(vals)) / med * 100 if med else None}

        summary = {"kind": "summary", "tag": args.tag, "prompt": name,
                   "prompt_tokens": good[0]["prompt_tokens"], "gen_tokens": good[0]["gen_tokens"],
                   "identical_outputs": len({tuple(r["token_ids"]) for r in good}) == 1,
                   "token_ids": good[0]["token_ids"], "margins": good[0]["margins"]}
        for key in ("prompt_tps_api", "prefill_s_api", "ttft_wall_s", "generation_tps_api", "decode_tps_wall",
                    "total_wall_s", "peak_mem_gb_api", "peak_mem_gib_mx", "maxrss_gib"):
            summary[key] = stat(key)
        emit(summary)
        s = summary
        log(f"SUMMARY {name}: prompt {s['prompt_tokens']} tok | prefill {s['prompt_tps_api']['median']:.1f} tok/s "
            f"(spread {s['prompt_tps_api']['spread_pct']:.1f}%) | TTFT {s['ttft_wall_s']['median']:.2f}s | "
            f"decode {s['decode_tps_wall']['median']:.2f} tok/s (spread {s['decode_tps_wall']['spread_pct']:.1f}%) "
            f"| peak {s['peak_mem_gib_mx']['max']:.1f} GiB | identical={s['identical_outputs']}")
    log(f"done; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB")
    emit({"kind": "end", "tag": args.tag})
    out.close()


if __name__ == "__main__":
    main()
