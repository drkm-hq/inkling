#!/usr/bin/env python
"""mlx-lm benchmark for Inkling-Small, mirroring bench/baseline/bench_mlx.py (mlx-vlm) so the
rows are comparable: chat template with reasoning effort "none", greedy, 128 new tokens, EOS
ignored, one warm-up then N timed runs per prompt, chunked parameter eval at load.
Writes one JSON line per run plus a summary line per prompt to --out.

  python bench_mlx_lm.py --tag mlxlm-2bit --prompts p_short,p_2k,p_8k,p_32k --repeats 3
  python bench_mlx_lm.py --tag mlxlm-attn8 --model ~/models/inkling-small/mlx-2bit-attn8
"""
import argparse
import glob
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpulock  # noqa: E402
from gpulock import log  # noqa: E402

PROMPTS = HERE.parent / "prompts"
RESULTS = HERE.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="p_short,p_2k,p_8k,p_32k")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-prompt", default="p_short")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--no-template", action="store_true")
    ap.add_argument("--prefill-step-size", type=int, default=2048)
    ap.add_argument("--chunk-gib", type=float, default=4.0)
    ap.add_argument("--model-type", default=None,
                    help="override config.json's model_type (selects mlx_lm/models/<model_type>.py)")
    ap.add_argument("--one-shot-load", action="store_true",
                    help="use mlx_lm.load(lazy=False) (one mx.eval of all parameters)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    out_path = Path(args.out) if args.out else RESULTS / f"K2-{args.tag}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    out = open(out_path, "a")

    def emit(row):
        row["_t"] = time.time()
        out.write(json.dumps(row) + "\n")
        out.flush()

    if not args.no_lock:
        gpulock.acquire(f"bench-{args.tag}")

    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    meta = {"kind": "meta", "tag": args.tag, "model": args.model, "model_type": args.model_type, "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__, "mlx_lm_file": mlx_lm.__file__,
            "device": mx.device_info().get("device_name"), "argv": sys.argv, "pid": os.getpid(),
            "load1": os.getloadavg()[0]}
    log(json.dumps(meta))
    emit(meta)

    t0 = time.perf_counter()
    mc = {"model_type": args.model_type} if args.model_type else None
    if args.one_shot_load:
        model, tokenizer = load(args.model, model_config=mc)
        t_lazy, t_eval, total = time.perf_counter() - t0, 0.0, sum(
            v.nbytes for _, v in tree_flatten(model.parameters()))
    else:
        model, tokenizer = load(args.model, lazy=True, model_config=mc)
        t_lazy = time.perf_counter() - t0
        total, t_eval = gpulock.chunked_eval([v for _, v in tree_flatten(model.parameters())],
                                             args.chunk_gib)
    log(f"load: lazy {t_lazy:.1f}s, eval {t_eval:.1f}s, {total / 2**30:.1f} GiB of parameters, "
        f"active {mx.get_active_memory() / 2**30:.1f} GiB, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
    emit({"kind": "load", "tag": args.tag, "lazy_s": t_lazy, "eval_s": t_eval, "param_gib": total / 2**30,
          "active_gib": mx.get_active_memory() / 2**30, "peak_gib": mx.get_peak_memory() / 2**30,
          "one_shot": args.one_shot_load})
    eos_ids = set(tokenizer.eos_token_ids)
    tokenizer._eos_token_ids = set()  # every run is exactly max-tokens long

    def build(name):
        text = (PROMPTS / f"{name}.txt").read_text()
        if args.no_template:
            return tokenizer.encode(text, add_special_tokens=False)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True, reasoning_effort=args.reasoning_effort)
        return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)

    names = [p for p in args.prompts.split(",") if p]
    prompts = {n: build(n) for n in dict.fromkeys(names + [args.warmup_prompt])}
    for n, ids in prompts.items():
        text = tokenizer.decode(ids)
        log(f"prompt {n}: {len(ids)} tokens; head={text[:140]!r} tail={text[-120:]!r}")
        emit({"kind": "prompt", "tag": args.tag, "prompt": n, "tokens": len(ids), "head": text[:200],
              "tail": text[-160:]})

    def run(name, rep, timed=True):
        ids = prompts[name]
        comp = [(p, r / 2**20, c) for p, r, c in gpulock.big_processes(5_000_000)]
        mx.clear_cache()
        mx.reset_peak_memory()
        pc = make_prompt_cache(model)
        t_start = time.perf_counter()
        t_first = None
        last = None
        tokens = []
        for r in stream_generate(model, tokenizer, ids, max_tokens=args.max_tokens, prompt_cache=pc,
                                 prefill_step_size=args.prefill_step_size):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            last = r
            tokens.append(r.token)
        t_end = time.perf_counter()
        gen_tokens = last.generation_tokens
        decode_wall = t_last - t_first
        text_all = tokenizer.decode(tokens)
        row = {
            "kind": "run" if timed else "warmup", "tag": args.tag, "prompt": name, "rep": rep,
            "prompt_tokens": last.prompt_tokens,
            "prefill_s_api": last.prompt_tokens / last.prompt_tps if last.prompt_tps else None,
            "prompt_tps_api": last.prompt_tps,
            "ttft_wall_s": t_first - t_start,
            "gen_tokens": gen_tokens,
            "generation_tps_api": last.generation_tps,
            "decode_tps_wall": (gen_tokens - 1) / decode_wall if gen_tokens > 1 and decode_wall > 0 else None,
            "decode_wall_s": decode_wall,
            "total_wall_s": t_end - t_start,
            "peak_mem_gb_api": last.peak_memory,
            "peak_mem_gib_mx": mx.get_peak_memory() / 2**30,
            "active_mem_gib_mx": mx.get_active_memory() / 2**30,
            "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
            "finish_reason": last.finish_reason,
            "first_eos_pos": next((i for i, t in enumerate(tokens) if t in eos_ids), None),
            "answer_ok": ("Ishmael" in text_all) and ("Pequod" in text_all),
            "first64": tokenizer.decode(tokens[:64]),
            "first64_ids": tokens[:64],
            "load1": os.getloadavg()[0],
            "competing": comp,
        }
        emit(row)
        log(f"{row['kind']} {name} rep{rep}: prompt {row['prompt_tokens']} tok, prefill {row['prefill_s_api']:.2f}s "
            f"({row['prompt_tps_api']:.1f} tok/s api), TTFT wall {row['ttft_wall_s']:.2f}s, decode {gen_tokens} tok "
            f"{(row['decode_tps_wall'] or 0):.2f} tok/s wall ({last.generation_tps:.2f} api), total {t_end - t_start:.1f}s, "
            f"peak {last.peak_memory:.1f} GB api / {row['peak_mem_gib_mx']:.1f} GiB, rss {row['maxrss_gib']:.1f} GiB, "
            f"eos@{row['first_eos_pos']} ok={row['answer_ok']} load1={row['load1']:.1f} competing={comp}")
        log(f"  first64: {row['first64']!r}")
        return row

    log(f"warm-up with {args.warmup_prompt}")
    run(args.warmup_prompt, 0, timed=False)
    for name in names:
        rows = [run(name, rep) for rep in range(1, args.repeats + 1)]
        summary = {"kind": "summary", "tag": args.tag, "prompt": name, "n": len(rows),
                   "prompt_tokens": rows[0]["prompt_tokens"], "gen_tokens": rows[0]["gen_tokens"]}
        for key in ("prompt_tps_api", "prefill_s_api", "ttft_wall_s", "generation_tps_api", "decode_tps_wall",
                    "total_wall_s", "peak_mem_gib_mx"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                summary[key + "_median"] = statistics.median(vals)
                summary[key + "_min"] = min(vals)
                summary[key + "_max"] = max(vals)
        summary["first64_ids_identical"] = all(r["first64_ids"] == rows[0]["first64_ids"] for r in rows)
        emit(summary)
        log("SUMMARY " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}))
    log("done")


if __name__ == "__main__":
    main()
