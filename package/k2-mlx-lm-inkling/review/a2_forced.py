#!/usr/bin/env python
"""A2 real-model check (runs inside A2's GPU hold).

Teacher-forced decoding of one implementation (``mlx_lm`` = K2's branch, or
``mlx_vlm`` 0.7.11 from ~/.venvs/inkling-base) on the mlx-vlm reference tokens
of K2's hold 2 (bench/results/K2-equiv-mlx_vlm.json), scored against the saved
reference log-probabilities (bench/results/K2-logprobs-mlx_vlm/<run>.npy) with
the same statistics as K2's equiv.py (fp16 logprobs, KL(ref||mine), top-1
agreement), the same prefill chunking and the same forward loop; plus a
per-layer dump (residual stream after every layer, routed expert set and the
6th/7th selection-score gap of every MoE layer, final logits) of chosen decode
steps, so the p_2k step-47 outlier can be located layer by layer across the two
implementations; plus, for mlx_lm, the p_32k chat answer with peak memory.

  python a2_forced.py --impl mlx_lm --runs p_short_raw,p_2k_raw,p_8k_raw \
      --dump p_2k_raw:45,46,47,48 --dump-dir review/a2-dumps --save-logprobs review/a2-logprobs-mlx_lm \
      --p32k --out review/a2-forced-mlx_lm.json
  ~/.venvs/inkling-base/bin/python a2_forced.py --impl mlx_vlm --runs p_short_raw,p_2k_raw \
      --dump p_2k_raw:45,46,47,48 --dump-dir review/a2-dumps --out review/a2-forced-mlx_vlm.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "bench"
RESULTS = BENCH / "results"
PROMPTS = BENCH / "prompts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chunked_eval(params, chunk_gib=4.0):
    """mx.eval in ~4 GiB pieces (a one-shot eval of 82 GB trips the Metal watchdog)."""
    import mlx.core as mx

    chunk, nb, total = [], 0, 0
    t0 = time.perf_counter()
    for v in params:
        chunk.append(v)
        nb += v.nbytes
        if nb > chunk_gib * 2**30:
            mx.eval(chunk)
            total += nb
            chunk, nb = [], 0
    if chunk:
        mx.eval(chunk)
        total += nb
    return total, time.perf_counter() - t0


def logprob_stats(ref, mine, ref_tokens, my_tokens):
    """Same definition as K2's equiv.py: ref/mine are fp16 [steps, V] logprobs."""
    ref = ref.astype(np.float32)
    mine = mine.astype(np.float32)
    p = np.exp(ref)
    kl = (p * (ref - mine)).sum(-1)
    steps = np.arange(ref.shape[0])
    top1 = ref.argmax(-1)
    d_top1 = np.abs(ref[steps, top1] - mine[steps, top1])
    agree = float(np.mean(np.array(ref_tokens) == np.array(my_tokens)))
    return dict(
        kl_mean=float(kl.mean()),
        kl_median=float(np.median(kl)),
        kl_max=float(kl.max()),
        kl_argmax=int(kl.argmax()),
        kl_per_step=[float(v) for v in kl],
        d_top1_mean=float(d_top1.mean()),
        d_top1_max=float(d_top1.max()),
        top1_agreement=agree,
    )


def fmt(st):
    return (
        f"KL mean {st['kl_mean']:.4f} median {st['kl_median']:.4f} max {st['kl_max']:.4f} (step {st['kl_argmax']}); "
        f"|dlogprob| top-1 mean {st['d_top1_mean']:.4f} max {st['d_top1_max']:.4f}; top-1 agreement {100 * st['top1_agreement']:.1f}%"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mlx_vlm", "mlx_lm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", default="p_short_raw,p_2k_raw,p_8k_raw")
    ap.add_argument("--teacher", default=str(RESULTS / "K2-equiv-mlx_vlm.json"))
    ap.add_argument("--ref-logprobs", default=str(RESULTS / "K2-logprobs-mlx_vlm"))
    ap.add_argument("--save-logprobs", default=None)
    ap.add_argument("--dump", default="", help="run:step,step;run:step")
    ap.add_argument("--dump-dir", default=str(HERE / "a2-dumps"))
    ap.add_argument("--p32k", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prefill-step-size", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.model is None:
        args.model = sorted(
            glob.glob(
                os.path.expanduser(
                    "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/"
                )
            )
        )[0]
    dumps = {}
    for part in filter(None, args.dump.split(";")):
        run, steps = part.split(":")
        dumps[run] = [int(s) for s in steps.split(",")]
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx.utils import tree_flatten

    info = dict(
        impl=args.impl,
        python=sys.prefix,
        mlx=mx.__version__,
        model=args.model,
        load1=os.getloadavg()[0],
        git_head=None,
        argv=sys.argv,
    )
    t0 = time.perf_counter()
    if args.impl == "mlx_vlm":
        import mlx_vlm
        from mlx_vlm import load

        model, processor = load(args.model, lazy=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        info["version"] = mlx_vlm.__version__
        info["module"] = mlx_vlm.__file__

        def make_cache():
            return model.language_model.make_cache()

        def forward(ids, cache):
            return model.language_model(ids, cache=cache).logits

        from mlx_vlm.models.inkling import language as L

        def install_hooks(rec):
            orig_layer = L.InklingDecoderLayer.__call__
            orig_route = L.InklingSparseMoE._route

            def layer_hook(self, x, cache=None, conv_mask=None):
                out = orig_layer(self, x, cache=cache, conv_mask=conv_mask)
                rec["layers"].append(out)
                return out

            def route_hook(self, logits):
                idx, topk_w, gamma = orig_route(self, logits)
                scores = mx.sort(
                    mx.sigmoid(logits[:, : self.n_routed].astype(mx.float32))
                    + self.e_score_correction_bias.astype(mx.float32),
                    axis=-1,
                )
                rec["gaps"].append(scores[:, -self.top_k] - scores[:, -self.top_k - 1])
                rec["routes"].append(mx.sort(idx.astype(mx.int32), axis=-1))
                rec["route_weights"].append(topk_w.astype(mx.float32))
                return idx, topk_w, gamma

            L.InklingDecoderLayer.__call__ = layer_hook
            L.InklingSparseMoE._route = route_hook
            return lambda: (
                setattr(L.InklingDecoderLayer, "__call__", orig_layer),
                setattr(L.InklingSparseMoE, "_route", orig_route),
            )

    else:
        import mlx_lm
        from mlx_lm import load
        from mlx_lm.models import inkling_mm_model as M
        from mlx_lm.models.cache import make_prompt_cache

        model, tokenizer = load(args.model, lazy=True)
        info["version"] = mlx_lm.__version__
        info["module"] = mlx_lm.__file__
        try:
            repo = Path(mlx_lm.__file__).resolve().parents[1]
            info["git_head"] = subprocess.run(
                ["git", "-C", str(repo), "log", "-1", "--format=%h %s"], capture_output=True, text=True
            ).stdout.strip()
            info["git_dirty"] = bool(
                subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True).stdout.strip()
            )
        except Exception as e:  # pragma: no cover
            info["git_head"] = f"unknown ({e})"

        def make_cache():
            return make_prompt_cache(model)

        def forward(ids, cache):
            return model(ids, cache=cache)

        def install_hooks(rec):
            orig_layer = M.DecoderLayer.__call__
            orig_sel = M.select_experts

            def layer_hook(self, x, cache=None, rel_index=None):
                out = orig_layer(self, x, cache, rel_index)
                rec["layers"].append(out)
                return out

            def sel_hook(logits, bias, gs, k, rs):
                inds, w = orig_sel(logits, bias, gs, k, rs)
                n_routed = bias.shape[0]
                scores = mx.sort(mx.sigmoid(logits[:, :n_routed]) + bias, axis=-1)
                rec["gaps"].append(scores[:, -k] - scores[:, -k - 1])
                rec["routes"].append(mx.sort(inds[:, :k].astype(mx.int32), axis=-1))
                rec["route_weights"].append(w[:, :k])
                return inds, w

            M.DecoderLayer.__call__ = layer_hook
            M.select_experts = sel_hook
            return lambda: (
                setattr(M.DecoderLayer, "__call__", orig_layer),
                setattr(M, "select_experts", orig_sel),
            )

    t_lazy = time.perf_counter() - t0
    wired = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(wired)
    total, t_eval = chunked_eval([v for _, v in tree_flatten(model.parameters())])
    info.update(load_lazy_s=t_lazy, load_eval_s=t_eval, param_gib=total / 2**30)
    log(f"{args.impl} {info['version']} ({info['module']}, head {info.get('git_head')}): lazy load {t_lazy:.1f}s, "
        f"chunked eval {t_eval:.1f}s, {total / 2**30:.1f} GiB, active {mx.get_active_memory() / 2**30:.1f} GiB")

    teacher = json.load(open(args.teacher))
    teacher_runs = {r["name"]: r for r in teacher["runs"]}

    def tokens_for(name, effort=None):
        text = (PROMPTS / f"{name}.txt").read_text()
        if effort is None:
            return tokenizer.encode(text, add_special_tokens=False)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, reasoning_effort=effort
        )
        return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)

    def dump_step(fn, path, token, position):
        rec = dict(layers=[], routes=[], gaps=[], route_weights=[])
        restore = install_hooks(rec)
        try:
            logits = fn()
            mx.eval(logits, rec["layers"], rec["routes"], rec["gaps"], rec["route_weights"])
        finally:
            restore()
        np.savez(
            path,
            token=token,
            position=position,
            logits=np.array(logits[0, -1].astype(mx.float32)),
            layers=np.stack([np.array(o[0, -1].astype(mx.float32)) for o in rec["layers"]]),
            routes=np.stack([np.array(r[-1]) for r in rec["routes"]]),
            gaps=np.array([float(g[-1].item()) for g in rec["gaps"]]),
            route_weights=np.stack([np.array(w[-1]) for w in rec["route_weights"]]),
        )
        return logits

    def generate(run_name, ids, forced, dump_steps):
        chunk = args.prefill_step_size
        cache = make_cache()
        x = mx.array(ids, dtype=mx.int32)[None]
        n = len(ids)
        pos = 0
        t0 = time.perf_counter()
        while n - pos > 1:
            k = min(chunk, n - pos - 1)
            forward(x[:, pos : pos + k], cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            pos += k
        logits = forward(x[:, pos:], cache)[:, -1, :].astype(mx.float32)
        out, lps, margins = [], [], []
        t_first = None
        for step in range(args.max_tokens):
            y = mx.argmax(logits, axis=-1)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            top2 = mx.sort(logits, axis=-1)[:, -2:]
            mx.eval(y, lp, top2)
            if t_first is None:
                t_first = time.perf_counter()
            out.append(int(y.item()))
            margins.append(float((top2[0, 1] - top2[0, 0]).item()))
            lps.append(np.array(lp[0].astype(mx.float16)))
            nxt = mx.array([[forced[step]]], dtype=mx.int32)
            position = n + step  # absolute position of the fed token
            if step in dump_steps:
                path = dump_dir / f"{args.impl}_{run_name}_step{step:03d}.npz"
                logits = dump_step(lambda: forward(nxt, cache), path, forced[step], position)[:, -1, :].astype(mx.float32)
                log(f"  dumped step {step} (fed token {forced[step]} at position {position}) -> {path.name}")
            else:
                logits = forward(nxt, cache)[:, -1, :].astype(mx.float32)
        t_end = time.perf_counter()
        return dict(tokens=out, fed=list(forced), margins=margins, prefill_s=t_first - t0,
                    decode_s=t_end - t_first, decode_tps=(args.max_tokens - 1) / (t_end - t_first),
                    logprobs=np.stack(lps))

    runs = []
    result = dict(info=info, runs=runs)

    def write():
        result["peak_gib"] = mx.get_peak_memory() / 2**30
        Path(args.out).write_text(json.dumps(result, indent=1))

    for run_name in [r for r in args.runs.split(",") if r]:
        rt = teacher_runs[run_name]
        ids = tokens_for(rt["prompt"], rt["effort"])
        assert len(ids) == rt["prompt_tokens"] and ids[-8:] == rt["prompt_tail_ids"], (run_name, len(ids), ids[-8:])
        forced = rt["tokens"]
        log(f"run {run_name}: {len(ids)} prompt tokens, teacher forced on the mlx-vlm 0.7.11 tokens of hold 2")
        r = generate(run_name, ids, forced, set(dumps.get(run_name, [])))
        lp = r.pop("logprobs")
        if args.save_logprobs:
            d = Path(args.save_logprobs)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{run_name}.npy", lp)
        ref_path = Path(args.ref_logprobs) / f"{run_name}.npy"
        if ref_path.exists():
            ref = np.load(ref_path)
            r["vs_reference"] = logprob_stats(ref, lp, forced, r["tokens"])
            log(f"  vs reference logprobs ({ref_path.parent.name}): {fmt(r['vs_reference'])}")
        r["teacher_agree"] = int(sum(a == b for a, b in zip(r["tokens"], forced)))
        r.update(name=run_name, prompt_tokens=len(ids), text=tokenizer.decode(r["tokens"]))
        log(f"  argmax agrees with the fed tokens at {r['teacher_agree']}/{len(forced)} steps; prefill {r['prefill_s']:.1f}s, "
            f"decode {r['decode_s']:.1f}s ({r['decode_tps']:.1f} tok/s), min margin {min(r['margins']):.3f}")
        runs.append(r)
        write()
        mx.clear_cache()

    if args.p32k and args.impl == "mlx_lm":
        from mlx_lm import stream_generate

        ids = tokens_for("p_32k", "none")
        mx.clear_cache()
        mx.reset_peak_memory()
        toks, t0, t_first = [], time.perf_counter(), None
        last = None
        for resp in stream_generate(model, tokenizer, ids, max_tokens=args.max_tokens, prompt_cache=make_cache(),
                                    prefill_step_size=args.prefill_step_size):
            if t_first is None:
                t_first = time.perf_counter()
            toks.append(resp.token)
            last = resp
        text = tokenizer.decode(toks)
        result["p32k"] = dict(prompt_tokens=len(ids), ttft_s=t_first - t0, prompt_tps=last.prompt_tps,
                              generation_tps=last.generation_tps, n_tokens=len(toks), finish_reason=last.finish_reason,
                              peak_gib=mx.get_peak_memory() / 2**30, active_gib=mx.get_active_memory() / 2**30,
                              answer_ok=("Ishmael" in text and "Pequod" in text), text=text, load1=os.getloadavg()[0])
        log(f"p_32k chat (effort none): {len(ids)} tokens, TTFT {t_first - t0:.1f}s ({last.prompt_tps:.0f} tok/s), "
            f"{len(toks)} tokens at {last.generation_tps:.1f} tok/s, peak {result['p32k']['peak_gib']:.1f} GiB, "
            f"answer_ok={result['p32k']['answer_ok']}, finish={last.finish_reason}")
        log(f"  text: {text[:300]!r}")
        write()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
