"""Greedy generation with one implementation (mlx_vlm or mlx_lm) on the same checkpoint,
driven at the token level (no samplers, no stop tokens): chunked prefill of 2048 tokens,
then argmax decode. Records the token sequence and the top-8 logits of every step so that
two runs can be diffed and a divergence diagnosed (equiv_diff.py).

  python equiv.py --impl mlx_vlm --out ../results/K2-equiv-mlx_vlm.json
  python equiv.py --impl mlx_lm  --out ../results/K2-equiv-mlx_lm.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpulock  # noqa: E402

PROMPTS = HERE.parent / "prompts"


def logprob_stats(ref, mine, ref_tokens, my_tokens):
    """Per-step KL(ref || mine), |dlogprob| of ref's argmax token and over ref's top-100,
    top-1 agreement; ref/mine are [steps, V] float16 log-probabilities on the same prefix."""
    ref = ref.astype(np.float32)
    mine = mine.astype(np.float32)
    p = np.exp(ref)
    kl = (p * (ref - mine)).sum(-1)
    steps = np.arange(ref.shape[0])
    top1 = ref.argmax(-1)
    d_top1 = np.abs(ref[steps, top1] - mine[steps, top1])
    top100 = np.argsort(-ref, axis=-1)[:, :100]
    d_top100 = np.abs(np.take_along_axis(ref, top100, -1) - np.take_along_axis(mine, top100, -1)).max(-1)
    agree = float(np.mean(np.array(ref_tokens) == np.array(my_tokens)))
    return dict(kl_mean=float(kl.mean()), kl_median=float(np.median(kl)), kl_max=float(kl.max()),
                kl_per_step=[float(v) for v in kl], d_top1_mean=float(d_top1.mean()), d_top1_max=float(d_top1.max()),
                d_top100_mean=float(d_top100.mean()), d_top100_max=float(d_top100.max()), top1_agreement=agree)


def fmt_stats(st):
    return (f"KL mean {st['kl_mean']:.4f} median {st['kl_median']:.4f} max {st['kl_max']:.4f} nats; "
            f"|dlogprob| top-1 mean {st['d_top1_mean']:.4f} max {st['d_top1_max']:.4f}, top-100 max {st['d_top100_max']:.4f}; "
            f"top-1 agreement {100 * st['top1_agreement']:.1f}%")


def dump_hidden(impl, model, ids, out_dir):
    """Cache-free forward over ``ids``: the residual stream after the embedding norm and after
    every layer, plus each MoE layer's routed expert set, as .npy files."""
    import mlx.core as mx

    out_dir.mkdir(parents=True, exist_ok=True)
    x = mx.array(ids, dtype=mx.int32)[None]
    routes = {}
    if impl == "mlx_lm":
        from mlx_lm.models import inkling_mm_model as M

        inner = model.model.llm
        h = inner.embed_tokens(x)
        layers = inner.layers
        orig = M.select_experts

        def hooked(logits, bias, gs, k, rs, _i=[0]):
            inds, w = orig(logits, bias, gs, k, rs)
            routes.setdefault(len(routes), np.array(mx.sort(inds[:, :k], axis=-1)))
            return inds, w

        M.select_experts = hooked
    else:
        from mlx_vlm.models.inkling import language as L

        lm = model.language_model
        h = lm.model.embed(x)
        layers = lm.model.layers
        orig = L.InklingSparseMoE._route

        def hooked(self, logits):
            idx, topk_w, gamma = orig(self, logits)
            routes.setdefault(len(routes), np.array(mx.sort(idx.astype(mx.int32), axis=-1)))
            return idx, topk_w, gamma

        L.InklingSparseMoE._route = hooked
    try:
        np.save(out_dir / f"{impl}_embed.npy", np.array(h.astype(mx.float32)))
        for i, layer in enumerate(layers):
            h = layer(h, None) if impl == "mlx_lm" else layer(h, cache=None)
            mx.eval(h)
            np.save(out_dir / f"{impl}_layer{i:02d}.npy", np.array(h.astype(mx.float32)))
    finally:
        if impl == "mlx_lm":
            M.select_experts = orig
        else:
            L.InklingSparseMoE._route = orig
    for j, r in routes.items():
        np.save(out_dir / f"{impl}_route{j:02d}.npy", r)


def router_audit(model, seq):
    """Teacher-forced pass over ``seq`` with every MoE layer routing twice: fp32 logits (this
    model) and bf16 logits (mlx-vlm / transformers style, the bf16 matmul output rounded). Records
    per layer how many tokens get a different expert set and the fp32 selection-score gap
    between the 6th and 7th routed expert for those tokens (bf16 resolves ~2^-8 near 1)."""
    import mlx.core as mx
    from mlx_lm.models import inkling_mm_model as M
    from mlx_lm.models.cache import make_prompt_cache

    recs = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer.mlp, M.MoE):
            recs[id(layer.mlp)] = dict(layer=i, n=0, differ=0, gaps_differ=[], gaps_all=[])
    orig_call = M.MoE.__call__

    def hooked(self, x):
        rec = recs[id(self)]
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        w = self.gate.weight
        logits32 = xf.astype(mx.float32) @ w.astype(mx.float32).T
        logits16 = (xf @ w.astype(xf.dtype).T).astype(mx.float32)
        inds32, weights = M.select_experts(logits32, self.gate.bias, self.gate.global_scale, self.top_k, self.route_scale)
        inds16, _ = M.select_experts(logits16, self.gate.bias, self.gate.global_scale, self.top_k, self.route_scale)
        n_routed = self.gate.bias.shape[0]
        scores = mx.sort(mx.sigmoid(logits32[:, :n_routed]) + self.gate.bias, axis=-1)
        gap = scores[:, -self.top_k] - scores[:, -self.top_k - 1]
        differ = mx.any(mx.sort(inds32[:, : self.top_k], axis=-1) != mx.sort(inds16[:, : self.top_k], axis=-1), axis=-1)
        mx.eval(differ, gap)
        d = differ.tolist()
        g = gap.tolist()
        rec["n"] += len(d)
        rec["differ"] += sum(d)
        rec["gaps_differ"] += [gv for dv, gv in zip(d, g) if dv]
        rec["gaps_all"] += g
        return orig_call(self, x)  # the model's own forward for the output (any layout)

    M.MoE.__call__ = hooked
    try:
        x = mx.array(seq, dtype=mx.int32)[None]
        cache = make_prompt_cache(model)
        for a in range(0, x.shape[1], 2048):
            mx.eval(model(x[:, a : a + 2048], cache=cache))
    finally:
        M.MoE.__call__ = orig_call
    stats = list(recs.values())
    total = sum(r["differ"] for r in stats)
    pairs = sum(r["n"] for r in stats)
    gaps = sorted(g for r in stats for g in r["gaps_differ"])
    all_gaps = sorted(g for r in stats for g in r["gaps_all"])
    summary = (f"{total}/{pairs} (token, layer) routing decisions differ between fp32 and bf16 router logits "
               f"({100 * total / max(pairs, 1):.2f}%); 6th/7th selection-score gap when they differ: "
               f"max {max(gaps) if gaps else 0:.5f}, median {gaps[len(gaps) // 2] if gaps else 0:.5f} "
               f"(bf16 resolution near 1 is 0.0039); median gap over all decisions {all_gaps[len(all_gaps) // 2]:.5f}")
    per_layer = [dict(layer=r["layer"], n=r["n"], differ=r["differ"],
                      max_gap_differ=(max(r["gaps_differ"]) if r["gaps_differ"] else None)) for r in stats]
    return dict(summary=summary, total_differ=total, pairs=pairs, n_tokens=stats[0]["n"] if stats else 0,
                gaps_differ_max=(max(gaps) if gaps else 0.0), gaps_differ=gaps[:2000], per_layer=per_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mlx_vlm", "mlx_lm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompts", default="p_short,p_2k,p_8k")
    ap.add_argument("--chat-prompt", default="p_short")
    ap.add_argument("--chat-efforts", default="none,low")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prefill-step-size", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=None,
                    help="equiv JSON whose tokens are fed instead of this model's argmax (same run names)")
    ap.add_argument("--router-audit", action="store_true",
                    help="mlx_lm only: compare fp32 and bf16 routing on the prompt + generated tokens")
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    teacher = json.load(open(args.teacher)) if args.teacher else None
    teacher_runs = {r["name"]: r for r in teacher["runs"]} if teacher else {}
    # Extras configured per (impl, teacher-forced or not) in K2/equiv_extras.json, so a queued
    # command line can be extended without editing the running script:
    #   save_logprobs: dir   per-step fp16 logprobs of the greedy run -> DIR/<run>.npy
    #   noise_floor: true    re-run each prompt teacher-forced on its own tokens with a 512
    #                        prefill chunk and record the KL to the first pass
    #   hidden: dir          per-layer hidden states and routed expert sets of a cache-free
    #                        forward over the first prompt -> DIR/<impl>_*.npy
    #   ref_logprobs: dir    forced runs: KL / logprob stats against the saved reference
    extras = {}
    extras_path = HERE / "equiv_extras.json"
    if extras_path.exists():
        allx = json.load(open(extras_path))
        mode = "forced" if teacher else "greedy"
        base = os.path.basename(os.path.normpath(args.model)) if args.model else ""
        extras = allx.get(f"{args.impl}@{base}:{mode}", allx.get(f"{args.impl}:{mode}", {}))
    gpulock.log(f"extras: {extras}")
    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    if not args.no_lock:
        gpulock.acquire(f"equiv-{args.impl}")

    import mlx.core as mx
    from mlx.utils import tree_flatten

    t0 = time.perf_counter()
    if args.impl == "mlx_vlm":
        import mlx_vlm
        from mlx_vlm import load

        model, processor = load(args.model, lazy=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        version = mlx_vlm.__version__

        def make_cache():
            return model.language_model.make_cache()

        def forward(ids, cache):
            return model.language_model(ids, cache=cache).logits

    else:
        import mlx_lm
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        model, tokenizer = load(args.model, lazy=True)
        version = mlx_lm.__version__

        def make_cache():
            return make_prompt_cache(model)

        def forward(ids, cache):
            return model(ids, cache=cache)

    t_lazy = time.perf_counter() - t0
    # Keep the weights resident on the GPU (what stream_generate does); without it the
    # driver pages the 82 GB in and out under memory pressure and decode drops to <1 tok/s
    wired = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(wired)
    total, t_eval = gpulock.chunked_eval([v for _, v in tree_flatten(model.parameters())])
    gpulock.log(f"{args.impl} {version}: lazy load {t_lazy:.1f}s, chunked eval {t_eval:.1f}s, "
                f"{total / 2**30:.1f} GiB, active {mx.get_active_memory() / 2**30:.1f} GiB")

    def tokens_for(name, effort=None):
        text = (PROMPTS / f"{name}.txt").read_text()
        if effort is None:
            return tokenizer.encode(text, add_special_tokens=False)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
            reasoning_effort=effort)
        return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)

    def generate(ids, forced=None, chunk=None, keep_logprobs=False):
        chunk = chunk or args.prefill_step_size
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
        out, top8, margins, lps = [], [], [], []
        t_first = None
        for step in range(args.max_tokens):
            y = mx.argmax(logits, axis=-1)
            part = mx.argpartition(-logits, kth=7, axis=-1)[:, :8]
            vals = mx.take_along_axis(logits, part, axis=-1)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            mx.eval(y, part, vals, lp)
            if t_first is None:
                t_first = time.perf_counter()
            tok = int(y.item())
            pairs = sorted(zip(part[0].tolist(), vals[0].tolist()), key=lambda p: -p[1])
            out.append(tok)
            top8.append([[int(i), float(v)] for i, v in pairs])
            margins.append(float(pairs[0][1] - pairs[1][1]))
            if keep_logprobs:
                lps.append(np.array(lp[0].astype(mx.float16)))
            nxt = y.reshape(1, 1) if forced is None else mx.array([[forced[step]]], dtype=mx.int32)
            logits = forward(nxt, cache)[:, -1, :].astype(mx.float32)
        t_end = time.perf_counter()
        r = dict(tokens=out, fed=(list(forced) if forced is not None else None), top8=top8,
                 margins=margins, prefill_s=t_first - t0, decode_s=t_end - t_first, chunk=chunk)
        if keep_logprobs:
            r["logprobs"] = np.stack(lps)
        return r

    runs = []
    specs = [(f"{p}_raw", p, None) for p in args.prompts.split(",") if p]
    specs += [(f"{args.chat_prompt}_chat_{e}", args.chat_prompt, e)
              for e in args.chat_efforts.split(",") if e]

    def write():
        result = dict(impl=args.impl, version=version, mlx=mx.__version__, model=args.model,
                      python=sys.prefix, wired_limit_gib=wired / 2**30,
                      max_tokens=args.max_tokens, prefill_step_size=args.prefill_step_size,
                      load_lazy_s=t_lazy, load_eval_s=t_eval, param_gib=total / 2**30,
                      peak_gib=mx.get_peak_memory() / 2**30, runs=runs)
        Path(args.out).write_text(json.dumps(result, indent=1))

    for run_name, prompt, effort in specs:
        ids = tokens_for(prompt, effort)
        forced = teacher_runs[run_name]["tokens"] if teacher_runs else None
        gpulock.log(f"run {run_name}: {len(ids)} prompt tokens" + (" (teacher forced)" if forced else ""))
        want_lp = bool(extras.get("save_logprobs") or extras.get("noise_floor") or extras.get("ref_logprobs"))
        r = generate(ids, forced, keep_logprobs=want_lp)
        lp_mine = r.pop("logprobs", None)
        if extras.get("save_logprobs") and lp_mine is not None:
            d = Path(extras["save_logprobs"]); d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{run_name}.npy", lp_mine)
        if extras.get("noise_floor") and lp_mine is not None:
            # the same model, teacher forced on its own tokens, with a different prefill chunk
            r2 = generate(ids, r["tokens"], chunk=128, keep_logprobs=True)
            r["noise_floor"] = logprob_stats(lp_mine, r2.pop("logprobs"), r["tokens"], r2["tokens"])
            r["noise_floor"]["chunks"] = [r["chunk"], 128]
            gpulock.log(f"  noise floor (self, prefill chunk {r['chunk']} vs 128): {fmt_stats(r['noise_floor'])}")
        if extras.get("ref_logprobs") and lp_mine is not None:
            ref = np.load(Path(extras["ref_logprobs"]) / f"{run_name}.npy")
            r["vs_reference"] = logprob_stats(ref, lp_mine, forced, r["tokens"])
            gpulock.log(f"  vs reference (same tokens): {fmt_stats(r['vs_reference'])}")
        if forced is not None:
            agree = sum(a == b for a, b in zip(r["tokens"], forced))
            r["teacher_agree"] = agree
            gpulock.log(f"  argmax agrees with the fed tokens at {agree}/{len(forced)} steps")
        if args.router_audit and args.impl == "mlx_lm" and prompt in ("p_short", "p_2k"):
            seq = ids + (forced if forced is not None else r["tokens"])
            r["router_audit"] = router_audit(model, seq)
            gpulock.log(f"  router audit: {r['router_audit']['summary']}")
        r.update(name=run_name, prompt=prompt, effort=effort, prompt_tokens=len(ids),
                 prompt_tail_ids=ids[-8:], text=tokenizer.decode(r["tokens"]))
        gpulock.log(f"  prefill {r['prefill_s']:.1f}s, decode {r['decode_s']:.1f}s "
                    f"({(args.max_tokens - 1) / r['decode_s']:.1f} tok/s), min margin "
                    f"{min(r['margins']):.3f}")
        gpulock.log(f"  text: {r['text'][:200]!r}")
        runs.append(r)
        write()  # incremental, so an interrupted run keeps its finished prompts
        mx.clear_cache()
    if extras.get("hidden"):
        dump_hidden(args.impl, model, tokens_for(args.prompts.split(",")[0]), Path(extras["hidden"]))
        gpulock.log(f"hidden states and routes written to {extras['hidden']}")
    gpulock.log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
