"""Where a decode step goes with the real weights (mlx-lm Inkling, batch 1): the whole step,
then the head and the three layer kinds, then the pieces inside one sliding MoE layer. Each
number is the median of synchronous evals of the piece chained 10x (dispatch amortized) after
a warm-up, with the sliding window full (prefill of p_short, 226 tokens, then 300 decode steps).

  python profile_decode.py --model DIR --out ../results/K2-profile-decode.log
"""
import argparse
import glob
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpulock  # noqa: E402

PROMPTS = HERE.parent / "prompts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chain", type=int, default=10)
    args = ap.parse_args()
    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models import inkling_mm_model as M
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model, lazy=True)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    gpulock.chunked_eval([v for _, v in tree_flatten(model.parameters())])
    ids = tokenizer.encode((PROMPTS / "p_short.txt").read_text(), add_special_tokens=False)
    cache = make_prompt_cache(model)
    x = mx.array(ids, dtype=mx.int32)[None]
    logits = model(x[:, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    y = x[:, -1:]
    for _ in range(300):  # fill the sliding windows
        y = mx.argmax(model(y, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(y)
    gpulock.log(f"context {cache[0][0].offset} tokens, load1 {os.getloadavg()[0]:.1f}")

    def timeit(name, fn, chain=1):
        ts = []
        for i in range(args.iters + 3):
            t0 = time.perf_counter()
            mx.eval(fn())
            t = time.perf_counter() - t0
            if i >= 3:
                ts.append(t)
        med = statistics.median(ts) * 1e3 / chain
        gpulock.log(f"  {name:58s} {med:8.3f} ms  (best {min(ts) * 1e3 / chain:.3f})")
        return med

    inner = model.model.llm
    h = mx.random.normal((1, 1, inner.args.hidden_size)).astype(mx.bfloat16)
    mx.eval(h)

    gpulock.log("whole decode step (model forward, 1 token, cache full):")
    step = timeit("model(y, cache)", lambda: model(y, cache=cache), 1)

    gpulock.log("head and layer kinds (chained x10):")
    timeit("final norm + /16 + unembed + slice", lambda: [inner.unembed(inner.norm(h) * (1 / 16))[..., :200058] for _ in range(args.chain)], args.chain)
    i_sl = next(i for i, l in enumerate(inner.layers) if l.attn.window > 0 and isinstance(l.mlp, M.MoE))
    i_gl = next(i for i, l in enumerate(inner.layers) if l.attn.window == 0)
    sl, gl, dl = inner.layers[i_sl], inner.layers[i_gl], inner.layers[0]
    c_sl, c_gl, c_dl = cache[i_sl], cache[i_gl], cache[0]
    rel_index = sl.attn.slot_rel_index(c_sl[0])

    def layer_chain(layer, c, ri):
        out = []
        hh = h
        for _ in range(args.chain):
            hh = layer(hh, c, ri)
            out.append(hh)
        return out

    t_sl = timeit("sliding MoE layer (x10, cache grows)", lambda: layer_chain(sl, c_sl, rel_index), args.chain)
    t_gl = timeit("global MoE layer (x10)", lambda: layer_chain(gl, c_gl, None), args.chain)
    t_dl = timeit("dense sliding layer 0 (x10)", lambda: layer_chain(dl, c_dl, rel_index), args.chain)
    gpulock.log(f"  => 33 sliding MoE + 7 global MoE + 2 dense = {33 * t_sl + 7 * t_gl + 2 * t_dl:.1f} ms of {step:.1f} ms per step")

    gpulock.log("inside one sliding MoE layer (chained x10):")
    a = sl.attn
    conv = c_sl[1]
    kv = c_sl[0]
    timeit("attn_norm", lambda: [sl.attn_norm(h) for _ in range(args.chain)], args.chain)
    timeit("qkvr_proj GEMV [4096 -> 6656] bf16", lambda: [a.qkvr_proj(h) for _ in range(args.chain)], args.chain)
    kvx = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16); mx.eval(kvx)
    timeit("kv short conv kernel (2048 ch)", lambda: [a.kv_sconv(kvx, conv, None) for _ in range(args.chain)], args.chain)
    q = mx.random.normal((1, 1, 32, 128)).astype(mx.bfloat16); k = mx.random.normal((1, 1, 8, 128)).astype(mx.bfloat16); mx.eval(q, k)
    timeit("q_norm + k_norm", lambda: [(a.q_norm(q), a.k_norm(k)) for _ in range(args.chain)], args.chain)
    kk = mx.random.normal((1, 8, 1, 128)).astype(mx.bfloat16); mx.eval(kk)
    timeit("RotatingKVCache.update_and_fetch (k and v)", lambda: [kv.update_and_fetch(kk, kk)[0] for _ in range(args.chain)], args.chain)
    r = mx.random.normal((1, 1, 32, 16)).astype(mx.bfloat16); mx.eval(r)
    timeit("rel matmul [32,16]@[16,512] + gather", lambda: [mx.take_along_axis((r @ a.rel_logits_proj.proj_reversed).transpose(0, 2, 1, 3), rel_index, axis=-1) for _ in range(args.chain)], args.chain)
    qq = mx.random.normal((1, 32, 1, 128)).astype(mx.bfloat16); kw, vw = kv.keys_and_values(); mask = mx.random.normal((1, 32, 1, kw.shape[2])).astype(mx.bfloat16); mx.eval(qq, mask)
    timeit(f"SDPA (32q/8kv heads, {kw.shape[2]} keys, additive mask)", lambda: [mx.fast.scaled_dot_product_attention(qq, kw, vw, scale=1 / 128, mask=mask) for _ in range(args.chain)], args.chain)
    o = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16); mx.eval(o)
    timeit("wo_ud GEMV [4096 -> 4096] bf16", lambda: [a.wo_ud(o) for _ in range(args.chain)], args.chain)
    timeit("attn_sconv kernel with residual (4096 ch)", lambda: [sl.attn_sconv(o, conv, None, residual=h) for _ in range(args.chain)], args.chain)
    timeit("mlp_norm", lambda: [sl.mlp_norm(h) for _ in range(args.chain)], args.chain)
    m = sl.mlp
    xf = h.reshape(-1, 4096)
    timeit("router GEMV fp32 [4096 -> 258]", lambda: [xf.astype(mx.float32) @ m.gate.weight.T for _ in range(args.chain)], args.chain)
    lg = xf.astype(mx.float32) @ m.gate.weight.T; mx.eval(lg)
    timeit("route kernel (top-6 + softmax)", lambda: [M.select_experts(lg, m.gate.bias, m.gate.global_scale, m.top_k, m.route_scale)[0] for _ in range(args.chain)], args.chain)
    inds, w = M.select_experts(lg, m.gate.bias, m.gate.global_scale, m.top_k, m.route_scale); mx.eval(inds, w)
    timeit("experts SwitchGLU (8 experts: 3 gather_qmm 2-bit + swiglu)", lambda: [m.experts(xf, inds) for _ in range(args.chain)], args.chain)
    ye = m.experts(xf, inds); mx.eval(ye)
    k = ye.shape[-2]
    timeit(f"weighted sum of the {k} routed expert outputs", lambda: [(ye * w[:, :k, None].astype(ye.dtype)).sum(axis=-2) for _ in range(args.chain)], args.chain)
    if hasattr(m, "shared_experts"):
        timeit("shared experts dense (3 quantized GEMV + swiglu)", lambda: [m.shared_experts(xf, w[:, k:]) for _ in range(args.chain)], args.chain)
    timeit("MoE block total (mlp(x))", lambda: [m(h) for _ in range(args.chain)], args.chain)
    timeit("mlp_sconv kernel with residual (4096 ch)", lambda: [sl.mlp_sconv(o, conv, None, residual=h) for _ in range(args.chain)], args.chain)
    # Does the generate loop overlap graph building with GPU execution for this model?
    from mlx_lm.generate import generate_step

    def loop(n, pipelined):
        c = make_prompt_cache(model)
        mx.eval(model(x[:, :-1], cache=c))
        mx.eval([cc.state for cc in c])
        t0 = time.perf_counter()
        if pipelined:
            for _ in generate_step(x[0, -1:], model, max_tokens=n, prompt_cache=c):
                pass
        else:
            tok = x[:, -1:]
            for _ in range(n):
                tok = mx.argmax(model(tok, cache=c)[:, -1:, :], axis=-1)
                mx.eval(tok)
        return n / (time.perf_counter() - t0)

    gpulock.log("decode loops over 64 tokens after the p_short prefill:")
    for pipelined in (True, False, True):
        kind = "generate_step (async_eval pipelined)" if pipelined else "synchronous mx.eval per token"
        gpulock.log(f"  {kind:58s} {loop(64, pipelined):8.2f} tok/s")
    gpulock.log("done")


if __name__ == "__main__":
    main()
