"""mlx-vlm 0.7.11 counterpart of profile_decode.py (run from ~/.venvs/inkling-base): where a
decode step goes with the real weights, same method (p_short prefill, 300 decode steps, then
medians of 10x-chained pieces)."""
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
    import mlx_vlm
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    from mlx_vlm.models.inkling import language as L

    model, processor = load(args.model, lazy=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    gpulock.chunked_eval([v for _, v in tree_flatten(model.parameters())])
    lm = model.language_model
    ids = tokenizer.encode((PROMPTS / "p_short.txt").read_text(), add_special_tokens=False)
    cache = lm.make_cache()
    x = mx.array(ids, dtype=mx.int32)[None]
    lm(x[:, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    y = x[:, -1:]
    for _ in range(300):
        y = mx.argmax(lm(y, cache=cache).logits[:, -1:, :], axis=-1)
        mx.eval(y)
    gpulock.log(f"mlx_vlm {mlx_vlm.__version__}, context {cache[0][0].offset} tokens, load1 {os.getloadavg()[0]:.1f}")

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

    h = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    mx.eval(h)
    gpulock.log("whole decode step (language model forward, 1 token):")
    step = timeit("lm(y, cache).logits", lambda: lm(y, cache=cache).logits, 1)
    gpulock.log("head and layer kinds (chained x10):")
    timeit("final norm + /16 + lm_head + slice", lambda: [lm._logits_from_norm(lm.model.norm(h)) for _ in range(args.chain)], args.chain)
    layers = lm.model.layers
    i_sl = next(i for i, l in enumerate(layers) if l.self_attn.sliding > 0 and isinstance(l.mlp, L.InklingSparseMoE))
    i_gl = next(i for i, l in enumerate(layers) if l.self_attn.sliding == 0)
    sl, gl, dl = layers[i_sl], layers[i_gl], layers[0]
    c_sl, c_gl, c_dl = cache[i_sl], cache[i_gl], cache[0]

    def chain(layer, c):
        out = []
        hh = h
        for _ in range(args.chain):
            hh = layer(hh, cache=c)
            out.append(hh)
        return out

    t_sl = timeit("sliding MoE layer (x10, cache grows)", lambda: chain(sl, c_sl), args.chain)
    t_gl = timeit("global MoE layer (x10)", lambda: chain(gl, c_gl), args.chain)
    t_dl = timeit("dense sliding layer 0 (x10)", lambda: chain(dl, c_dl), args.chain)
    gpulock.log(f"  => 33 sliding MoE + 7 global MoE + 2 dense = {33 * t_sl + 7 * t_gl + 2 * t_dl:.1f} ms of {step:.1f} ms per step")
    gpulock.log("inside one sliding MoE layer (chained x10):")
    a = sl.self_attn
    conv = c_sl[1]
    kv = c_sl[0]
    timeit("input_layernorm", lambda: [sl.input_layernorm(h) for _ in range(args.chain)], args.chain)
    timeit("qkvr_proj GEMV [4096 -> 6656] bf16", lambda: [a.qkvr_proj(h) for _ in range(args.chain)], args.chain)
    kx = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16); mx.eval(kx)
    timeit("k_sconv + v_sconv kernels (2 x 1024 ch)", lambda: [(a.k_sconv(kx, cache=conv), a.v_sconv(kx, cache=conv)) for _ in range(args.chain)], args.chain)
    q = mx.random.normal((1, 1, 32, 128)).astype(mx.bfloat16); k = mx.random.normal((1, 1, 8, 128)).astype(mx.bfloat16); mx.eval(q, k)
    timeit("q_norm + k_norm", lambda: [(a.q_norm(q), a.k_norm(k)) for _ in range(args.chain)], args.chain)
    kk = mx.random.normal((1, 8, 1, 128)).astype(mx.bfloat16); mx.eval(kk)
    timeit("KVCache.update_and_fetch (full history, k and v)", lambda: [kv.update_and_fetch(kk, kk)[0] for _ in range(args.chain)], args.chain)
    S = min(kv.offset, 512)
    r = mx.random.normal((1, 1, 32, 16)).astype(mx.bfloat16); mx.eval(r)
    kview = kv.keys[:, :, :S]
    timeit(f"banded mask kernel (L=1, S={S})", lambda: [L.banded_additive_mask(r, a.rel_proj, S - 1, S, 512, 512, shape_ref=kview) for _ in range(args.chain)], args.chain)
    qq = mx.random.normal((1, 32, 1, 128)).astype(mx.bfloat16); mask = mx.random.normal((1, 32, 1, S)).astype(mx.bfloat16); mx.eval(qq, mask)
    timeit(f"SDPA (32q/8kv heads, {S} keys, additive mask)", lambda: [mx.fast.scaled_dot_product_attention(qq, kview, kview, scale=1 / 128, mask=mask) for _ in range(args.chain)], args.chain)
    o = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16); mx.eval(o)
    timeit("o_proj GEMV [4096 -> 4096] bf16", lambda: [a.o_proj(o) for _ in range(args.chain)], args.chain)
    timeit("attn_sconv kernel with residual (4096 ch)", lambda: [sl.attn_sconv(o, cache=conv, residual=h) for _ in range(args.chain)], args.chain)
    timeit("post_attention_layernorm", lambda: [sl.post_attention_layernorm(h) for _ in range(args.chain)], args.chain)
    m = sl.mlp
    xf = h.reshape(-1, 4096)
    timeit("router GEMV bf16 [4096 -> 258]", lambda: [xf @ m.gate_weight.T for _ in range(args.chain)], args.chain)
    lg = xf @ m.gate_weight.T; mx.eval(lg)
    timeit("route kernel", lambda: [m._route(lg)[0] for _ in range(args.chain)], args.chain)
    idx, topk_w, gamma = m._route(lg); mx.eval(idx, topk_w, gamma)
    timeit("switch_mlp (6 experts: 3 gather_qmm 2-bit + swiglu)", lambda: [m.switch_mlp(xf, idx) for _ in range(args.chain)], args.chain)
    yr = m.switch_mlp(xf, idx); mx.eval(yr)
    timeit("weighted sum of the 6 expert outputs", lambda: [(yr * topk_w[..., None]).sum(axis=-2) for _ in range(args.chain)], args.chain)
    timeit("shared experts dense (3 quantized GEMV + swiglu)", lambda: [m.shared_experts(xf, gamma) for _ in range(args.chain)], args.chain)
    timeit("MoE block total (mlp(x))", lambda: [m(h) for _ in range(args.chain)], args.chain)
    timeit("mlp_sconv kernel with residual (4096 ch)", lambda: [sl.mlp_sconv(o, cache=conv, residual=h) for _ in range(args.chain)], args.chain)
    gpulock.log("done")


if __name__ == "__main__":
    main()
