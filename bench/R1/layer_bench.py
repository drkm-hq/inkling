#!/usr/bin/env python
"""Synthetic per-layer benchmark of mlx-vlm's Inkling implementation at exact
Inkling-Small shapes, with RANDOM weights (no checkpoint needed).

The text model is instantiated through the real mlx_vlm classes
(mlx_vlm.models.inkling.language.LanguageModel) with num_hidden_layers=3:
  layer 0: dense MLP (16384) + sliding-window attention   (stands for real layers 0-1)
  layer 1: MoE (256 experts top-6 + 2 shared) + sliding    (stands for real layers 2,3,4,6,... : 33 layers)
  layer 2: MoE + global attention (rel_extent 1024, log-scaling branch)  (real 5,11,...,41 : 7 layers)
Everything else is the real config.json of mlx-community/Inkling-Small-mlx-2bit.

Quantization mirrors the mlx-community checkpoints (inkling_mlx/convert.py
`experts_only`: gate/up/down of routed experts, shared experts AND the dense
MLP at the tier's bits, group 64; attention, embeddings and lm_head bf16).
--attn-bits / --head-bits add the hypothetical recipes.

Timing: every number is the median of `reps` synchronous mx.eval() runs after
warmup. Per-layer decode numbers are measured as a chain of 8 dependent layer
calls inside ONE eval (so the ~0.3 ms command-buffer latency is amortized the
way it is inside a real 42-layer forward). The full 3-layer decode loop is
timed exactly the way generate_step pipelines it (mx.async_eval of token n+1
before waiting on token n).

Usage: python layer_bench.py --experts-bits 2 [--attn-bits 8] [--head-bits 8]
                             [--contexts 0,2048,8192,32768] [--prefill 512,2048,8192]
Results -> results/layer_bench_e{E}_a{A}_h{H}.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling import language as L
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.switch_layers import SwitchLinear

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

LAYER_MULT = {0: 2, 1: 33, 2: 7}  # how many real layers each synthetic layer stands for
LAYER_NAME = {0: "L0 dense+SWA", 1: "L1 MoE+SWA", 2: "L2 MoE+global"}


def log(*a):
    print(*a, flush=True)


def make_config():
    return TextConfig(
        model_type="inkling",
        hidden_size=4096,
        num_hidden_layers=3,
        vocab_size=201024,
        unpadded_vocab_size=200058,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_embed_norm=True,
        logits_mup_width_multiplier=16.0,
        max_position_embeddings=1048576,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        swa_num_attention_heads=32,
        swa_num_key_value_heads=8,
        swa_head_dim=128,
        sliding_window_size=512,
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        d_rel=16,
        rel_extent=1024,
        log_scaling_n_floor=128000,
        log_scaling_alpha=0.1,
        sconv_kernel_size=4,
        dense_mlp_idx=2,
        intermediate_size=2048,
        dense_intermediate_size=16384,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=2,
        route_scale=8.0,
    )


def qbytes(nparams, bits):
    if bits is None:
        return nparams * 2
    return nparams * bits / 8 + nparams / 64 * 4


def build_model(experts_bits, attn_bits, head_bits, seed=0, dense_scale_dtype=mx.bfloat16):
    """dense_scale_dtype: dtype of InklingDenseMLP.global_scale. The mlx-community
    checkpoints store it BF16 (blob header: model.llm.layers.0.mlp.global_scale BF16 [1]);
    a float32 value silently promotes the residual stream to fp32 (see diag_loop.py)."""
    mx.random.seed(seed)
    cfg = make_config()
    model = L.LanguageModel(cfg)
    for layer in model.model.layers:
        at = layer.self_attn
        at.rel_proj = mx.random.normal(at.rel_proj.shape) * 0.05
        if isinstance(layer.mlp, L.InklingSparseMoE):
            layer.mlp.gate_weight = mx.random.normal(layer.mlp.gate_weight.shape) * 0.05
            layer.mlp.e_score_correction_bias = mx.random.normal((cfg.n_routed_experts,)) * 0.01
    model.set_dtype(mx.bfloat16)

    def pred(path, m):
        if isinstance(m, SwitchLinear):
            return {"group_size": 64, "bits": experts_bits}
        if isinstance(m, nn.Embedding):
            return False
        if isinstance(m, nn.Linear):
            if path.endswith("lm_head"):
                return False if head_bits is None else {"group_size": 64, "bits": head_bits}
            if ".self_attn." in path:
                return False if attn_bits is None else {"group_size": 64, "bits": attn_bits}
            if ".mlp." in path:  # dense MLP and the dense-ified shared experts
                return {"group_size": 64, "bits": experts_bits}
        return False

    nn.quantize(model, class_predicate=pred)
    for layer in model.model.layers:
        if isinstance(layer.mlp, L.InklingSparseMoE):
            # router scale (model.llm.layers.N.mlp.gate.global_scale) is F32 in the checkpoint;
            # it only feeds the fp32 _wscale, so it cannot promote activations.
            layer.mlp.global_scale = mx.ones((1,), dtype=mx.float32)
            layer.mlp.e_score_correction_bias = layer.mlp.e_score_correction_bias.astype(mx.float32)
            layer.mlp.switch_mlp.gate_scale = mx.ones((cfg.n_routed_experts,), dtype=mx.float32)
            layer.mlp.switch_mlp.out_scale = mx.ones((cfg.n_routed_experts,), dtype=mx.float32)
        else:
            layer.mlp.global_scale = mx.ones((1,), dtype=dense_scale_dtype)
    mx.eval(model.parameters())
    model.eval()
    return model, cfg


def rand_h(T):
    return mx.random.normal((1, T, 4096)).astype(mx.bfloat16)


def fill_cache(c, ctx, conv_dims=(1024, 1024, 4096, 4096)):
    kv, conv = c[0], c[1]
    if ctx > 0:
        k = mx.random.normal((1, 8, ctx, 128)).astype(mx.bfloat16)
        v = mx.random.normal((1, 8, ctx, 128)).astype(mx.bfloat16)
        kv.update_and_fetch(k, v)
        for i, C in enumerate(conv_dims):
            conv[i] = mx.random.normal((1, 3, C))
        mx.eval(kv.keys, kv.values, conv.cache)


def make_caches(model, ctx):
    caches = model.make_cache()
    for c in caches:
        fill_cache(c, ctx)
    return caches


def med(run, warm=3, reps=11):
    for _ in range(warm):
        mx.eval(run())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(run())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def gpu_warmup(seconds=4.0):
    """Burn the GPU for a few seconds so clocks are up before timing (the first
    measurements after a build otherwise run 1.5-2x slow)."""
    a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    mx.eval(a)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = (a @ a) * 1e-3
        mx.eval(a)


def chained(fn, x0, chain=12, warm=3, reps=11):
    def run():
        x = x0
        for _ in range(chain):
            x = fn(x)
        return x

    return med(run, warm, reps) / chain


def time_full_decode(model, caches, n=24):
    """Per-token wall time of the 3-layer model, pipelined like generate_step."""

    def step(y):
        out = model(y, cache=caches)
        return mx.argmax(out.logits[:, -1, :], axis=-1)

    y = mx.array([[1234]])
    nxt = step(y)
    mx.eval(nxt)
    nxt = step(nxt[None])
    mx.async_eval(nxt)
    ts, builds = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        nxt2 = step(nxt[None])
        tb = time.perf_counter()
        mx.async_eval(nxt2)
        mx.eval(nxt)
        ts.append(time.perf_counter() - t0)
        builds.append(tb - t0)
        nxt = nxt2
    mx.eval(nxt)
    # synchronous variant (no overlap): build + eval each token
    ts_sync = []
    y = nxt[None]
    for _ in range(n):
        t0 = time.perf_counter()
        y = step(y)
        mx.eval(y)
        ts_sync.append(time.perf_counter() - t0)
        y = y[None]
    return {
        "async_ms": statistics.median(ts) * 1e3,
        "graph_build_ms": statistics.median(builds) * 1e3,
        "sync_ms": statistics.median(ts_sync) * 1e3,
    }


def layer_bytes(cfg, li, experts_bits, attn_bits, ctx):
    attn = qbytes((6656 + 4096) * 4096, attn_bits)
    kv_read = (min(ctx, 511) if li < 2 else ctx) * 8 * 128 * 2 * 2
    if li == 0:
        mlp = qbytes(3 * 16384 * 4096, experts_bits)
    else:
        mlp = (
            258 * 4096 * 2
            + qbytes(2 * 3 * 2048 * 4096, experts_bits)
            + 6 * qbytes(3 * 2048 * 4096, experts_bits)
        )
    return attn + mlp + kv_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-bits", type=int, default=2)
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--head-bits", type=int, default=None)
    ap.add_argument("--contexts", default="0,2048,8192,32768")
    ap.add_argument("--prefill", default="512,2048,8192")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    E, A, H = args.experts_bits, args.attn_bits, args.head_bits
    tag = args.tag or f"e{E}_a{A or 'bf16'}_h{H or 'bf16'}"
    log(f"== layer_bench {tag}  mlx {mx.__version__}")

    t0 = time.time()
    model, cfg = build_model(E, A, H)
    nparams = sum(v.size * v.itemsize for _, v in nn.utils.tree_flatten(model.parameters())) if hasattr(nn, "utils") else None
    log(f"model built in {time.time()-t0:.1f}s; active memory {mx.get_active_memory()/1e9:.2f} GB")
    res = {
        "tag": tag,
        "experts_bits": E,
        "attn_bits": A,
        "head_bits": H,
        "mlx": mx.__version__,
        "device": str(mx.device_info()),
        "decode": {},
        "prefill": {},
        "head": {},
    }

    # ---------------- head: embed + final norm + lm_head ----------------
    gpu_warmup()
    h1 = rand_h(1)
    res["head"]["lm_head_ms"] = chained(lambda x: model.lm_head(x)[..., :4096], h1, chain=4)
    res["head"]["lm_head_bytes"] = qbytes(201024 * 4096, H)

    def head_step(x):
        logits = model._logits_from_norm(model.model.norm(x))
        return mx.argmax(logits[:, -1, :], axis=-1)

    res["head"]["norm_lmhead_argmax_ms"] = med(lambda: head_step(h1), warm=3, reps=9)
    ids = mx.array([[123]])
    res["head"]["embed_ms"] = med(lambda: model.model.embed(ids), warm=3, reps=9)
    log(f"head: lm_head {res['head']['lm_head_ms']:.3f} ms, norm+lm_head+argmax {res['head']['norm_lmhead_argmax_ms']:.3f} ms, embed {res['head']['embed_ms']:.3f} ms")
    # dtype sanity: the residual stream must stay bf16 through all three layers
    _c = make_caches(model, 0)
    _pre = model.model(ids, _c, None, skip_final_norm=True)
    mx.eval(_pre)
    res["residual_dtype_after_3_layers"] = str(_pre.dtype)
    log(f"residual stream dtype after 3 layers: {_pre.dtype}")
    del _c, _pre

    # ---------------- decode ----------------
    for ctx in [int(c) for c in args.contexts.split(",")]:
        log(f"-- decode at context {ctx}")
        d = {"ctx": ctx, "layers": {}}
        gpu_warmup(2.0)
        caches = make_caches(model, ctx)
        d["full3"] = time_full_decode(model, caches)
        log(f"   full 3-layer loop: async {d['full3']['async_ms']:.3f} ms/tok, sync {d['full3']['sync_ms']:.3f}, graph build {d['full3']['graph_build_ms']:.3f}")
        del caches
        for li, layer in enumerate(model.model.layers):
            c = model.make_cache()[li]
            fill_cache(c, ctx)
            lr = {}
            lr["layer_ms"] = chained(lambda x: layer(x, cache=c), h1)
            lr["attn_ms"] = chained(lambda x: layer.self_attn(layer.input_layernorm(x), cache=c), h1)
            lr["mlp_ms"] = chained(lambda x: layer.mlp(layer.post_attention_layernorm(x)), h1)
            lr["sconv_ms"] = chained(lambda x: layer.attn_sconv(x, cache=c[1], residual=x), h1)
            lr["norm_ms"] = chained(lambda x: layer.input_layernorm(x), h1)
            if li == 2:
                sa = layer.self_attn
                lf = sa.log_floor
                sa.log_floor = None
                lr["attn_no_tau_ms"] = chained(lambda x: layer.self_attn(layer.input_layernorm(x), cache=c), h1)
                sa.log_floor = lf
            if li == 1 and ctx > 512:
                L._SLIDING_KV_SLICE = False
                lr["attn_no_kv_slice_ms"] = chained(lambda x: layer.self_attn(layer.input_layernorm(x), cache=c), h1)
                L._SLIDING_KV_SLICE = True
            if li >= 1 and E == 4:
                L._DOWN_COMBINE = False
                lr["mlp_gather_down_ms"] = chained(lambda x: layer.mlp(layer.post_attention_layernorm(x)), h1)
                L._DOWN_COMBINE = True
            if li >= 1:
                moe = layer.mlp
                xf = h1.reshape(1, 4096)
                gw = moe.gate_weight
                lr["router_logits+route_kernel_ms"] = med(lambda: moe._route(xf @ gw.T), warm=3, reps=9)
                logits = xf @ gw.T
                mx.eval(logits)
                lr["route_kernel_only_ms"] = med(lambda: moe._route(logits), warm=3, reps=9)
                idx, w, gamma = moe._route(logits)
                mx.eval(idx, w, gamma)
                lr["shared_experts_ms"] = chained(lambda x: moe.shared_experts(x, gamma), xf)
                lr["routed_experts_ms"] = chained(lambda x: (moe.switch_mlp(x, idx) * w[..., None]).sum(axis=-2), xf)
            lr["bytes"] = layer_bytes(cfg, li, E, A, ctx)
            lr["GB/s"] = lr["bytes"] / (lr["layer_ms"] * 1e-3) / 1e9
            d["layers"][li] = lr
            log(f"   {LAYER_NAME[li]:14s} layer {lr['layer_ms']:.3f} ms (attn {lr['attn_ms']:.3f}, mlp {lr['mlp_ms']:.3f}, sconv {lr['sconv_ms']:.3f}) -> {lr['GB/s']:.0f} GB/s eff; extras {{{', '.join(f'{k}={v:.3f}' for k, v in lr.items() if k.endswith('_ms') and k not in ('layer_ms','attn_ms','mlp_ms','sconv_ms'))}}}")
            del c
            mx.clear_cache()
        # extrapolation to 42 layers
        per_layer = sum(LAYER_MULT[li] * d["layers"][li]["layer_ms"] for li in range(3))
        pred_sum = per_layer + res["head"]["norm_lmhead_argmax_ms"] + res["head"]["embed_ms"]
        pred_loop = d["full3"]["async_ms"] + sum((LAYER_MULT[li] - 1) * d["layers"][li]["layer_ms"] for li in range(3))
        total_bytes = sum(LAYER_MULT[li] * d["layers"][li]["bytes"] for li in range(3)) + res["head"]["lm_head_bytes"]
        d["extrapolated_42"] = {
            "sum_of_parts_ms": pred_sum,
            "loop_based_ms": pred_loop,
            "tok_s_sum_of_parts": 1000 / pred_sum,
            "tok_s_loop_based": 1000 / pred_loop,
            "bytes_per_token": total_bytes,
            "roofline_tok_s_at_480GBs": 480e9 / total_bytes,
        }
        log(f"   => 42-layer prediction: {1000/pred_sum:.1f} tok/s (sum of parts), {1000/pred_loop:.1f} tok/s (loop-based); {total_bytes/1e9:.2f} GB/token, roofline@480GB/s {480e9/total_bytes:.1f} tok/s")
        res["decode"][ctx] = d
        gc.collect()
        mx.clear_cache()

    # ---------------- prefill ----------------
    if not args.skip_prefill:
        for T in [int(t) for t in args.prefill.split(",")]:
            log(f"-- prefill T={T}")
            p = {"T": T, "layers": {}}
            hT = rand_h(T)
            for li, layer in enumerate(model.model.layers):
                reps = 3 if T >= 8192 else 5
                cs = [model.make_cache()[li] for _ in range(reps + 1)]
                it = iter(cs)
                lr = {}
                lr["layer_ms"] = med(lambda: layer(hT, cache=next(it)), warm=1, reps=reps)
                cs = [model.make_cache()[li] for _ in range(reps + 1)]
                it = iter(cs)
                lr["attn_ms"] = med(lambda: layer.self_attn(layer.input_layernorm(hT), cache=next(it)), warm=1, reps=reps)
                lr["mlp_ms"] = med(lambda: layer.mlp(layer.post_attention_layernorm(hT)), warm=1, reps=reps)
                cs = [model.make_cache()[li] for _ in range(reps + 1)]
                it = iter(cs)
                lr["sconv_ms"] = med(lambda: layer.attn_sconv(hT, cache=next(it)[1], residual=hT), warm=1, reps=reps)
                if li == 2:
                    sa = layer.self_attn
                    lf = sa.log_floor
                    sa.log_floor = None
                    cs = [model.make_cache()[li] for _ in range(reps + 1)]
                    it = iter(cs)
                    lr["attn_no_tau_ms"] = med(lambda: layer.self_attn(layer.input_layernorm(hT), cache=next(it)), warm=1, reps=reps)
                    sa.log_floor = lf
                if li >= 1:
                    moe = layer.mlp
                    xf = hT.reshape(-1, 4096)
                    logits = xf @ moe.gate_weight.T
                    mx.eval(logits)
                    lr["route_kernel_only_ms"] = med(lambda: moe._route(logits), warm=2, reps=reps)
                    idx, w, gamma = moe._route(logits)
                    mx.eval(idx, w, gamma)
                    lr["shared_experts_ms"] = med(lambda: moe.shared_experts(xf, gamma), warm=1, reps=reps)
                    lr["routed_experts_ms"] = med(lambda: (moe.switch_mlp(xf, idx) * w[..., None]).sum(axis=-2), warm=1, reps=reps)
                p["layers"][li] = lr
                log(f"   {LAYER_NAME[li]:14s} layer {lr['layer_ms']:.1f} ms (attn {lr['attn_ms']:.1f}, mlp {lr['mlp_ms']:.1f}, sconv {lr['sconv_ms']:.2f}) extras {{{', '.join(f'{k}={v:.2f}' for k, v in lr.items() if k.endswith('_ms') and k not in ('layer_ms','attn_ms','mlp_ms','sconv_ms'))}}}")
                del cs
                mx.clear_cache()
            # chunk of 2048 at an offset (the chunked-prefill steady state) for layers 1 and 2
            if T == 2048:
                for off in (2048, 6144):
                    h2 = rand_h(2048)
                    for li in (1, 2):
                        layer = model.model.layers[li]
                        cs = []
                        for _ in range(4):
                            c = model.make_cache()[li]
                            fill_cache(c, off)
                            cs.append(c)
                        it = iter(cs)
                        ms = med(lambda: layer(h2, cache=next(it)), warm=1, reps=3)
                        p["layers"][li][f"chunk2048_at_offset{off}_ms"] = ms
                        log(f"   {LAYER_NAME[li]:14s} chunk 2048 at offset {off}: {ms:.1f} ms")
                        del cs
                        mx.clear_cache()
            per_layer = sum(LAYER_MULT[li] * p["layers"][li]["layer_ms"] for li in range(3))
            p["extrapolated_42"] = {"ms": per_layer, "tok_s": T / (per_layer * 1e-3)}
            log(f"   => 42-layer prefill prediction at T={T}: {per_layer:.0f} ms -> {T/(per_layer*1e-3):.0f} tok/s")
            res["prefill"][T] = p
            del hT
            gc.collect()
            mx.clear_cache()

    res["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    log(f"peak MLX memory {res['peak_memory_gb']:.2f} GB")
    # contention canary: a 1 GB bf16 streaming op should take ~2.5 ms (393 GB/s) when the GPU is ours
    can = mx.random.normal((512, 1024, 1024)).astype(mx.bfloat16)
    mx.eval(can)
    res["canary_1GB_copy_ms_end"] = med(lambda: can + mx.array(0, dtype=mx.bfloat16), warm=3, reps=9)
    res["canary_lm_head_ms_end"] = chained(lambda x: model.lm_head(x)[..., :4096], h1, chain=4)
    log(f"canary at end: 1GB copy {res['canary_1GB_copy_ms_end']:.2f} ms, lm_head {res['canary_lm_head_ms_end']:.3f} ms (start {res['head']['lm_head_ms']:.3f})")
    path = os.path.join(OUT, f"layer_bench_{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    log("wrote", path)


if __name__ == "__main__":
    main()
