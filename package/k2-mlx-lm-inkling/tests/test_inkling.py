# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models import inkling_mm_model as inkling
from mlx_lm.models.cache import (
    BatchRotatingKVCache,
    RotatingKVCache,
    make_prompt_cache,
)


def tiny_text_config(**overrides):
    config = dict(
        hidden_size=64,
        num_hidden_layers=3,
        vocab_size=64,
        unpadded_vocab_size=60,
        logits_mup_width_multiplier=16.0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        sliding_window_size=4,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding"],
        d_rel=4,
        rel_extent=8,
        log_scaling_n_floor=4,
        log_scaling_alpha=0.1,
        sconv_kernel_size=4,
        mlp_layer_types=["dense", "sparse", "sparse"],
        intermediate_size=32,
        dense_intermediate_size=96,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        route_scale=8.0,
    )
    config.update(overrides)
    return config


def random_model(seed=0, dtype=mx.float32, **overrides):
    """A tiny model whose every parameter is random and non-trivial."""
    mx.random.seed(seed)
    args = inkling.ModelArgs.from_dict(
        {"model_type": "inkling_mm_model", "text_config": tiny_text_config(**overrides)}
    )
    model = inkling.Model(args)
    params = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight") or k.endswith("global_scale"):
            v = 1.0 + 0.3 * mx.random.normal(v.shape)
        elif k.endswith("_sconv.weight"):
            v = 0.3 * mx.random.normal(v.shape)
        elif v.ndim >= 2:
            v = mx.random.normal(v.shape) * (0.3 / v.shape[-1] ** 0.5)
        else:
            v = 0.3 * mx.random.normal(v.shape)
        if ".mlp.gate." not in k:  # the router stays fp32
            v = v.astype(dtype)
        params.append((k, v))
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model, args


def banded_mask_numpy(rel_logits, n_keys, window, extent):
    # rel_logits[..., d] is the bias against the key d positions back
    B, H, L, E = rel_logits.shape
    assert E == extent
    dist = (np.arange(L) + n_keys - L)[:, None] - np.arange(n_keys)[None]
    idx = np.clip(dist, 0, E - 1)
    bias = np.take_along_axis(
        rel_logits, np.broadcast_to(idx, (B, H, L, n_keys)), axis=-1
    )
    bias = np.where(dist >= E, 0.0, bias)
    masked = dist < 0
    if window > 0:
        masked |= dist >= window
    return np.where(masked, -np.inf, bias)


def sconv_numpy(x, w, state):
    # y[t] = x[t] + sum_k w[k] x[t - (K-1) + k]; state = the previous K-1 inputs
    B, L, C = x.shape
    K = w.shape[1]
    xp = np.concatenate([state, x], axis=1)
    y = x.copy()
    for k in range(K):
        y += w[:, k] * xp[:, k : k + L]
    return y, xp[:, L:]


def router_numpy(logits, bias, global_scale, route_scale, top_k):
    n_routed = bias.shape[0]
    routed, shared = logits[:, :n_routed], logits[:, n_routed:]
    scores = 1 / (1 + np.exp(-routed)) + bias
    inds = np.argsort(-scores, axis=-1, kind="stable")[:, :top_k]
    active = np.concatenate([np.take_along_axis(routed, inds, -1), shared], -1)
    lp = -np.logaddexp(0, -active)
    w = np.exp(lp - lp.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True) * route_scale * global_scale
    return inds, w


class TestInkling(unittest.TestCase):

    def test_banded_mask(self):
        # Shapes crossing the sliding window (512) and the bias extent (1024)
        cases = [
            (1, 2, 1, 600, 512, 512),
            (2, 2, 8, 2000, 512, 512),
            (1, 3, 37, 548, 512, 512),
            (1, 2, 600, 1111, 512, 512),
            (1, 2, 1, 1500, 0, 1024),
            (2, 2, 8, 1030, 0, 1024),
            (1, 2, 37, 1100, 0, 1024),
            (1, 2, 1300, 1300, 0, 1024),
        ]
        for B, H, L, S, window, extent in cases:
            rel = mx.random.normal((B, H, L, extent)).astype(mx.float16)
            ref = banded_mask_numpy(np.array(rel.astype(mx.float32)), S, window, extent)
            # The model passes the bias in reversed distance order
            out = inkling.banded_mask(rel[..., ::-1], S, window, extent)
            self.assertEqual(out.shape, (B, H, L, S))
            self.assertEqual(out.dtype, mx.float16)
            out = np.array(out.astype(mx.float32))
            np.testing.assert_array_equal(np.isinf(out), np.isinf(ref))
            finite = np.isfinite(ref)
            np.testing.assert_allclose(out[finite], ref[finite], rtol=0, atol=0)
            # The fallback used without Metal
            if inkling._banded_mask_kernel is not None:
                kernel = inkling._banded_mask_kernel
                inkling._banded_mask_kernel = None
                try:
                    fallback = np.array(
                        inkling.banded_mask(rel[..., ::-1], S, window, extent).astype(
                            mx.float32
                        )
                    )
                finally:
                    inkling._banded_mask_kernel = kernel
                np.testing.assert_array_equal(fallback, out)

    def test_decode_bias_matches_prefill(self):
        # The reversed-slice bias of a single query equals the banded mask row
        model, args = random_model()
        attn = model.layers[1].attn  # global layer, extent 8
        x = mx.random.normal((1, 12, args.text_config.hidden_size))
        cache = make_prompt_cache(model)[1]
        full = attn(x, cache[0], cache[1], None)
        cache = make_prompt_cache(model)[1]
        steps = [attn(x[:, :5], cache[0], cache[1], None)]
        steps += [attn(x[:, t : t + 1], cache[0], cache[1], None) for t in range(5, 12)]
        # fp32 GEMMs may run at reduced (TF32) precision on the GPU
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, 1), full, rtol=1e-2, atol=1e-2)
        )

    def test_short_conv_state(self):
        B, L, C, K = 2, 11, 16, 4
        conv = inkling.ShortConv(C, K, slot=0)
        conv.weight = mx.random.normal((C, K, 1))
        x = mx.random.normal((B, L, C)).astype(mx.bfloat16)
        cache = make_prompt_cache(random_model()[0])[0][1]
        # Prefill of L tokens == L single-token decodes, bit for bit
        full = conv(x, cache)
        state_full = cache[0]
        cache2 = make_prompt_cache(random_model()[0])[0][1]
        steps = [conv(x[:, t : t + 1], cache2) for t in range(L)]
        self.assertTrue(mx.array_equal(mx.concatenate(steps, 1), full))
        self.assertTrue(mx.array_equal(cache2[0], state_full))
        # Against the numpy reference in fp32
        xf = mx.random.normal((B, L, C))
        state = mx.random.normal((B, K - 1, C))
        cache3 = make_prompt_cache(random_model()[0])[0][1]
        cache3[0] = state
        y = conv(xf, cache3)
        y_ref, state_ref = sconv_numpy(
            np.array(xf), np.array(conv.weight)[:, :, 0], np.array(state)
        )
        np.testing.assert_allclose(np.array(y), y_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.array(cache3[0]), state_ref)

    def test_router(self):
        for N, R, S, top_k in [
            (37, 8, 2, 3),
            (1, 256, 2, 6),
            (300, 256, 2, 6),
            (5, 40, 1, 4),
        ]:
            route_scale = 8.0
            logits = mx.random.normal((N, R + S))
            bias = 0.1 * mx.random.normal((R,))
            global_scale = mx.array([1.3])
            ref_inds, ref_w = router_numpy(
                np.array(logits), np.array(bias), 1.3, route_scale, top_k
            )
            for name, fn in [
                ("kernel", inkling.select_experts),
                ("compiled", inkling._select_experts),
            ]:
                inds, w = fn(logits, bias, global_scale, top_k, route_scale)
                inds, w = np.array(inds), np.array(w)
                self.assertEqual(inds.shape, (N, top_k + S), name)
                np.testing.assert_array_equal(
                    inds[:, top_k:], np.broadcast_to(np.arange(R, R + S), (N, S))
                )
                # Same expert sets, same weights per expert
                for n in range(N):
                    self.assertEqual(set(inds[n, :top_k]), set(ref_inds[n]), name)
                    got = dict(zip(inds[n], w[n]))
                    ref = dict(zip(list(ref_inds[n]) + list(range(R, R + S)), ref_w[n]))
                    for e in ref:
                        self.assertAlmostEqual(got[e], ref[e], places=5, msg=name)
        # The kernel breaks exact ties towards the lowest expert index
        if inkling._route_kernel is not None and mx.default_device() == mx.gpu:
            inds, w = inkling.select_experts(
                mx.zeros((2, 10)), mx.zeros((8,)), mx.array([1.0]), 3, 8.0
            )
            self.assertEqual(np.array(inds)[1, :3].tolist(), [0, 1, 2])
            self.assertTrue(mx.allclose(w, mx.full((2, 5), 8.0 / 5)))

    def test_sanitize_layouts(self):
        # The HF source layout (interleaved w13, [C,1,K] convs, separate q/k/v/r,
        # separate shared experts) and the mlx-community layout (split
        # gate/up, [C,K,1] convs) load to the same parameters.
        model, args = random_model()
        cfg = args.text_config
        hf, mlx_layout = {}, {}
        prefix = "model.llm.layers.1."
        H, I, E, S = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.n_routed_experts,
            cfg.n_shared_experts,
        )
        attn = model.layers[1].attn
        dq, dkv = (
            cfg.num_attention_heads * cfg.head_dim,
            cfg.num_key_value_heads * cfg.head_dim,
        )
        fused = attn.qkvr_proj.weight
        for name, a, b in [
            ("wq_du", 0, dq),
            ("wk_dv", dq, dq + dkv),
            ("wv_dv", dq + dkv, dq + 2 * dkv),
            ("wr_du", dq + 2 * dkv, fused.shape[0]),
        ]:
            hf[f"{prefix}attn.{name}.weight"] = fused[a:b]
            mlx_layout[f"{prefix}attn.{name}.weight"] = fused[a:b]
        kv_w = attn.kv_sconv.weight
        hf[f"{prefix}attn.k_sconv.weight"] = kv_w[:dkv].swapaxes(1, 2)
        hf[f"{prefix}attn.v_sconv.weight"] = kv_w[dkv:].swapaxes(1, 2)
        mlx_layout[f"{prefix}attn.k_sconv.weight"] = kv_w[:dkv]
        mlx_layout[f"{prefix}attn.v_sconv.weight"] = kv_w[dkv:]
        gate = mx.random.normal((E + S, I, H))
        up = mx.random.normal((E + S, I, H))
        down = mx.random.normal((E + S, H, I))
        w13 = mx.stack([gate, up], axis=2).reshape(E + S, 2 * I, H)
        hf[f"{prefix}mlp.experts.w13_weight"] = w13[:E]
        hf[f"{prefix}mlp.experts.w2_weight"] = down[:E]
        hf[f"{prefix}mlp.shared_experts.shared_w13_weight"] = w13[E:]
        hf[f"{prefix}mlp.shared_experts.shared_w2_weight"] = down[E:]
        for n, w in [("gate_proj", gate), ("up_proj", up), ("down_proj", down)]:
            mlx_layout[f"{prefix}mlp.experts.{n}.weight"] = w[:E]
            mlx_layout[f"{prefix}mlp.shared_experts.{n}.weight"] = w[E:]
        hf[f"{prefix}mlp.gate.weight"] = mx.random.normal((E + S, H)).astype(
            mx.bfloat16
        )
        mlx_layout[f"{prefix}mlp.gate.weight"] = hf[f"{prefix}mlp.gate.weight"]
        proj = mx.random.normal((cfg.d_rel, cfg.rel_extent))
        hf[f"{prefix}attn.rel_logits_proj.proj"] = proj
        mlx_layout[f"{prefix}attn.rel_logits_proj.proj"] = proj
        hf["model.visual.x"] = mx.zeros(1)
        hf["model.audio.x"] = mx.zeros(1)
        hf["model.mtp.x"] = mx.zeros(1)

        a = model.sanitize(hf)
        b = model.sanitize(mlx_layout)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(mx.array_equal(a[k], b[k]), k)
        self.assertTrue(mx.array_equal(a[f"{prefix}attn.qkvr_proj.weight"], fused))
        self.assertTrue(mx.array_equal(a[f"{prefix}attn.kv_sconv.weight"], kv_w))
        self.assertTrue(
            mx.array_equal(a[f"{prefix}mlp.experts.gate_proj.weight"], gate)
        )
        self.assertTrue(mx.array_equal(a[f"{prefix}mlp.experts.up_proj.weight"], up))
        self.assertTrue(
            mx.array_equal(a[f"{prefix}mlp.experts.down_proj.weight"], down)
        )
        self.assertEqual(a[f"{prefix}mlp.gate.weight"].dtype, mx.float32)
        self.assertTrue(
            mx.array_equal(
                a[f"{prefix}attn.rel_logits_proj.proj_reversed"], proj[:, ::-1]
            )
        )

    def test_sanitize_quantized_fold(self):
        # Appending quantized shared experts to quantized routed experts is exact
        model, args = random_model()
        E, S, I, H = 8, 2, 64, 128
        gate = mx.random.normal((E + S, I, H))
        prefix = "model.llm.layers.1.mlp."
        weights = {}
        for part, sl in [("experts", slice(0, E)), ("shared_experts", slice(E, E + S))]:
            w, s, b = mx.quantize(gate[sl], group_size=64, bits=2)
            weights[f"{prefix}{part}.gate_proj.weight"] = w
            weights[f"{prefix}{part}.gate_proj.scales"] = s
            weights[f"{prefix}{part}.gate_proj.biases"] = b
        out = model.sanitize(weights)
        w, s, b = mx.quantize(gate, group_size=64, bits=2)
        self.assertTrue(mx.array_equal(out[f"{prefix}experts.gate_proj.weight"], w))
        self.assertTrue(mx.array_equal(out[f"{prefix}experts.gate_proj.scales"], s))
        self.assertTrue(mx.array_equal(out[f"{prefix}experts.gate_proj.biases"], b))
        # Mismatched quantization is refused
        bad = dict(weights)
        bad[f"{prefix}shared_experts.gate_proj.scales"] = weights[
            f"{prefix}shared_experts.gate_proj.scales"
        ][..., :1]
        with self.assertRaises(ValueError):
            model.sanitize(bad)

    def test_key_distances(self):
        b, h, d, W = 1, 2, 8, 6
        for cls in (RotatingKVCache, BatchRotatingKVCache):
            cache = cls(W) if cls is RotatingKVCache else cls(W, [0])
            xs = mx.random.normal((b, h, 23, d))
            pos = 0
            for n in [3, 1, 1, 4, 1, 1, 1, 7, 1, 1, 1, 1]:
                k, _ = cache.update_and_fetch(
                    xs[..., pos : pos + n, :], xs[..., pos : pos + n, :]
                )
                pos += n
                dist = cache.key_distances()
                self.assertEqual(dist.shape[0], k.shape[2])
                # slot s holds the key dist[s] positions behind the newest one
                expected = xs[..., pos - 1 - dist, :]
                self.assertTrue(mx.array_equal(k, expected), (pos, n))

    def test_batched_prompts(self):
        # A right-padded batch through merged caches equals the single-sequence runs
        model, args = random_model()
        toks_a = mx.random.randint(0, 60, (1, 7))
        toks_b = mx.random.randint(0, 60, (1, 3))
        ref = []
        for toks in (toks_a, toks_b):
            cache = make_prompt_cache(model)
            lp = model(toks, cache=cache)[:, -1]
            nxt = mx.argmax(lp, -1, keepdims=True)
            ref.append((lp, model(nxt, cache=cache)[:, -1], nxt))
        caches = [make_prompt_cache(model) for _ in range(2)]
        merged = [
            c[0].merge([c[i] for c in caches]) for i, c in enumerate(zip(*caches))
        ]
        for c in merged:
            c.prepare(lengths=[7, 3], right_padding=[0, 4])
        padded = mx.concatenate([toks_a, mx.pad(toks_b, [(0, 0), (0, 4)])], 0)
        lp = model(padded, cache=merged)
        for c in merged:
            c.finalize()
        # fp32 GEMMs may run at reduced (TF32) precision on the GPU
        tol = dict(rtol=1e-2, atol=1e-2)
        self.assertTrue(mx.allclose(lp[0, -1], ref[0][0][0], **tol))
        self.assertTrue(mx.allclose(lp[1, 2], ref[1][0][0], **tol))
        nxt = mx.concatenate([ref[0][2], ref[1][2]], 0)
        lp = model(nxt, cache=merged)[:, -1]
        self.assertTrue(mx.allclose(lp[0], ref[0][1][0], **tol))
        self.assertTrue(mx.allclose(lp[1], ref[1][1][0], **tol))

    def test_prefill_decode_consistency(self):
        model, args = random_model(dtype=mx.float16)
        ids = mx.random.randint(0, 60, (2, 12))
        full = model(ids)
        cache = make_prompt_cache(model)
        steps = [model(ids[:, :7], cache=cache)]
        steps += [model(ids[:, t : t + 1], cache=cache) for t in range(7, 12)]
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, 1), full, rtol=2e-2, atol=2e-2)
        )

    def test_cpu_exactness(self):
        # On the CPU device fp32 matmuls are exact (the GPU may use TF32), so the
        # cached decode and the chunked prefill match the full forward at fp32
        # precision; the pure-MLX paths of the kernels are exercised here
        model, args = random_model()
        ids = mx.random.randint(0, 60, (2, 12))
        default = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            full = model(ids)
            cache = make_prompt_cache(model)
            steps = [model(ids[:, :7], cache=cache)]
            steps += [model(ids[:, t : t + 1], cache=cache) for t in range(7, 12)]
            decode = mx.concatenate(steps, 1)
            cache = make_prompt_cache(model)
            chunks = [(0, 5), (5, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
            chunked = mx.concatenate(
                [model(ids[:, a:b], cache=cache) for a, b in chunks], 1
            )
            mx.eval(full, decode, chunked)
        finally:
            mx.set_default_device(default)
        self.assertTrue(mx.allclose(decode, full, rtol=1e-5, atol=1e-6))
        self.assertTrue(mx.allclose(chunked, full, rtol=1e-5, atol=1e-6))

    def test_kv_bits_warns(self):
        import warnings

        from mlx_lm.generate import generate_step

        model, args = random_model(dtype=mx.float16)
        prompt = mx.random.randint(0, 60, (5,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in generate_step(prompt, model, max_tokens=2, kv_bits=4):
                pass
        self.assertTrue(any("kv_bits" in str(w.message) for w in caught))

    def test_flat_text_config(self):
        args = inkling.ModelArgs.from_dict(
            {"model_type": "inkling_text", **tiny_text_config()}
        )
        self.assertEqual(args.text_config.hidden_size, 64)
        model = inkling.Model(args)
        self.assertEqual(len(model.layers), 3)
        with self.assertRaises(ValueError):
            inkling.ModelArgs.from_dict(
                {"model_type": "inkling_text", **tiny_text_config(use_sconv=False)}
            )


if __name__ == "__main__":
    unittest.main()
