"""mlx-lm Inkling (fp32, GPU, TF32 off) vs the numpy float64 oracle on the R5 tiny
random model. Weights are produced in the HF *source* layout (interleaved w13,
[C,1,K] convs, separate q/k/v/r) and go through Model.sanitize, so this also
covers the HF-layout conversion path. Run: MLX_ENABLE_TF32=0 python test_vs_oracle.py
"""
import os, sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "..", "research", "oracle")
sys.path.insert(0, ORACLE)
import inkling_ref as R  # noqa: E402
from mlx_lm.models import inkling_mm_model as M  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

blob = np.load(os.path.join(ORACLE, "tiny_params.npy"), allow_pickle=True).item()
params, ocfg, ids, TINY = blob["params"], blob["cfg"], blob["ids"], blob["tiny"]


def err(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(d.max()), float(d.max() / (np.abs(b).max() + 1e-300))


def record(name, a, b, tol=1e-4):
    ma, mr = err(a, b)
    flag = "" if mr <= tol else "   <-- FAIL"
    print(f"  {name:60s} max_abs={ma:.3e}  max_rel={mr:.3e}{flag}")
    return mr <= tol


def interleave(g, u):
    # transformers gate/up [.., I, H] -> checkpoint w13 [.., 2I, H] rows [g0, u0, g1, u1, ...]
    return np.stack([g, u], axis=-2).reshape(*g.shape[:-2], 2 * g.shape[-2], g.shape[-1])


def f(a):
    return mx.array(np.asarray(a, np.float32))


def hf_layout(p, cfg):
    w = {}
    w["model.llm.embed.weight"] = f(p["model.embed_tokens.weight"])
    w["model.llm.embed_norm.weight"] = f(p["model.embed_tokens.embed_norm.weight"])
    w["model.llm.norm.weight"] = f(p["model.norm.weight"])
    w["model.llm.unembed.weight"] = f(p["lm_head.weight"])
    w["model.visual.dummy"] = f(np.zeros(3)); w["model.audio.dummy"] = f(np.zeros(3)); w["model.mtp.dummy"] = f(np.zeros(3))
    for i, lt in enumerate(cfg["layer_types"]):
        pin, pout = f"model.layers.{i}.", f"model.llm.layers.{i}."
        a_in, a_out = pin + "self_attn.", pout + "attn."
        for src, dst in [("q_proj", "wq_du"), ("k_proj", "wk_dv"), ("v_proj", "wv_dv"), ("r_proj", "wr_du"), ("o_proj", "wo_ud")]:
            w[a_out + dst + ".weight"] = f(p[a_in + src + ".weight"])
        for n in ("k_sconv", "v_sconv"):
            w[a_out + n + ".weight"] = f(p[a_in + n + ".conv1d.weight"])  # [C,1,K]
        w[a_out + "q_norm.weight"] = f(p[a_in + "q_norm.weight"])
        w[a_out + "k_norm.weight"] = f(p[a_in + "k_norm.weight"])
        w[a_out + "rel_logits_proj.proj"] = f(p[a_in + "rel_logits_proj.proj"])
        w[pout + "attn_norm.weight"] = f(p[pin + "input_layernorm.weight"])
        w[pout + "mlp_norm.weight"] = f(p[pin + "post_attention_layernorm.weight"])
        w[pout + "attn_sconv.weight"] = f(p[pin + "attn_sconv.conv1d.weight"])
        w[pout + "mlp_sconv.weight"] = f(p[pin + "mlp_sconv.conv1d.weight"])
        m_in, m_out = pin + "mlp.", pout + "mlp."
        if cfg["mlp_layer_types"][i] == "sparse":
            w[m_out + "gate.weight"] = f(p[m_in + "gate.weight"])
            w[m_out + "gate.bias"] = f(p[m_in + "gate.e_score_correction_bias"])
            w[m_out + "gate.global_scale"] = f(p[m_in + "gate.global_scale"])
            gu = p[m_in + "experts.gate_up_proj"]; I = gu.shape[1] // 2
            w[m_out + "experts.w13_weight"] = f(interleave(gu[:, :I], gu[:, I:]))
            w[m_out + "experts.w2_weight"] = f(p[m_in + "experts.down_proj"])
            w[m_out + "shared_experts.shared_w13_weight"] = f(interleave(p[m_in + "shared_experts.gate_proj"], p[m_in + "shared_experts.up_proj"]))
            w[m_out + "shared_experts.shared_w2_weight"] = f(p[m_in + "shared_experts.down_proj"])
        else:
            w[m_out + "w13_dn.weight"] = f(interleave(p[m_in + "gate_proj.weight"], p[m_in + "up_proj.weight"]))
            w[m_out + "w2_md.weight"] = f(p[m_in + "down_proj.weight"])
            w[m_out + "global_scale"] = f(p[m_in + "global_scale"])
    return w


def text_config():
    return dict(
        hidden_size=TINY["hidden_size"], num_hidden_layers=TINY["num_hidden_layers"], vocab_size=TINY["vocab_size"],
        unpadded_vocab_size=TINY["unpadded_vocab_size"], rms_norm_eps=TINY["rms_norm_eps"], use_embed_norm=True,
        logits_mup_width_multiplier=TINY["logits_mup_width_multiplier"], num_attention_heads=TINY["num_attention_heads"],
        num_key_value_heads=TINY["num_key_value_heads"], head_dim=TINY["head_dim"],
        swa_num_attention_heads=TINY["swa_num_attention_heads"], swa_num_key_value_heads=TINY["swa_num_key_value_heads"],
        swa_head_dim=TINY["swa_head_dim"], sliding_window_size=TINY["sliding_window_size"], layer_types=TINY["layer_types"],
        d_rel=TINY["d_rel"], rel_extent=TINY["rel_extent"], log_scaling_n_floor=TINY["log_scaling_n_floor"],
        log_scaling_alpha=TINY["log_scaling_alpha"], sconv_kernel_size=TINY["conv_kernel_size"],
        mlp_layer_types=TINY["mlp_layer_types"], intermediate_size=TINY["moe_intermediate_size"],
        dense_intermediate_size=TINY["intermediate_size"], n_routed_experts=TINY["n_routed_experts"],
        num_experts_per_tok=TINY["num_experts_per_tok"], n_shared_experts=TINY["n_shared_experts"], route_scale=TINY["route_scale"],
    )


def main():
    ok = True
    ref = R.model_forward_ref(params, ocfg, ids)
    B, L = ids.shape
    for dev_name, device in [("gpu", mx.gpu), ("cpu", mx.cpu)]:
        mx.set_default_device(device)
        print(f"\n[{dev_name}] mlx-lm inkling_mm_model fp32 vs oracle float64 (B={B}, L={L})")
        model = M.Model(M.ModelArgs(model_type="inkling_mm_model", text_config=text_config()))
        weights = model.sanitize(hf_layout(params, ocfg))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        x = mx.array(ids)
        ok &= record(f"{dev_name}: full forward logits", model(x), ref["logits"])
        h = model.model.llm.embed_tokens(x)
        ok &= record(f"{dev_name}: embedding + embed_norm", h, R.embed_ref(params, ocfg, ids))
        for i, layer in enumerate(model.model.llm.layers):
            h = layer(h, None)
            ok &= record(f"{dev_name}: layer {i} output ({TINY['layer_types'][i]}, {TINY['mlp_layer_types'][i]})", h, ref["layer_outputs"][i])
        cache = make_prompt_cache(model)
        steps = [model(x[:, :7], cache=cache)] + [model(x[:, t:t + 1], cache=cache) for t in range(7, L)]
        ok &= record(f"{dev_name}: cached decode (7 + 5x1) logits", mx.concatenate(steps, 1), ref["logits"])
        cache = make_prompt_cache(model)
        steps = [model(x[:, a:b], cache=cache) for a, b in [(0, 5), (5, 8), (8, 9), (9, 10), (10, 11), (11, 12)]]
        ok &= record(f"{dev_name}: chunked prefill (5,3,1,1,1,1) logits", mx.concatenate(steps, 1), ref["logits"])
        # batch of 2 with different content agrees with per-sequence runs
        o2 = model(x); o_a = model(x[:1]); o_b = model(x[1:])
        ok &= record(f"{dev_name}: batch rows == single rows", mx.concatenate([o_a, o_b], 0), o2, tol=1e-5)
    print("\nOK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
