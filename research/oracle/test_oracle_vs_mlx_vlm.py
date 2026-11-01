"""Compare mlx-vlm's Inkling LanguageModel (and its MTP drafter) against the numpy float64
oracle on the same tiny random weights used in test_oracle_vs_transformers.py.

mlx runs in float32 (GPU Metal path and CPU path) so logic differences are separable from
bf16 rounding: a logic match shows up as ~1e-5 relative error, a logic difference as O(1).

Run after test_oracle_vs_transformers.py (it writes tiny_params.npy):
  source ~/.venvs/inkling/bin/activate && python test_oracle_vs_mlx_vlm.py
"""

from __future__ import annotations

import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_ref as R  # noqa: E402

from mlx_vlm.models.inkling.config import TextConfig  # noqa: E402
from mlx_vlm.models.inkling.language import LanguageModel  # noqa: E402
from mlx_vlm.speculative.drafters.inkling_mtp.config import InklingMTPConfig  # noqa: E402
from mlx_vlm.speculative.drafters.inkling_mtp.inkling_mtp import InklingMTPDraftModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
blob = np.load(os.path.join(HERE, "tiny_params.npy"), allow_pickle=True).item()
params, mtp_params, ocfg, ids, TINY = blob["params"], blob["mtp_params"], blob["cfg"], blob["ids"], blob["tiny"]

results = {}


def err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(d.max()), float(d.max() / (np.abs(b).max() + 1e-300))


def record(name, a, b):
    mx_abs, mx_rel = err(a, b)
    results[name] = {"max_abs": mx_abs, "max_rel": mx_rel}
    print(f"  {name:70s} max_abs={mx_abs:.3e}  max_rel={mx_rel:.3e}")
    return mx_rel


def text_config() -> TextConfig:
    return TextConfig(
        hidden_size=TINY["hidden_size"], num_hidden_layers=TINY["num_hidden_layers"], vocab_size=TINY["vocab_size"],
        unpadded_vocab_size=TINY["unpadded_vocab_size"], rms_norm_eps=TINY["rms_norm_eps"], tie_word_embeddings=False,
        use_embed_norm=True, logits_mup_width_multiplier=TINY["logits_mup_width_multiplier"],
        num_attention_heads=TINY["num_attention_heads"], num_key_value_heads=TINY["num_key_value_heads"],
        head_dim=TINY["head_dim"], swa_num_attention_heads=TINY["swa_num_attention_heads"],
        swa_num_key_value_heads=TINY["swa_num_key_value_heads"], swa_head_dim=TINY["swa_head_dim"],
        sliding_window_size=TINY["sliding_window_size"], layer_types=TINY["layer_types"], d_rel=TINY["d_rel"],
        rel_extent=TINY["rel_extent"], log_scaling_n_floor=TINY["log_scaling_n_floor"],
        log_scaling_alpha=TINY["log_scaling_alpha"], sconv_kernel_size=TINY["conv_kernel_size"],
        mlp_layer_types=TINY["mlp_layer_types"], intermediate_size=TINY["moe_intermediate_size"],
        dense_intermediate_size=TINY["intermediate_size"], n_routed_experts=TINY["n_routed_experts"],
        num_experts_per_tok=TINY["num_experts_per_tok"], n_shared_experts=TINY["n_shared_experts"],
        route_scale=TINY["route_scale"],
    )


def conv_w(w):  # torch [C, 1, K] -> MLX Conv1d [C, K, 1]
    return mx.array(np.ascontiguousarray(np.transpose(w, (0, 2, 1))), dtype=mx.float32)


def f32(a):
    return mx.array(np.asarray(a, np.float32))


def mlx_weights_from_transformers(p: dict, cfg: dict, prefix_in="model.", prefix_out="model.") -> dict:
    """Map transformers-layout params onto mlx-vlm LanguageModel parameter names."""
    w = {}
    w[prefix_out + "embed_tokens.weight"] = f32(p[prefix_in + "embed_tokens.weight"])
    w[prefix_out + "embed_norm.weight"] = f32(p[prefix_in + "embed_tokens.embed_norm.weight"])
    w[prefix_out + "norm.weight"] = f32(p[prefix_in + "norm.weight"])
    w["lm_head.weight"] = f32(p["lm_head.weight"])
    for i, lt in enumerate(cfg["layer_types"]):
        w.update(mlx_layer_weights(p, f"{prefix_in}layers.{i}.", f"{prefix_out}layers.{i}.", cfg["mlp_layer_types"][i] == "sparse"))
    return w


def mlx_layer_weights(p: dict, pin: str, pout: str, sparse: bool) -> dict:
    w = {}
    a_in, a_out = pin + "self_attn.", pout + "self_attn."
    w[a_out + "qkvr_proj.weight"] = f32(np.concatenate([p[a_in + n + "_proj.weight"] for n in "qkvr"], axis=0))
    w[a_out + "o_proj.weight"] = f32(p[a_in + "o_proj.weight"])
    w[a_out + "k_sconv.conv.weight"] = conv_w(p[a_in + "k_sconv.conv1d.weight"])
    w[a_out + "v_sconv.conv.weight"] = conv_w(p[a_in + "v_sconv.conv1d.weight"])
    w[a_out + "q_norm.weight"] = f32(p[a_in + "q_norm.weight"])
    w[a_out + "k_norm.weight"] = f32(p[a_in + "k_norm.weight"])
    w[a_out + "rel_proj"] = f32(p[a_in + "rel_logits_proj.proj"])
    w[pout + "input_layernorm.weight"] = f32(p[pin + "input_layernorm.weight"])
    w[pout + "post_attention_layernorm.weight"] = f32(p[pin + "post_attention_layernorm.weight"])
    w[pout + "attn_sconv.conv.weight"] = conv_w(p[pin + "attn_sconv.conv1d.weight"])
    w[pout + "mlp_sconv.conv.weight"] = conv_w(p[pin + "mlp_sconv.conv1d.weight"])
    m_in, m_out = pin + "mlp.", pout + "mlp."
    if sparse:
        w[m_out + "gate_weight"] = f32(p[m_in + "gate.weight"])
        w[m_out + "e_score_correction_bias"] = f32(p[m_in + "gate.e_score_correction_bias"])
        w[m_out + "global_scale"] = f32(p[m_in + "gate.global_scale"])
        gu = p[m_in + "experts.gate_up_proj"]  # [E, 2I, H] transformers layout (gate rows first)
        I = gu.shape[1] // 2
        w[m_out + "switch_mlp.gate_proj.weight"] = f32(gu[:, :I])
        w[m_out + "switch_mlp.up_proj.weight"] = f32(gu[:, I:])
        w[m_out + "switch_mlp.down_proj.weight"] = f32(p[m_in + "experts.down_proj"])
        E = gu.shape[0]
        w[m_out + "switch_mlp.gate_scale"] = mx.ones((E,))
        w[m_out + "switch_mlp.out_scale"] = mx.ones((E,))
        # shared experts -> dense concatenated layout (expert-major on the intermediate axis)
        sg, su, sd = p[m_in + "shared_experts.gate_proj"], p[m_in + "shared_experts.up_proj"], p[m_in + "shared_experts.down_proj"]
        w[m_out + "shared_experts.gate_proj.weight"] = f32(sg.reshape(-1, sg.shape[-1]))
        w[m_out + "shared_experts.up_proj.weight"] = f32(su.reshape(-1, su.shape[-1]))
        w[m_out + "shared_experts.down_proj.weight"] = f32(np.transpose(sd, (1, 0, 2)).reshape(sd.shape[1], -1))
    else:
        w[m_out + "gate_proj.weight"] = f32(p[m_in + "gate_proj.weight"])
        w[m_out + "up_proj.weight"] = f32(p[m_in + "up_proj.weight"])
        w[m_out + "down_proj.weight"] = f32(p[m_in + "down_proj.weight"])
        w[m_out + "global_scale"] = f32(p[m_in + "global_scale"])
    return w


def run_lm(device):
    mx.set_default_device(device)
    tc = text_config()
    lm = LanguageModel(tc)
    weights = mlx_weights_from_transformers(params, ocfg)
    lm.load_weights(list(weights.items()), strict=True)
    mx.eval(lm.parameters())
    return lm, tc


def main():
    ref = R.model_forward_ref(params, ocfg, ids)
    B, L = ids.shape
    for dev_name, device in [("gpu", mx.gpu), ("cpu", mx.cpu)]:
        print(f"\n[{dev_name}] mlx-vlm LanguageModel float32 vs oracle float64 (B={B}, L={L})")
        lm, tc = run_lm(device)
        out = lm(mx.array(ids), cache=None)
        logits = np.array(out.logits.astype(mx.float32))
        record(f"{dev_name}: full forward logits", logits, ref["logits"])
        # layer-by-layer residual stream
        h = lm.model.embed(mx.array(ids))
        record(f"{dev_name}: embedding + embed_norm", np.array(h), R.embed_ref(params, ocfg, ids))
        for i, layer in enumerate(lm.model.layers):
            h = layer(h, cache=None)
            record(f"{dev_name}: layer {i} output ({tc.layer_types[i]}, {TINY['mlp_layer_types'][i]})", np.array(h), ref["layer_outputs"][i])
        # cached decode: prefill 7 then 5 single tokens (exercises the Metal sconv / banded-mask / route kernels)
        cache = lm.make_cache()
        steps = [np.array(lm(mx.array(ids[:, :7]), cache=cache).logits)]
        for t in range(7, L):
            steps.append(np.array(lm(mx.array(ids[:, t : t + 1]), cache=cache).logits))
        record(f"{dev_name}: cached decode (7 + 5x1) logits", np.concatenate(steps, 1), ref["logits"])
        cache = lm.make_cache()
        steps = []
        for a, b in [(0, 5), (5, 8), (8, 9), (9, 10), (10, 11), (11, 12)]:
            steps.append(np.array(lm(mx.array(ids[:, a:b]), cache=cache).logits))
        record(f"{dev_name}: chunked prefill (5,3,1,1,1,1) logits", np.concatenate(steps, 1), ref["logits"])

    # ---------------- MTP drafter ----------------
    next_ids = np.concatenate([ids[:, 1:], ids[:, :1]], axis=1)
    pre_norm = np.asarray(ref["hidden"])    # target hidden BEFORE final norm (what mlx-vlm feeds)
    post_norm = np.asarray(ref["normed"])   # target hidden AFTER final norm (what vLLM/SGLang/transformers feed)
    allp = {**params, **mtp_params}
    lo_mlxsem, h_mlxsem = R.mtp_step_ref(allp, ocfg, pre_norm, next_ids, np.arange(L), R.new_layer_cache(), 0, variant="mlx")
    lo_vllm, h_vllm = R.mtp_step_ref(allp, ocfg, post_norm, next_ids, np.arange(L), R.new_layer_cache(), 0, variant="vllm")
    lo_tf, h_tf = R.mtp_step_ref(allp, ocfg, post_norm, next_ids, np.arange(L), R.new_layer_cache(), 0, variant="transformers")
    for dev_name, device in [("cpu", mx.cpu), ("gpu", mx.gpu)]:
        print(f"\n[{dev_name}] mlx-vlm InklingMTPDraftModel vs mtp_step_ref variants")
        mx.set_default_device(device)
        lm, tc = run_lm(device)
        mcfg = InklingMTPConfig(text_config=tc, num_mtp_layers=TINY["num_mtp_layers"], mtp_local_layer_ids=TINY["mtp_local_layer_ids"])
        drafter = InklingMTPDraftModel(mcfg)
        dw = mlx_layer_weights(allp, "mtp.layers.0.transformer_block.", "blocks.0.transformer_block.", sparse=False)
        dw["blocks.0.hidden_norm.weight"] = f32(mtp_params["mtp.layers.0.hidden_norm.weight"])
        dw["blocks.0.embed_norm.weight"] = f32(mtp_params["mtp.layers.0.embed_norm.weight"])
        dw["blocks.0.input_proj.weight"] = f32(mtp_params["mtp.layers.0.input_proj.weight"])
        dw["norm.weight"] = f32(params["model.norm.weight"])  # split.py copies model.llm.norm.weight here
        drafter.load_weights(list(dw.items()), strict=True)
        drafter.bind(lm)
        drafter._cache = drafter.make_cache()
        mx.eval(drafter.parameters())
        h_mlx = drafter._forward_seq(mx.array(next_ids), mx.array(pre_norm.astype(np.float32)), 0, mx.int32)
        logits_mlx = np.array(drafter._block_logits(h_mlx))
        record(f"{dev_name}: mlx drafter chain hidden vs oracle variant=mlx (pre-norm hidden, raw embed)", np.array(h_mlx), h_mlxsem)
        record(f"{dev_name}: mlx drafter logits vs oracle variant=mlx (final_norm before head)", logits_mlx, lo_mlxsem)
        record(f"{dev_name}: mlx drafter logits vs oracle variant=transformers (post-norm hidden, raw embed, no head norm)", logits_mlx, lo_tf)
        record(f"{dev_name}: mlx drafter logits vs oracle variant=vllm (post-norm hidden, backbone embed_norm, no head norm)", logits_mlx, lo_vllm)
        am = np.argmax(logits_mlx, -1)
        print(f"  greedy-token agreement of the mlx drafter with: mlx-semantics {np.mean(am == np.argmax(lo_mlxsem, -1)):.2f} | "
              f"transformers-semantics {np.mean(am == np.argmax(lo_tf, -1)):.2f} | vllm-semantics {np.mean(am == np.argmax(lo_vllm, -1)):.2f}")
    print("  (vllm vs transformers semantics greedy agreement on the same inputs: "
          f"{np.mean(np.argmax(lo_vllm, -1) == np.argmax(lo_tf, -1)):.2f})")

    json.dump(results, open(os.path.join(HERE, "results_mlx_vlm.json"), "w"), indent=1)
    print("\nDONE")


if __name__ == "__main__":
    main()
