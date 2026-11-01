"""Compare the numpy float64 oracle against transformers' InklingForCausalLM (the reference)
on a tiny random-weight config, in float64 so that any disagreement is a logic difference,
not rounding.

Run:  source ~/.venvs/inkling/bin/activate && python test_oracle_vs_transformers.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_ref as R  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from transformers import InklingForCausalLM, InklingTextConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingDecoderLayer,
    InklingRMSNorm,
)

# ---------------------------------------------------------------------------
# Tiny config: hidden 64, 2 q heads / 1 kv head (GQA), head_dim 32, d_rel 4, rel_extent 8,
# window 4, 8 routed experts top-2 + 2 shared, 3 layers (local, global, local), layer 0 dense,
# log scaling active from position 3 (n_floor 4), 1 MTP layer (local).
# ---------------------------------------------------------------------------
TINY = dict(
    vocab_size=64,
    unpadded_vocab_size=60,
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    swa_num_attention_heads=2,
    swa_num_key_value_heads=1,
    swa_head_dim=32,
    sliding_window_size=4,
    d_rel=4,
    rel_extent=8,
    log_scaling_n_floor=4,
    log_scaling_alpha=0.1,
    layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding"],
    mlp_layer_types=["dense", "sparse", "sparse"],
    rms_norm_eps=1e-6,
    conv_kernel_size=4,
    intermediate_size=96,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=2,
    shared_expert_sink=True,
    route_scale=8.0,
    logits_mup_width_multiplier=16.0,
    num_mtp_layers=1,
    mtp_local_layer_ids=[0],
    chain_hidden_post_norm=False,
    pad_token_id=None,
)


def oracle_cfg(c: InklingTextConfig) -> dict:
    return dict(
        layer_types=list(c.layer_types),
        mlp_layer_types=list(c.mlp_layer_types),
        mtp_layer_types=list(c.mtp_layer_types or []),
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        swa_num_attention_heads=c.swa_num_attention_heads,
        swa_num_key_value_heads=c.swa_num_key_value_heads,
        swa_head_dim=c.swa_head_dim,
        d_rel=c.d_rel,
        rel_extent=c.rel_extent,
        sliding_window_size=c.sliding_window_size,
        log_scaling_n_floor=c.log_scaling_n_floor,
        log_scaling_alpha=c.log_scaling_alpha,
        rms_norm_eps=c.rms_norm_eps,
        num_experts_per_tok=c.num_experts_per_tok,
        n_shared_experts=c.n_shared_experts,
        route_scale=c.route_scale,
        logits_mup_width_multiplier=c.logits_mup_width_multiplier,
        unpadded_vocab_size=c.unpadded_vocab_size,
        mtp_hidden_states_first=c.mtp_hidden_states_first,
        chain_hidden_post_norm=c.chain_hidden_post_norm,
    )


def random_fill(state_dict: dict, seed: int = 1234) -> dict:
    """Deterministic non-trivial random values for every parameter (so norm weights, scales and
    the correction bias are not identity/zero)."""
    rng = np.random.RandomState(seed)
    out = {}
    for name in sorted(state_dict):
        shape = tuple(state_dict[name].shape)
        if name.endswith("global_scale"):
            v = rng.uniform(0.5, 1.5, size=shape)
        elif name.endswith("e_score_correction_bias"):
            v = rng.normal(0.0, 0.15, size=shape)
        elif "norm" in name and len(shape) == 1:
            v = rng.uniform(0.5, 1.5, size=shape)
        elif "conv1d.weight" in name:
            v = rng.normal(0.0, 0.35, size=shape)
        elif name.endswith("rel_logits_proj.proj"):
            v = rng.normal(0.0, 0.6, size=shape)
        elif name.endswith("gate.weight"):
            v = rng.normal(0.0, 0.25, size=shape)
        elif "embed_tokens.weight" in name or name == "lm_head.weight":
            v = rng.normal(0.0, 1.0, size=shape)
        else:
            fan_in = shape[-1]
            v = rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=shape)
        out[name] = v.astype(np.float64)
    return out


def err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(d.max()), float(d.max() / (np.abs(b).max() + 1e-300))


results = {}


def record(name, a, b):
    mx_abs, mx_rel = err(a, b)
    results[name] = {"max_abs": mx_abs, "max_rel": mx_rel}
    print(f"  {name:58s} max_abs={mx_abs:.3e}  max_rel={mx_rel:.3e}")
    return mx_abs


def main():
    cfg = InklingTextConfig(**TINY)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"  # grouped_mm rejects float64
    model = InklingForCausalLM(cfg).double().eval()
    params = random_fill(model.state_dict())
    model.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
    ocfg = oracle_cfg(cfg)
    print("config:", json.dumps({k: (v if not isinstance(v, list) else v) for k, v in ocfg.items()}))

    B, L = 2, 12
    rng = np.random.RandomState(7)
    ids = rng.randint(0, cfg.vocab_size, size=(B, L))

    # ---------------- full-sequence forward ----------------
    with torch.no_grad():
        out = model(torch.from_numpy(ids), output_hidden_states=True, use_cache=False)
    ref = R.model_forward_ref(params, ocfg, ids)
    print("\n[1] full-sequence forward (transformers float64 vs oracle float64), B=%d L=%d" % (B, L))
    # HF convention: hidden_states = (embeddings, layer_0 out, ..., layer_{n-2} out, post-final-norm output)
    hs = [h.numpy() for h in out.hidden_states]
    assert len(hs) == cfg.num_hidden_layers + 1
    record("embedding + embed_norm", hs[0], R.embed_ref(params, ocfg, ids))
    for i in range(cfg.num_hidden_layers - 1):
        record(f"layer {i} output ({cfg.layer_types[i]}, {cfg.mlp_layer_types[i]})", hs[i + 1], ref["layer_outputs"][i])
    record("final norm output (last_hidden_state)", hs[-1], ref["normed"])
    record("logits (sliced to unpadded vocab)", out.logits.numpy(), ref["logits"])
    assert out.logits.shape[-1] == cfg.unpadded_vocab_size

    # ---------------- per-op checks against the transformers modules ----------------
    print("\n[2] per-op checks (isolated module vs oracle)")
    layer0 = model.model.layers[0]
    x = torch.from_numpy(rng.normal(size=(B, 7, cfg.hidden_size)))
    with torch.no_grad():
        y_t = layer0.attn_sconv(x).numpy()
    y_o, _ = R.sconv_ref(x.numpy(), params["model.layers.0.attn_sconv.conv1d.weight"])
    record("sconv_ref (prefill, no state)", y_t, y_o)
    # decode-with-state: run transformers on 7 tokens vs oracle 4 + 3 with carried state
    y_a, st = R.sconv_ref(x.numpy()[:, :4], params["model.layers.0.attn_sconv.conv1d.weight"])
    y_b, _ = R.sconv_ref(x.numpy()[:, 4:], params["model.layers.0.attn_sconv.conv1d.weight"], st)
    record("sconv_ref (chunked with carried state) vs prefill", np.concatenate([y_a, y_b], 1), y_t)

    attn1 = model.model.layers[1].self_attn  # global layer, rel_extent 8, log scaling
    r = torch.from_numpy(rng.normal(size=(B, 5, attn1.num_heads, cfg.d_rel)))
    qp = torch.arange(5) + 3
    kp = torch.arange(8)
    with torch.no_grad():
        bias_t = attn1.rel_logits_proj(r, qp, kp).numpy()
    bias_o = R.rel_bias_ref(r.numpy(), params["model.layers.1.self_attn.rel_logits_proj.proj"], qp.numpy(), kp.numpy(), 8)
    record("rel_bias_ref (global layer, q offset 3, 8 keys)", bias_t, bias_o)

    gate1 = model.model.layers[1].mlp.gate
    xx = torch.from_numpy(rng.normal(size=(9, cfg.hidden_size)))
    with torch.no_grad():
        _, w_t, idx_t, g_t = gate1(xx)
    idx_o, w_o, g_o, _ = R.router_ref(xx.numpy(), params["model.layers.1.mlp.gate.weight"],
                                      params["model.layers.1.mlp.gate.e_score_correction_bias"],
                                      params["model.layers.1.mlp.gate.global_scale"], cfg.route_scale,
                                      cfg.num_experts_per_tok, cfg.n_shared_experts)
    # topk order is unspecified in transformers (sorted=False): compare as index-sorted pairs
    ot = np.argsort(idx_t.numpy(), axis=-1)
    oo = np.argsort(idx_o, axis=-1)
    same_sets = np.array_equal(np.take_along_axis(idx_t.numpy(), ot, -1), np.take_along_axis(idx_o, oo, -1))
    print(f"  router_ref: identical expert sets for all 9 tokens: {same_sets}")
    assert same_sets
    record("router_ref topk weights (index-sorted)", np.take_along_axis(w_t.numpy(), ot, -1), np.take_along_axis(w_o, oo, -1))
    record("router_ref shared gammas", g_t.numpy(), g_o)
    with torch.no_grad():
        moe_t = model.model.layers[1].mlp(xx.view(1, 9, -1)).numpy()
    moe_o = R.moe_ref(params, "model.layers.1.mlp.", xx.numpy().reshape(1, 9, -1), R.layer_cfg(ocfg, 1))
    record("moe_ref (router + experts + shared)", moe_t, moe_o)
    with torch.no_grad():
        mlp_t = model.model.layers[0].mlp(xx).numpy()
    record("dense_mlp_ref (global_scale)", mlp_t, R.dense_mlp_ref(params, "model.layers.0.mlp.", xx.numpy()))

    # attention core: eager_attention_forward vs attention_ref on random inputs incl. GQA + bias
    from transformers.models.inkling.modeling_inkling import eager_attention_forward
    q = torch.from_numpy(rng.normal(size=(B, 2, 5, 32)))
    k = torch.from_numpy(rng.normal(size=(B, 1, 8, 32)))
    v = torch.from_numpy(rng.normal(size=(B, 1, 8, 32)))
    bias = torch.from_numpy(rng.normal(size=(B, 2, 5, 8)))
    qpos = np.arange(5) + 3
    kpos = np.arange(8)
    dist = qpos[:, None] - kpos[None, :]
    mask_full = torch.from_numpy(np.where(dist < 0, -np.inf, 0.0))[None, None]
    mask_win = torch.from_numpy(np.where((dist < 0) | (dist >= 4), -np.inf, 0.0))[None, None]
    with torch.no_grad():
        a_t, _ = eager_attention_forward(attn1, q, k, v, mask_full, scaling=1 / 32, position_bias=bias)
        a_w, _ = eager_attention_forward(attn1, q, k, v, mask_win, scaling=1 / 32, position_bias=bias)
    a_o = R.attention_ref(q.numpy(), k.numpy(), v.numpy(), bias.numpy(), qpos, kpos, 1 / 32).transpose(0, 2, 1, 3)
    a_ow = R.attention_ref(q.numpy(), k.numpy(), v.numpy(), bias.numpy(), qpos, kpos, 1 / 32, sliding_window=4).transpose(0, 2, 1, 3)
    record("attention_ref full (GQA 2:1 + bias + causal)", a_t.numpy(), a_o)
    record("attention_ref sliding window 4", a_w.numpy(), a_ow)

    # ---------------- cached decode: prefill 7, then 5 single-token steps ----------------
    print("\n[3] cached decode (transformers DynamicCache prefill 7 + 5 x 1 token) vs oracle stateful decode vs full forward")
    cache = DynamicCache(config=cfg)
    logits_steps = []
    with torch.no_grad():
        o = model(torch.from_numpy(ids[:, :7]), past_key_values=cache, use_cache=True)
        logits_steps.append(o.logits.numpy())
        for t in range(7, L):
            o = model(torch.from_numpy(ids[:, t : t + 1]), past_key_values=cache, use_cache=True)
            logits_steps.append(o.logits.numpy())
    logits_dec_t = np.concatenate(logits_steps, axis=1)
    record("transformers cached decode vs transformers full forward", logits_dec_t, out.logits.numpy())
    caches = [R.new_layer_cache() for _ in range(cfg.num_hidden_layers)]
    o7 = R.model_forward_ref(params, ocfg, ids[:, :7], positions=np.arange(7), caches=caches)
    steps = [o7["logits"]]
    for t in range(7, L):
        ot_ = R.model_forward_ref(params, ocfg, ids[:, t : t + 1], positions=np.array([t]), caches=caches)
        steps.append(ot_["logits"])
    logits_dec_o = np.concatenate(steps, axis=1)
    record("oracle stateful decode vs oracle full forward", logits_dec_o, ref["logits"])
    record("oracle stateful decode vs transformers cached decode", logits_dec_o, logits_dec_t)
    # what the local-layer cache retains
    for i, lc in enumerate(caches):
        kept = lc["k"].shape[2]
        print(f"  layer {i} ({cfg.layer_types[i]}): oracle retains {kept} keys, sconv states {[s.shape[1] for s in lc['conv']]}")
    # chunked prefill 5 + 3 + singles
    caches = [R.new_layer_cache() for _ in range(cfg.num_hidden_layers)]
    cache = DynamicCache(config=cfg)
    steps_o, steps_t = [], []
    with torch.no_grad():
        for a, b in [(0, 5), (5, 8), (8, 9), (9, 10), (10, 11), (11, 12)]:
            steps_t.append(model(torch.from_numpy(ids[:, a:b]), past_key_values=cache, use_cache=True).logits.numpy())
            steps_o.append(R.model_forward_ref(params, ocfg, ids[:, a:b], positions=np.arange(a, b), caches=caches)["logits"])
    record("chunked prefill (5,3,1,1,1,1): transformers vs full", np.concatenate(steps_t, 1), out.logits.numpy())
    record("chunked prefill (5,3,1,1,1,1): oracle vs full", np.concatenate(steps_o, 1), ref["logits"])

    # ---------------- MTP step: torch reference assembled from transformers modules ----------------
    print("\n[4] MTP depth step (torch reference built from transformers modules, vLLM/SGLang semantics) vs mtp_step_ref")
    mcfg = InklingTextConfig(**{**TINY, "layer_types": ["hybrid_sliding" if i in TINY["mtp_local_layer_ids"] else "hybrid" for i in range(TINY["num_mtp_layers"])],
                                "mlp_layer_types": ["dense"] * TINY["num_mtp_layers"], "num_hidden_layers": TINY["num_mtp_layers"]})
    mcfg._attn_implementation = "eager"
    mcfg._experts_implementation = "eager"
    torch.manual_seed(3)
    block = InklingDecoderLayer(mcfg, 0).double()
    hidden_norm = InklingRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).double()
    embed_norm = InklingRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).double()
    input_proj = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False).double()
    mtp_sd = {"mtp.layers.0.transformer_block." + k: v for k, v in block.state_dict().items()}
    mtp_sd["mtp.layers.0.hidden_norm.weight"] = hidden_norm.weight.detach()
    mtp_sd["mtp.layers.0.embed_norm.weight"] = embed_norm.weight.detach()
    mtp_sd["mtp.layers.0.input_proj.weight"] = input_proj.weight.detach()
    mtp_params = random_fill(mtp_sd, seed=99)
    block.load_state_dict({k[len("mtp.layers.0.transformer_block."):]: torch.from_numpy(v) for k, v in mtp_params.items() if "transformer_block." in k})
    hidden_norm.weight.data = torch.from_numpy(mtp_params["mtp.layers.0.hidden_norm.weight"])
    embed_norm.weight.data = torch.from_numpy(mtp_params["mtp.layers.0.embed_norm.weight"])
    input_proj.weight.data = torch.from_numpy(mtp_params["mtp.layers.0.input_proj.weight"])
    allp = {**params, **mtp_params}

    prev_hidden = out.hidden_states[-1]  # target's post-final-norm hidden states (vLLM/SGLang convention)
    next_ids = torch.from_numpy(np.concatenate([ids[:, 1:], ids[:, :1]], axis=1))
    with torch.no_grad():
        emb = model.model.embed_tokens(next_ids)  # backbone embed_norm(embed)
        comb = torch.cat([hidden_norm(prev_hidden), embed_norm(emb)], dim=-1)
        h = input_proj(comb)
        from transformers.masking_utils import create_sliding_window_causal_mask, create_causal_mask
        mask_kwargs = {"config": mcfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None, "position_ids": torch.arange(L)[None]}
        amask = create_sliding_window_causal_mask(**mask_kwargs) if mcfg.layer_types[0] == "hybrid_sliding" else create_causal_mask(**mask_kwargs)
        h = block(h, attention_mask=amask, conv_mask=None, past_key_values=None)
        logits_t = model.lm_head(h / cfg.logits_mup_width_multiplier)[..., : cfg.unpadded_vocab_size]
    logits_o, h_o = R.mtp_step_ref(allp, ocfg, prev_hidden.numpy(), next_ids.numpy(), np.arange(L), R.new_layer_cache(), depth=0, variant="vllm")
    record("mtp_step_ref chain hidden", h.numpy(), h_o)
    record("mtp_step_ref logits", logits_t.numpy(), logits_o)

    # ---------------- transformers' own generic MtpModel (fourth implementation) ----------------
    print("\n[5] transformers MtpModel (generic MTP head, Inkling weight mapping) vs mtp_step_ref")
    from transformers.modeling_layers import MtpModel
    from transformers.masking_utils import create_sliding_window_causal_mask

    def check_mtp_model(mtp_local_ids, label):
        cfg2 = InklingTextConfig(**{**TINY, "mtp_local_layer_ids": mtp_local_ids})
        cfg2._attn_implementation = "eager"
        cfg2._experts_implementation = "eager"
        model2 = InklingForCausalLM(cfg2).double().eval()
        model2.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
        ocfg2 = oracle_cfg(cfg2)
        mtp = MtpModel(model2, 1).double().eval()
        sd = {k: v for k, v in mtp.state_dict().items() if k not in ("embed_tokens.weight", "shared_head.weight")}
        mp = random_fill(sd, seed=5)
        mtp.load_state_dict({k: torch.from_numpy(v) for k, v in mp.items()}, strict=False)
        # oracle naming
        omp = {}
        for k, v in mp.items():
            k2 = k.replace("layers.0.hnorm.", "mtp.layers.0.hidden_norm.").replace("layers.0.enorm.", "mtp.layers.0.embed_norm.")
            k2 = k2.replace("layers.0.eh_proj.", "mtp.layers.0.input_proj.").replace("layers.0.mtp_block.", "mtp.layers.0.transformer_block.")
            omp[k2] = v
        allp2 = {**params, **omp}
        print(f"  [{label}] MtpModel: use_post_norm={mtp.use_post_norm} use_shared_post_norm={mtp.use_shared_post_norm} "
              f"embed_norm={type(mtp.embed_tokens.embed_norm).__name__} mtp layer_types={mtp.config.layer_types}")
        pos = torch.arange(L)[None].expand(B, -1)
        with torch.no_grad():
            emb = mtp.embed_tokens(next_ids)
            masks = mtp.create_masks_for_mtp_layer(0, emb, None, pos)
            am = masks["attention_mask"]
            row = am[0, 0, 4].numpy() if am is not None else None
            print(f"  [{label}] MtpModel attention mask row for query 4 (window {cfg2.sliding_window_size}): visible keys = "
                  f"{[int(j) for j in np.nonzero(row > -1e300)[0]] if row is not None else 'None (is_causal)'}")
            h_tm = mtp.forward_layer(0, prev_hidden, pos, None, input_ids=next_ids)
            logits_tm = mtp._project_to_logits(h_tm)
        for variant in ("transformers", "vllm", "mlx"):
            lo_v, h_v = R.mtp_step_ref(allp2, ocfg2, prev_hidden.numpy(), next_ids.numpy(), np.arange(L), R.new_layer_cache(), 0, variant=variant)
            record(f"[{label}] MtpModel chain hidden vs mtp_step_ref variant={variant}", h_tm.numpy(), h_v)
            record(f"[{label}] MtpModel logits vs mtp_step_ref variant={variant}", logits_tm.numpy(), lo_v)
        if mtp_local_ids:
            # same MtpModel block, but with the sliding-window mask the main model would use
            with torch.no_grad():
                comb = torch.cat([mtp.layers[0].hnorm(prev_hidden), mtp.layers[0].enorm(emb)], -1)
                h0 = mtp.layers[0].eh_proj(comb)
                am2 = create_sliding_window_causal_mask(config=mtp.config, inputs_embeds=h0, attention_mask=None, past_key_values=None, position_ids=pos[:1])
                h_fixed = mtp.layers[0].mtp_block(h0, attention_mask=am2, conv_mask=None, past_key_values=None)
            _, h_v = R.mtp_step_ref(allp2, ocfg2, prev_hidden.numpy(), next_ids.numpy(), np.arange(L), R.new_layer_cache(), 0, variant="transformers")
            record(f"[{label}] MtpModel block with sliding mask forced vs variant=transformers", h_fixed.numpy(), h_v)

    check_mtp_model([0], "local depth")
    check_mtp_model([], "global depth")

    # ---------------- generate(): masks built through LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING ----------------
    print("\n[6] generate() path: first-step logits vs plain forward (masks from create_masks_for_generate)")
    from transformers.masking_utils import create_masks_for_generate
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.from_numpy(ids))
        gm = create_masks_for_generate(cfg, emb, torch.ones(B, L, dtype=torch.long), None, torch.arange(L)[None].expand(B, -1))
        if isinstance(gm, dict):
            row = gm["sliding_attention"][0, 0, 8].numpy()
            print(f"  create_masks_for_generate['sliding_attention'] row for query 8 (window {cfg.sliding_window_size}): visible keys = {[int(j) for j in np.nonzero(row > -1e300)[0]]}")
            fwd_with_gen_masks = model(torch.from_numpy(ids), attention_mask=gm, use_cache=False).logits.numpy()
            record("forward with generate-style mask dict vs plain forward logits", fwd_with_gen_masks, out.logits.numpy())
        else:
            print("  create_masks_for_generate returned a plain mask:", type(gm))
        gen = model.generate(torch.from_numpy(ids), attention_mask=torch.ones(B, L, dtype=torch.long), max_new_tokens=1,
                             do_sample=False, output_logits=True, return_dict_in_generate=True)
    record("generate() first-step logits vs oracle last-position logits", gen.logits[0].numpy(), ref["logits"][:, -1, :])

    # Pass criterion: every same-semantics check. Excluded (they document discrepancies): the cross-variant MTP rows and
    # the local-depth MtpModel rows (transformers builds a full causal mask for a hybrid_sliding MTP depth).
    def is_core(k):
        if k.startswith("[local depth]"):
            return k.endswith("sliding mask forced vs variant=transformers")
        if k.startswith("[global depth]"):
            return k.endswith("variant=transformers")
        if "generate" in k:
            return False  # documents the generate()-path mask behaviour; reported, not asserted
        return True
    core = {k: v for k, v in results.items() if is_core(k)}
    worst = max(v["max_abs"] for v in core.values())
    # transformers upcasts RMSNorm / sconv internals to float32 even with float64 parameters, so
    # agreement is at float32 rounding (~1e-7 relative), not float64.
    print(f"\nWORST max_abs over all same-semantics checks: {worst:.3e}  (float32 internals in transformers norms/sconv)")
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_transformers.json"), "w"), indent=1)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_params.npy"), {"params": params, "mtp_params": mtp_params, "cfg": ocfg, "ids": ids, "tiny": TINY}, allow_pickle=True)
    assert worst < 1e-4, "oracle disagrees with transformers beyond float32 rounding"
    print("OK")


if __name__ == "__main__":
    main()
