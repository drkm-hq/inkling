"""Pure-numpy (float64) reference semantics for the Inkling-specific operations.

The parameter layout follows the transformers ``InklingForCausalLM`` state_dict
(``model.layers.{i}.self_attn.q_proj.weight`` etc.), which is the reference
implementation every function here is tested against (see test_oracle_vs_transformers.py).
Checkpoint-layout helpers (interleaved ``w13`` de-interleave, conv weight squeeze) live at
the bottom.

Conventions
-----------
* Linear weights are ``[out, in]`` and applied as ``x @ W.T`` (torch.nn.Linear).
* Short-conv weights are the checkpoint tensors ``[C, 1, K]`` (``nn.Conv1d`` depthwise);
  ``w[c, 0, K-1]`` multiplies the *current* token, ``w[c, 0, 0]`` the token ``K-1`` steps back.
* Attention tensors are ``[B, H, L, D]``; GQA maps query head ``h`` to kv head ``h // (H // Hkv)``.
* ``positions`` are absolute token positions (0-based) of the query tokens; a cache carries the
  absolute positions of its keys.
* All math is float64; no intermediate rounding.
"""

from __future__ import annotations

import numpy as np

F64 = np.float64


# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------
def rmsnorm_ref(x, w, eps=1e-6):
    """RMSNorm: ``w * x / sqrt(mean(x^2) + eps)`` (weight applied after normalisation)."""
    x = np.asarray(x, F64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return np.asarray(w, F64) * (x / np.sqrt(var + eps))


def silu(x):
    x = np.asarray(x, F64)
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, F64)))


def logsigmoid(x):
    return -np.logaddexp(0.0, -np.asarray(x, F64))


def logsumexp(x, axis=-1, keepdims=True):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


# ---------------------------------------------------------------------------
# Short convolution (sconv)
# ---------------------------------------------------------------------------
def sconv_ref(x, w, state=None):
    """Depthwise causal short convolution with the built-in residual.

    y[b, t, c] = x[b, t, c] + sum_{k=0}^{K-1} w[c, k] * x[b, t - (K-1) + k, c]

    with ``x[t'] := state[t' + (K-1)]`` for ``t' < 0`` (zeros when ``state`` is None, i.e.
    the start of a sequence). No bias, no activation.

    x      : [B, L, C]
    w      : [C, 1, K] (checkpoint layout) or [C, K]
    state  : [B, K-1, C] -- the previous K-1 *inputs* of this stream (what a decode cache must
             retain), or None
    returns (y [B, L, C], new_state [B, K-1, C])
    """
    x = np.asarray(x, F64)
    B, L, C = x.shape
    w = np.asarray(w, F64).reshape(C, -1)
    K = w.shape[1]
    if state is None:
        state = np.zeros((B, K - 1, C), F64)
    xp = np.concatenate([np.asarray(state, F64), x], axis=1)  # [B, K-1+L, C]
    conv = np.zeros_like(x)
    for k in range(K):
        conv += w[:, k] * xp[:, k : k + L, :]
    y = x + conv
    new_state = xp[:, L:, :]
    return y, new_state


# ---------------------------------------------------------------------------
# Relative position bias
# ---------------------------------------------------------------------------
def rel_logits_ref(r, proj):
    """``rel_logits[b, h, i, e] = sum_d r[b, i, h, d] * proj[d, e]`` -> [B, H, Lq, E]."""
    return np.einsum("bihd,de->bhie", np.asarray(r, F64), np.asarray(proj, F64))


def rel_bias_ref(r, proj, q_pos, k_pos, rel_extent=None):
    """Hidden-state conditioned relative position bias.

    r        : [B, Lq, H, d_rel]   per-token, per-head relative states (``r_proj`` output)
    proj     : [d_rel, E]          learned bank of bias-vs-distance profiles (E = rel_extent)
    q_pos    : [Lq] absolute query positions, k_pos: [Lk] absolute key positions
    returns bias [B, H, Lq, Lk] with
        dist = q_pos[i] - k_pos[j]
        bias[b,h,i,j] = (r[b,i,h,:] @ proj)[dist]   if 0 <= dist < E
                      = 0                           otherwise (in-context but beyond the band,
                                                    or future tokens -- causality is a mask)
    """
    proj = np.asarray(proj, F64)
    E = proj.shape[1]
    if rel_extent is not None:
        assert rel_extent == E, (rel_extent, E)
    rl = rel_logits_ref(r, proj)  # [B, H, Lq, E]
    B, H, Lq, _ = rl.shape
    q_pos = np.asarray(q_pos)
    k_pos = np.asarray(k_pos)
    dist = q_pos[:, None] - k_pos[None, :]  # [Lq, Lk]
    idx = np.clip(dist, 0, E - 1)
    bias = np.take_along_axis(rl, np.broadcast_to(idx, (B, H, Lq, dist.shape[1])), axis=-1)
    in_band = (dist >= 0) & (dist < E)
    return np.where(in_band[None, None], bias, 0.0)


# ---------------------------------------------------------------------------
# Attention core
# ---------------------------------------------------------------------------
def log_scaling_tau_ref(q_pos, n_floor, alpha):
    """tau[i] = 1 + alpha * log(max((q_pos[i] + 1) / n_floor, 1))  (global layers only)."""
    eff = (np.asarray(q_pos, F64) + 1.0) / float(n_floor)
    return 1.0 + alpha * np.log(np.maximum(eff, 1.0))


def attention_ref(q, k, v, bias, q_pos, k_pos, scale, sliding_window=None, log_scaling=None,
                  return_probs=False):
    """Softmax attention with additive relative bias, causal + optional sliding-window mask,
    GQA, and the optional long-context logit log-scaling.

    q    : [B, H, Lq, D]   (already per-head RMS-normed; the kernel does not norm)
    k    : [B, Hkv, Lk, D] (already per-head RMS-normed, post-sconv), v: [B, Hkv, Lk, D]
    bias : [B, H, Lq, Lk] or None
    scale: 1 / head_dim (NOT 1/sqrt(head_dim) -- q/k are unit-RMS so the dot product is O(D))
    sliding_window: int or None. Key j is visible to query i iff 0 <= q_pos[i]-k_pos[j] < window
                    (the window counts the current token; 512 -> 512 keys incl. self).
    log_scaling: (n_floor, alpha) or None. logits are multiplied by tau[i] (equivalently q and
                 bias are both scaled by tau, as the reference does).

    logits[b,h,i,j] = tau[i] * (scale * <q[b,h,i], k[b,h//g,j]> + bias[b,h,i,j]); masked -> -inf
    out = softmax_j(logits) @ v
    returns out [B, H, Lq, D] (and probs if requested)
    """
    q = np.asarray(q, F64)
    k = np.asarray(k, F64)
    v = np.asarray(v, F64)
    B, H, Lq, D = q.shape
    Hkv = k.shape[1]
    g = H // Hkv
    kr = np.repeat(k, g, axis=1)  # repeat_interleave over heads: head h -> kv head h // g
    vr = np.repeat(v, g, axis=1)
    logits = np.einsum("bhid,bhjd->bhij", q, kr) * float(scale)
    if bias is not None:
        logits = logits + np.asarray(bias, F64)
    q_pos = np.asarray(q_pos)
    k_pos = np.asarray(k_pos)
    if log_scaling is not None:
        tau = log_scaling_tau_ref(q_pos, *log_scaling)
        logits = logits * tau[None, None, :, None]
    dist = q_pos[:, None] - k_pos[None, :]
    masked = dist < 0
    if sliding_window is not None:
        masked = masked | (dist >= int(sliding_window))
    logits = np.where(masked[None, None], -np.inf, logits)
    m = np.max(logits, axis=-1, keepdims=True)
    p = np.exp(logits - m)
    p = p / np.sum(p, axis=-1, keepdims=True)
    out = np.einsum("bhij,bhjd->bhid", p, vr)
    return (out, p) if return_probs else out


# ---------------------------------------------------------------------------
# Attention sublayer (projections, sconv on k/v, norms, cache, rel bias, attention, o_proj)
# ---------------------------------------------------------------------------
def new_layer_cache():
    """Per-layer decode state: 4 sconv states (k, v, attn-out, mlp-out), the (normed) K / V
    entries this layer must retain, and their absolute positions."""
    return {"conv": [None, None, None, None], "k": None, "v": None, "k_pos": None}


def attn_sublayer_ref(p, pre, x, positions, lcfg, cache):
    """x: [B, L, hidden] (post input_layernorm). Returns o_proj output [B, L, hidden].

    Order (matches transformers InklingAttention.forward):
      q = x Wq^T; k = sconv_k(x Wk^T); v = sconv_v(x Wv^T); r = x Wr^T
      q = rmsnorm_head(q) ; k = rmsnorm_head(k)        (no bias on any projection)
      cache update (k, v after norm/sconv)
      bias = rel_bias(r, proj, q_pos, k_pos) ; tau log scaling on global layers
      attention(scale = 1/head_dim, window on local layers) ; o_proj
    """
    H, Hkv, D, d_rel = lcfg["num_heads"], lcfg["num_kv_heads"], lcfg["head_dim"], lcfg["d_rel"]
    eps = lcfg["eps"]
    x = np.asarray(x, F64)
    B, L, _ = x.shape
    q = x @ p[pre + "q_proj.weight"].T
    k = x @ p[pre + "k_proj.weight"].T
    v = x @ p[pre + "v_proj.weight"].T
    r = x @ p[pre + "r_proj.weight"].T
    k, cache["conv"][0] = sconv_ref(k, p[pre + "k_sconv.conv1d.weight"], cache["conv"][0])
    v, cache["conv"][1] = sconv_ref(v, p[pre + "v_sconv.conv1d.weight"], cache["conv"][1])
    q = rmsnorm_ref(q.reshape(B, L, H, D), p[pre + "q_norm.weight"], eps).transpose(0, 2, 1, 3)
    k = rmsnorm_ref(k.reshape(B, L, Hkv, D), p[pre + "k_norm.weight"], eps).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)
    positions = np.asarray(positions)
    if cache["k"] is None:
        k_all, v_all, k_pos = k, v, positions
    else:
        k_all = np.concatenate([cache["k"], k], axis=2)
        v_all = np.concatenate([cache["v"], v], axis=2)
        k_pos = np.concatenate([cache["k_pos"], positions])
    r = r.reshape(B, L, H, d_rel)
    bias = rel_bias_ref(r, p[pre + "rel_logits_proj.proj"], positions, k_pos, lcfg["rel_extent"])
    out = attention_ref(q, k_all, v_all, bias, positions, k_pos, 1.0 / D,
                        sliding_window=lcfg["sliding_window"], log_scaling=lcfg["log_scaling"])
    # retain: local layers keep the last (window - 1) keys, global layers keep everything
    if lcfg["sliding_window"] is not None:
        keep = int(lcfg["sliding_window"]) - 1
        k_all, v_all, k_pos = k_all[:, :, -keep:], v_all[:, :, -keep:], k_pos[-keep:]
    cache["k"], cache["v"], cache["k_pos"] = k_all, v_all, k_pos
    out = out.transpose(0, 2, 1, 3).reshape(B, L, H * D)
    return out @ p[pre + "o_proj.weight"].T


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------
def router_ref(x, gate_w, corr_bias, global_scale, route_scale, top_k, n_shared):
    """Sigmoid gate with selection bias, log-sigmoid renormalisation over the top-k routed
    + the n_shared sink logits, times route_scale * global_scale.

    x          : [N, hidden]  (the post_attention_layernorm output; NO extra router norm)
    gate_w     : [R + S, hidden] -- rows [0, R) routed experts, rows [R, R+S) shared sinks
    corr_bias  : [R] selection-only bias (``e_score_correction_bias`` / ``gate.bias``)
    returns (topk_idx [N, K], topk_w [N, K], shared_gamma [N, S], logits [N, R+S])

      logits = x @ gate_w^T
      sel    = sigmoid(logits[:, :R]) + corr_bias          (selection only)
      topk_idx = top-K of sel  (ties -> lowest index)
      active = [logits[:, topk_idx], logits[:, R:]]        (K + S raw logits)
      w      = softmax(logsigmoid(active)) * route_scale * global_scale
             = sigmoid(active_j) / sum_j' sigmoid(active_j') * route_scale * global_scale
      topk_w = w[:, :K], shared_gamma = w[:, K:]
    """
    x = np.asarray(x, F64)
    gate_w = np.asarray(gate_w, F64)
    R = gate_w.shape[0] - n_shared
    logits = x @ gate_w.T
    sel = sigmoid(logits[:, :R]) + np.asarray(corr_bias, F64)
    topk_idx = np.argsort(-sel, axis=-1, kind="stable")[:, :top_k]
    chosen = np.take_along_axis(logits[:, :R], topk_idx, axis=-1)
    active = np.concatenate([chosen, logits[:, R:]], axis=-1)
    lp = logsigmoid(active)
    w = np.exp(lp - logsumexp(lp)) * float(route_scale) * float(np.asarray(global_scale).reshape(-1)[0])
    return topk_idx, w[:, :top_k], w[:, top_k:], logits


def experts_ref(x, topk_idx, topk_w, gate_up, down):
    """Routed experts: ``sum_j topk_w[n,j] * down[e] @ (silu(gate_e @ x) * (up_e @ x))``.

    gate_up: [E, 2I, hidden] with gate = rows [0, I), up = rows [I, 2I)  (transformers layout,
             de-interleaved from the checkpoint's [g0, u0, g1, u1, ...] rows)
    down   : [E, hidden, I]
    """
    x = np.asarray(x, F64)
    N, K = topk_idx.shape
    out = np.zeros_like(x)
    for n in range(N):
        for j in range(K):
            e = int(topk_idx[n, j])
            gu = gate_up[e] @ x[n]
            I = gu.shape[0] // 2
            h = silu(gu[:I]) * gu[I:]
            out[n] += topk_w[n, j] * (down[e] @ h)
    return out


def shared_experts_ref(x, gammas, gate_w, up_w, down_w):
    """Shared (sink) experts: ``sum_s down_s @ (silu(gate_s x) * (up_s x) * gamma[:, s])``.
    gate_w/up_w: [S, I, hidden]; down_w: [S, hidden, I]; gammas: [N, S]."""
    x = np.asarray(x, F64)
    out = np.zeros_like(x)
    for s in range(gate_w.shape[0]):
        h = silu(x @ gate_w[s].T) * (x @ up_w[s].T) * gammas[:, s : s + 1]
        out += h @ down_w[s].T
    return out


def moe_ref(p, pre, x, lcfg):
    """Full sparse MLP: router + routed experts + shared experts. x: [B, L, hidden]."""
    x = np.asarray(x, F64)
    B, L, Dm = x.shape
    flat = x.reshape(-1, Dm)
    topk_idx, topk_w, gammas, _ = router_ref(
        flat, p[pre + "gate.weight"], p[pre + "gate.e_score_correction_bias"],
        p[pre + "gate.global_scale"], lcfg["route_scale"], lcfg["top_k"], lcfg["n_shared"])
    y = experts_ref(flat, topk_idx, topk_w, p[pre + "experts.gate_up_proj"], p[pre + "experts.down_proj"])
    y = y + shared_experts_ref(flat, gammas, p[pre + "shared_experts.gate_proj"],
                               p[pre + "shared_experts.up_proj"], p[pre + "shared_experts.down_proj"])
    return y.reshape(B, L, Dm)


def dense_mlp_ref(p, pre, x):
    """Dense SwiGLU MLP with the learned scalar output scale: ``global_scale * down(silu(gate x) * up x)``."""
    x = np.asarray(x, F64)
    h = silu(x @ p[pre + "gate_proj.weight"].T) * (x @ p[pre + "up_proj.weight"].T)
    return (h @ p[pre + "down_proj.weight"].T) * np.asarray(p[pre + "global_scale"], F64).reshape(-1)[0]


# ---------------------------------------------------------------------------
# Decoder layer and model
# ---------------------------------------------------------------------------
def decoder_layer_ref(p, pre, x, positions, lcfg, cache):
    """One Inkling block (pre-norm, residual, four sconvs):

      h = x + attn_sconv( attn( attn_norm(x) ) )
      y = h + mlp_sconv ( mlp ( mlp_norm (h) ) )
    where each sconv already contains its own ``+ input`` residual.
    """
    eps = lcfg["eps"]
    x = np.asarray(x, F64)
    h = rmsnorm_ref(x, p[pre + "input_layernorm.weight"], eps)
    a = attn_sublayer_ref(p, pre + "self_attn.", h, positions, lcfg, cache)
    a, cache["conv"][2] = sconv_ref(a, p[pre + "attn_sconv.conv1d.weight"], cache["conv"][2])
    x = x + a
    h = rmsnorm_ref(x, p[pre + "post_attention_layernorm.weight"], eps)
    m = moe_ref(p, pre + "mlp.", h, lcfg) if lcfg["sparse"] else dense_mlp_ref(p, pre + "mlp.", h)
    m, cache["conv"][3] = sconv_ref(m, p[pre + "mlp_sconv.conv1d.weight"], cache["conv"][3])
    return x + m


def layer_cfg(cfg, i, mtp=False):
    """Per-layer config dict from a model config dict (see test for the keys)."""
    types = cfg["mtp_layer_types"] if mtp else cfg["layer_types"]
    mlp_types = (["dense"] * len(types)) if mtp else cfg["mlp_layer_types"]
    sliding = types[i] == "hybrid_sliding"
    return {
        "num_heads": cfg["swa_num_attention_heads"] if sliding else cfg["num_attention_heads"],
        "num_kv_heads": cfg["swa_num_key_value_heads"] if sliding else cfg["num_key_value_heads"],
        "head_dim": cfg["swa_head_dim"] if sliding else cfg["head_dim"],
        "d_rel": cfg["d_rel"],
        "rel_extent": cfg["sliding_window_size"] if sliding else cfg["rel_extent"],
        "sliding_window": cfg["sliding_window_size"] if sliding else None,
        "log_scaling": (None if (sliding or cfg.get("log_scaling_n_floor") is None)
                        else (cfg["log_scaling_n_floor"], cfg["log_scaling_alpha"])),
        "sparse": mlp_types[i] == "sparse",
        "eps": cfg["rms_norm_eps"],
        "top_k": cfg["num_experts_per_tok"],
        "n_shared": cfg["n_shared_experts"],
        "route_scale": cfg["route_scale"],
    }


def embed_ref(p, cfg, input_ids, prefix="model."):
    """Token embedding followed by the embedding RMSNorm (no muP multiplier on the input side)."""
    x = p[prefix + "embed_tokens.weight"][np.asarray(input_ids)]
    return rmsnorm_ref(x, p[prefix + "embed_tokens.embed_norm.weight"], cfg["rms_norm_eps"])


def lm_head_ref(p, cfg, hidden_normed):
    """logits = (h / logits_mup_width_multiplier) @ unembed^T, sliced to unpadded_vocab_size."""
    logits = (np.asarray(hidden_normed, F64) / float(cfg["logits_mup_width_multiplier"])) @ p["lm_head.weight"].T
    uv = cfg.get("unpadded_vocab_size")
    if uv is not None and uv < logits.shape[-1]:
        logits = logits[..., :uv]
    return logits


def model_forward_ref(p, cfg, input_ids, positions=None, caches=None, prefix="model."):
    """Full text model. input_ids: [B, L]. Returns dict(logits, hidden (pre final norm),
    normed (post final norm), layer_outputs (list of per-layer residual-stream outputs), caches)."""
    input_ids = np.asarray(input_ids)
    B, L = input_ids.shape
    n_layers = len(cfg["layer_types"])
    if caches is None:
        caches = [new_layer_cache() for _ in range(n_layers)]
    if positions is None:
        positions = np.arange(L)
    x = embed_ref(p, cfg, input_ids, prefix)
    layer_outputs = []
    for i in range(n_layers):
        x = decoder_layer_ref(p, f"{prefix}layers.{i}.", x, positions, layer_cfg(cfg, i), caches[i])
        layer_outputs.append(x)
    normed = rmsnorm_ref(x, p[prefix + "norm.weight"], cfg["rms_norm_eps"])
    return {"logits": lm_head_ref(p, cfg, normed), "hidden": x, "normed": normed,
            "layer_outputs": layer_outputs, "caches": caches}


# ---------------------------------------------------------------------------
# MTP (multi-token prediction) depth step
# ---------------------------------------------------------------------------
MTP_VARIANTS = {
    # name: (apply backbone embed_norm to the token embedding, apply backbone final norm before the head)
    "vllm": (True, False),          # vLLM nvidia/mtp.py and SGLang InklingMTPLayer
    "sglang": (True, False),
    "transformers": (False, False),  # transformers MtpModel (embed_norm replaced by Identity)
    "mlx": (False, True),            # mlx-vlm InklingMTPDraftModel (target's norm before lm_head)
}


def mtp_step_ref(p, cfg, prev_hidden, token_ids, positions, cache, depth=0, variant="vllm",
                 prefix="mtp."):
    """One MTP depth layer applied to a run of tokens.

    prev_hidden : [B, L, hidden] the hidden state the depth conditions on. vLLM, SGLang and
                  transformers feed the target's POST-final-norm hidden; mlx-vlm feeds the
                  PRE-final-norm hidden (the caller chooses; see test).
    token_ids   : [B, L] the token *following* each hidden state's token (the next token)
    returns (logits [B, L, V_unpadded], chain_hidden [B, L, hidden]) where chain_hidden is what
    the next depth receives as ``prev_hidden``.

    Common core (all four implementations):
        e  = embed_norm_depth( E )                     # E = embed[token], optionally backbone-normed
        hN = hidden_norm_depth( prev_hidden )
        h  = input_proj( [hN | e] )                    # mtp_hidden_states_first = True -> hidden first
        h  = decoder_block(h)   (dense MLP, own sconv/KV caches, local or global attention
                                 per mtp_local_layer_ids; rel_extent = window on local depths)
        h  = chain_norm(h) if chain_hidden_post_norm   # Inkling-Small: false, no tensor in ckpt
        chain_hidden = h
    variant="vllm"/"sglang":  E = embed_norm_backbone(embed[token]);  logits = (h / mup) @ unembed^T
    variant="transformers":   E = embed[token] (raw);                 logits = (h / mup) @ unembed^T
    variant="mlx":            E = embed[token] (raw);                 logits = (final_norm(h) / mup) @ unembed^T
    """
    use_backbone_embed_norm, head_norm = MTP_VARIANTS[variant]
    pre = f"{prefix}layers.{depth}."
    eps = cfg["rms_norm_eps"]
    emb = p["model.embed_tokens.weight"][np.asarray(token_ids)]
    if use_backbone_embed_norm:
        emb = rmsnorm_ref(emb, p["model.embed_tokens.embed_norm.weight"], eps)
    e = rmsnorm_ref(emb, p[pre + "embed_norm.weight"], eps)
    hN = rmsnorm_ref(prev_hidden, p[pre + "hidden_norm.weight"], eps)
    comb = np.concatenate([hN, e], axis=-1) if cfg.get("mtp_hidden_states_first", True) \
        else np.concatenate([e, hN], axis=-1)
    h = comb @ p[pre + "input_proj.weight"].T
    h = decoder_layer_ref(p, pre + "transformer_block.", h, positions, layer_cfg(cfg, depth, mtp=True), cache)
    if cfg.get("chain_hidden_post_norm", False):
        h = rmsnorm_ref(h, p[prefix + "chain_norm.weight"], eps)
    head_in = rmsnorm_ref(h, p["model.norm.weight"], eps) if head_norm else h
    return lm_head_ref(p, cfg, head_in), h


# ---------------------------------------------------------------------------
# Checkpoint-layout helpers
# ---------------------------------------------------------------------------
def deinterleave_w13(w13):
    """Checkpoint fused gate/up rows are interleaved ``[g0, u0, g1, u1, ...]`` along the output
    dim: ``w13[..., 0::2, :]`` is gate, ``w13[..., 1::2, :]`` is up. Returns (gate, up)."""
    return w13[..., 0::2, :], w13[..., 1::2, :]


def w13_to_transformers_gate_up(w13):
    """[E, 2I (interleaved), H] -> transformers ``gate_up_proj`` [E, 2I, H] = cat(gate, up)."""
    g, u = deinterleave_w13(w13)
    return np.concatenate([g, u], axis=-2)


def sconv_weight_2d(w):
    """Checkpoint ``[C, 1, K]`` -> ``[C, K]`` (tap K-1 = current token)."""
    return np.asarray(w).reshape(w.shape[0], -1)


__all__ = [
    "rmsnorm_ref", "silu", "sigmoid", "logsigmoid", "sconv_ref", "rel_logits_ref", "rel_bias_ref",
    "log_scaling_tau_ref", "attention_ref", "attn_sublayer_ref", "router_ref", "experts_ref",
    "shared_experts_ref", "moe_ref", "dense_mlp_ref", "decoder_layer_ref", "layer_cfg", "embed_ref",
    "lm_head_ref", "model_forward_ref", "mtp_step_ref", "new_layer_cache", "deinterleave_w13",
    "w13_to_transformers_gate_up", "sconv_weight_2d",
]
