"""Tiny Inkling configs and random weights for development (no real weights)."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import inkling_mm_model as M


def tiny_text_config(**over):
    cfg = dict(
        hidden_size=64, num_hidden_layers=3, vocab_size=64, unpadded_vocab_size=60,
        rms_norm_eps=1e-6, use_embed_norm=True, logits_mup_width_multiplier=16.0,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        sliding_window_size=4, layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding"],
        d_rel=4, rel_extent=8, log_scaling_n_floor=4, log_scaling_alpha=0.1,
        sconv_kernel_size=4, mlp_layer_types=["dense", "sparse", "sparse"],
        intermediate_size=32, dense_intermediate_size=96, n_routed_experts=8,
        num_experts_per_tok=2, n_shared_experts=2, route_scale=8.0,
    )
    cfg.update(over)
    return cfg


def random_fill(model, seed=0, scale=0.3):
    """Make every parameter random and non-trivial (norm gains, global scales, biases, taps)."""
    mx.random.seed(seed)
    from mlx.utils import tree_flatten, tree_unflatten
    new = []
    for k, v in tree_flatten(model.parameters()):
        if not mx.issubdtype(v.dtype, mx.floating):
            new.append((k, v)); continue
        if k.endswith("norm.weight") or k.endswith("global_scale"):
            r = 1.0 + scale * mx.random.normal(v.shape)
        elif k.endswith("_sconv.weight"):
            r = scale * mx.random.normal(v.shape)
        else:
            r = mx.random.normal(v.shape) * (scale / (v.shape[-1] ** 0.5) if v.ndim >= 2 else scale)
        new.append((k, r.astype(v.dtype)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model


def make_model(seed=0, dtype=mx.float32, **over):
    args = M.ModelArgs(model_type="inkling_mm_model", text_config=tiny_text_config(**over))
    model = M.Model(args)
    random_fill(model, seed)
    model.update(__import__("mlx.utils", fromlist=["tree_map"]).tree_map(
        lambda p: p.astype(dtype) if (mx.issubdtype(p.dtype, mx.floating) and p.dtype != mx.float32) else p, model.parameters()))
    return model, args
