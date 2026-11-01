"""Per-token dispatch cost of one decode step: the mlx-vlm Inkling LanguageModel vs the mlx-lm
model on the same tiny-width, full-depth configuration (42 layers, 35 sliding / 7 global, 256+2
experts top-6, window 512, extent 1024, random weights). At tiny widths the GPU work is
negligible, so the step time is the graph-build plus kernel-launch cost, i.e. a proxy for the
number of ops per token. Each step is one synchronous eval after a 600-token prefill.

  python dispatch_bench.py [--iters 40]
"""
import argparse
import statistics
import time

import mlx.core as mx

HIDDEN, HEADS, KV, D, DREL = 64, 4, 2, 16, 4
LOCAL = [i for i in range(42) if (i + 1) % 6 != 0]
TINY = dict(
    hidden_size=HIDDEN, num_hidden_layers=42, vocab_size=256, unpadded_vocab_size=250,
    logits_mup_width_multiplier=16.0, num_attention_heads=HEADS, num_key_value_heads=KV,
    head_dim=D, sliding_window_size=512, local_layer_ids=LOCAL, d_rel=DREL, rel_extent=1024,
    log_scaling_n_floor=128000, log_scaling_alpha=0.1, sconv_kernel_size=4, dense_mlp_idx=2,
    intermediate_size=16, dense_intermediate_size=64, n_routed_experts=256, num_experts_per_tok=6,
    n_shared_experts=2, route_scale=8.0,
)


def timeit(name, step, iters):
    ts = []
    for i in range(iters + 5):
        t0 = time.perf_counter()
        mx.eval(step())
        t = time.perf_counter() - t0
        if i >= 5:
            ts.append(t)
    print(f"  {name:28s} {statistics.median(ts) * 1e3:.2f} ms/token (best {min(ts) * 1e3:.2f})")


def bench_mlx_lm(iters):
    from mlx_lm.models import inkling_mm_model as M
    from mlx_lm.models.cache import make_prompt_cache

    model = M.Model(M.ModelArgs.from_dict({"model_type": "inkling_mm_model", "text_config": TINY}))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    mx.eval(model(mx.random.randint(0, 250, (1, 600)), cache=cache))
    y = mx.array([[7]])

    def step():
        return model(y, cache=cache)

    timeit("mlx-lm inkling_mm_model", step, iters)


def bench_mlx_vlm(iters):
    from mlx_vlm.models.inkling.config import TextConfig
    from mlx_vlm.models.inkling.language import LanguageModel

    cfg = TextConfig(
        hidden_size=HIDDEN, num_hidden_layers=42, vocab_size=256, unpadded_vocab_size=250,
        logits_mup_width_multiplier=16.0, num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=D,
        swa_num_attention_heads=HEADS, swa_num_key_value_heads=KV, swa_head_dim=D, sliding_window_size=512,
        local_layer_ids=LOCAL, d_rel=DREL, rel_extent=1024, log_scaling_n_floor=128000, log_scaling_alpha=0.1,
        sconv_kernel_size=4, dense_mlp_idx=2, intermediate_size=16, dense_intermediate_size=64,
        n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=2, route_scale=8.0,
    )
    lm = LanguageModel(cfg)
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, 250, (1, 600)), cache=cache).logits)
    y = mx.array([[7]])

    def step():
        return lm(y, cache=cache).logits

    timeit("mlx-vlm LanguageModel", step, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--impl", choices=["both", "mlx_vlm", "mlx_lm"], default="both")
    args = ap.parse_args()
    import sys

    print(f"decode step, tiny widths, 42 layers, context 600 (sliding window full), bf16 [{sys.prefix}]")
    for _ in range(2):
        if args.impl in ("both", "mlx_vlm"):
            bench_mlx_vlm(args.iters)
        if args.impl in ("both", "mlx_lm"):
            bench_mlx_lm(args.iters)


if __name__ == "__main__":
    main()
