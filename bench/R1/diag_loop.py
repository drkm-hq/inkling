"""Hazard reproduction: a float32 `global_scale` on InklingDenseMLP (language.py
L482-490: `super().__call__(x) * self.global_scale`) promotes the residual
stream to float32 for the rest of the forward. Every later bf16 weight
(attention projections, lm_head) is then upcast per call and all activations
double in size. The mlx-community checkpoints store it BF16, so they are not
affected; a checkpoint that keeps it fp32 (the reference keeps scales in fp32)
would be. Measures the 3-layer synthetic model both ways at ctx=0.
Results -> results/diag_loop.json"""
import json
import os
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_bench import build_model, make_caches, med, time_full_decode

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {}
E = int(sys.argv[1]) if len(sys.argv) > 1 else 2
y = mx.array([[1234]])
for name, dt in (("bf16_scale", mx.bfloat16), ("fp32_scale", mx.float32)):
    model, cfg = build_model(E, None, None, dense_scale_dtype=dt)
    caches = make_caches(model, 0)
    pre = model.model(y, caches, None, skip_final_norm=True)
    mx.eval(pre)
    r = {"residual_dtype": str(pre.dtype)}
    r["layer0_out_dtype"] = str(model.model.layers[0](model.model.embed(y), cache=caches[0]).dtype)
    r["full_model_ms"] = med(lambda: mx.argmax(model(y, cache=caches).logits[:, -1, :], axis=-1), 3, 9)
    r["model_no_head_ms"] = med(lambda: model.model(y, caches, None, skip_final_norm=True), 3, 9)
    r["head_on_pre_ms"] = med(lambda: model._logits_from_norm(model.model.norm(pre)), 3, 9)
    h = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    mx.eval(h)
    r["head_on_bf16_ms"] = med(lambda: model._logits_from_norm(model.model.norm(h)), 3, 9)
    r["loop"] = time_full_decode(model, caches)
    res[name] = r
    print(name, json.dumps(r, indent=1), flush=True)
    del model, caches, pre
    mx.clear_cache()
with open(os.path.join(OUT, "diag_loop.json"), "w") as f:
    json.dump(res, f, indent=1)
print("wrote diag_loop.json")
