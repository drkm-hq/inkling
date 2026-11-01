"""Cost model inputs for Inkling MTP speculative decoding (mlx_vlm/speculative/mtp.py
+ language.py L957-986), measured on the 3-layer synthetic model:
  * target verify forward for N = 1, 3, 5, 8 tokens (block_size = drafts + 1)
  * _snapshot_cache_state + _restore_cache_state cost (full KV copy) at ctx 2K / 32K
  * a drafter-like step: one dense decoder layer + lm_head on 1 token (draft token cost)
Per-round cost = drafts * draft_step + verify(N) + snapshot (+ restore + replay on partial
acceptance). Results -> results/mtp_probe.json"""
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_bench import build_model, make_caches, med, gpu_warmup
from mlx_vlm.models.inkling.language import _restore_cache_state, _snapshot_cache_state

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {}
E = int(sys.argv[1]) if len(sys.argv) > 1 else 2
model, cfg = build_model(E, None, None)
gpu_warmup()
for ctx in (2048, 32768):
    r = {}
    for N in (1, 3, 5, 8):
        caches = make_caches(model, ctx)
        ids = mx.random.randint(0, 200000, (1, N))
        mx.eval(ids)
        # verify as the loop does it: hidden only (skip_logits), then argmax from hidden
        r[f"verify_N{N}_3layers_ms"] = med(lambda: model(ids, cache=caches, return_hidden=True, skip_logits=True).hidden_states[-1], warm=3, reps=9)
        r[f"verify_N{N}_plus_argmax_head_ms"] = med(lambda: model.speculative_argmax_from_hidden(model(ids, cache=caches, return_hidden=True, skip_logits=True).hidden_states[-1]), warm=3, reps=9)
        del caches
        mx.clear_cache()
    caches = make_caches(model, ctx)
    r["snapshot_3layers_ms"] = med(lambda: _snapshot_cache_state(caches) and mx.array(0), warm=2, reps=7)
    snap = _snapshot_cache_state(caches)

    def restore():
        _restore_cache_state(caches, snap)
        return caches[0][0].keys

    r["restore_3layers_ms"] = med(restore, warm=2, reps=7)
    r["kv_bytes_3layers"] = sum(c[0].keys.nbytes + c[0].values.nbytes for c in caches)
    res[f"ctx{ctx}"] = r
    print(f"ctx {ctx}:", json.dumps(r, indent=1), flush=True)
    del caches, snap
    mx.clear_cache()

# drafter-like step: dense layer 0 + lm_head on one token (the MTP blocks are dense InklingDecoderLayers)
caches = make_caches(model, 2048)
h = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
mx.eval(h)
lay = model.model.layers[0]
res["draft_step_dense_layer_plus_lm_head_argmax_ms"] = med(lambda: mx.argmax(model.lm_head(model.model.norm(lay(h, cache=caches[0])))[..., :200058], axis=-1), warm=3, reps=9)
res["lm_head_argmax_only_ms"] = med(lambda: mx.argmax(model.lm_head(h)[..., :200058], axis=-1), warm=3, reps=9)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
with open(os.path.join(OUT, "mtp_probe.json"), "w") as f:
    json.dump(res, f, indent=1)
print("wrote mtp_probe.json")
