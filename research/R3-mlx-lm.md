# R3: Inkling-Small on mlx-lm and MLX core (Apple Silicon)

Date: 2026-10-28. Machine: MacBook Pro M5 Max (Metal arch `applegpu_g17s`, NAX
available), 128 GB. Software: mlx 0.32.4 (installed; clone pinned to tag
v0.32.4 under `upstream/mlx`), mlx-lm 0.32.2 installed, mlx-lm clone at
`1f7057ca` (2026-10-28) under `upstream/mlx-lm`, mlx-vlm 0.7.11. Nothing was
pushed, downloaded beyond small text files, or submitted anywhere.

Inputs read: the vendored `inkling_mlx/` package shipped inside
`mlx-community/Inkling-Small-mlx-3bit` (vendored from PipeNetwork/inkling-mlx
commit b6dc0ac, Apache-2.0), the 2-bit/3-bit checkpoint indexes and configs,
the source `thinkingmachines/Inkling-Small` index and config, mlx-lm and MLX
sources at the versions above. Copies are in
`bench/R3/inkling_mlx_vendored/`.

## 1. Summary

1. mlx-lm has no Inkling today, but every building block exists: the
   per-layer `CacheList(ArraysCache, RotatingKVCache|KVCache)` pattern
   (baichuan_m1), the depthwise short conv with cached state (lfm2), the
   sigmoid+bias top-k router with shared experts (deepseek_v3, kimi_k3,
   muse_spark), the `text_config` multimodal wrapper that strips vision/audio
   weights (gemma4, kimi_k3), `quant_predicate` for mixed precision, and a
   loader that quantizes exactly the modules whose `.scales` exist in the
   checkpoint. The vendored `inkling_mlx` already imports mlx-lm's `SwitchGLU`,
   so the shipped mlx-community checkpoints load into an mlx-lm model that
   keeps the same parameter paths (`model.llm.layers.N.attn.wq_du`,
   `model.llm.layers.N.mlp.experts.{gate,up,down}_proj`, ...). A faithful port
   is ~900-1,100 lines (model + cache subclass + tests + sanitize), 3-5 days,
   and gets the server's continuous batching, prompt cache, KV quantization
   and prompt-cache save/load for free (§4).
2. The checkpoints ship attention, embedding and unembedding in bf16 (the
   model cards' "non-expert 8-bit" claim is wrong, §3.1). At 3-bit that is
   5.35 GB of the ~9.2 GB read per decoded token; 8-bit attention and unembed
   cut per-token traffic by 27% (2-bit: 31%) and, measured, halve the
   attention-projection time per layer (0.282 -> 0.139 ms). It is a
   `quant_predicate` and a re-conversion, no model change.
3. The batch-1 decode step is not bandwidth-bound on MLX: the vendored forward
   issues ~100 MLX ops per layer (~4,200 per token) and the measured per-op
   floor inside one dependent graph is 6 us (`rms_norm`), 10 us (unfused
   elementwise), 2.5 us (compiled elementwise), so launch cost alone is
   25-40 ms per token against ~20 ms of weight traffic at 3-bit. Component
   sums (§5.4) put the vendored structure at ~15 tok/s and an op-count-driven
   design at ~27-30 tok/s on this machine, with the bandwidth limit at
   60-80 tok/s. `mx.compile` does not change this (matmul, gather, conv and
   fast ops do not fuse; measured no gain on a whole layer).
4. MLX core 0.32.4 has what is needed: fused SDPA with additive or boolean
   array masks in the decode (sdpa_vector/2-pass) and prefill (steel/NAX)
   kernels, `gather_qmm` for 2/3/4/5/6/8-bit affine experts with a sorted
   "rhs" kernel for prefill, a dedicated depthwise conv1d kernel, NAX GEMMs on
   M5. Two measured gaps matter for Inkling: (a) the relative-position bias
   must be materialized as a `[B,H,L,S]` additive mask, and building it the
   vendored way costs 8.6 ms per layer at L=2048 (4x the 2.2 ms causal SDPA),
   so a distance-indexed bias table in the fused SDPA kernels is the MLX-core
   change with the largest payoff; (b) the 3-bit (and 6-bit) quantized GEMV
   kernels are slower in absolute time than 4-bit (six-expert GLU: 0.223 ms at
   3-bit vs 0.191 ms at 4-bit; lm_head 6-bit 2.76 ms vs 8-bit 2.13 ms), so the
   128 GB "3-bit tier" is the slowest tier until those kernels are tuned.
   Sliding-window SDPA (`window_size`, #4734) and mid-row `sdpa_split`
   kernels for speculative verification (#4723) landed after 0.32.4.
5. MTP: the source checkpoint carries 8 chained dense nextn modules (~2.2B
   params; ~0.3 GB per draft token at 8-bit, ~5% of a target token's weight
   traffic) that the mlx-community conversion dropped. mlx-lm's
   `speculative_generate_step` only accepts a token-fed draft model with a
   trimmable cache, so an MTP path is a separate, small generation-loop PR
   (§4.6); in a dispatch-bound regime it is the largest decode win after the
   port itself. Every cache must stay trimmable (RotatingKVCache is not once
   rotated, ArraysCache never), which shapes the cache design (§4.2).

## 2. mlx-lm reuse map (file:line, clone 1f7057ca)

Model skeleton and conventions
- `mlx_lm/utils.py:200-227` `_get_classes`: imports `mlx_lm.models.<model_type>`;
  `MODEL_REMAPPING` at `utils.py:48-61` (e.g. `gemma4_unified -> gemma4`,
  `kimi_k2 -> deepseek_v3`). Inkling's `model_type` is `inkling_mm_model`, so
  either name the file `inkling_mm_model.py` or add
  `"inkling_mm_model": "inkling"` to the remap.
- `mlx_lm/models/base.py:11-21` `BaseModelArgs.from_dict` (keyword filter);
  `kimi_k3.py:114-127` and `gemma4.py:14-40` show the `text_config` wrapper
  (`ModelArgs(text_config=...)` + `TextArgs.from_dict`).
- `kimi_k3.py:1229-1245` `Model.sanitize` drops vision tower keys and delegates;
  `gemma4.py:54-84` strips `vision_tower/audio_tower/...` and re-prefixes.
  Inkling must drop `model.visual.*`, `model.audio.*` and (if present)
  `model.mtp.*` (`deepseek_v41.py:1014` `_DROP_PREFIXES`).
- Required surface (from `tests/test_models.py:382-420` `model_test_runner`):
  `model.layers`, `model.model_type`, `model(inputs, cache=None, input_embeddings=None)`
  returning `[B, L, vocab]` in the parameter dtype, works with
  `make_prompt_cache(model)`, batch size 2, and `copy.deepcopy`.
- `utils.py:1246` `does_model_support_input_embeddings` (signature check) is
  what the generate path uses for multimodal embeddings.

Experts
- `mlx_lm/models/switch_layers.py:182-220` `SwitchGLU` (3 `gather_qmm` calls,
  sorts when `indices.size >= 64`, `_gather_sort` at :11-23);
  `QuantizedSwitchLinear.__call__` :93-109 is the `mx.gather_qmm` call
  (`rhs_indices`, `transpose=True`, `sorted_indices`).
- Router with sigmoid + correction bias + shared experts: `deepseek_v3.py:195-229`
  `group_expert_select`, `deepseek_v3.py:230-290` (`MoEGate`, `DeepseekV3MoE`:
  shared experts as one dense MLP of width `moe_intermediate * n_shared`,
  which is also the faster layout here, §5.3); `kimi_k3.py:26-63` fp32
  variant; `muse_spark.py:273-354` (Apple-authored style reference, fp32
  router weights, `expert_routing`).
- Per-module quantization: `utils.py:496-516` quantizes exactly the modules
  with `{path}.scales` in the weights and infers bits/group from shapes
  (`utils.py:382-407` `infer_quant_config`); `quant_predicate` property
  (`kimi_k3.py:1084-1094`: router at 8-bit, some paths `False`) drives
  `mlx_lm.convert`; mixed recipes in `convert.py:20-80`
  (`mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`).

Attention, sliding windows, masks
- `base.py:24-42` `create_causal_mask(N, offset, window_size, left/right_padding)`
  (boolean), `base.py:45-55` `create_attention_mask` (returns `"causal"`,
  `None` for N==1, or an array), `base.py:110-139` `scaled_dot_product_attention`
  wrapper that switches to `quantized_scaled_dot_product_attention`
  (`base.py:64-107`, `quantized_matmul` on the packed cache) when the cache has
  `bits`.
- `gemma3_text.py:195-210` builds one global and one sliding mask per forward
  and `make_cache` :247-257 mixes `KVCache()` and `RotatingKVCache(max_size=window)`;
  `baichuan_m1.py` `make_cache` (`CacheList(ArraysCache(2), RotatingKVCache|KVCache)`)
  and mask creation :194-212 from the first global / first SWA layer caches.
- `cache.py:423-593` `RotatingKVCache` (in-place rotation for L==1 at :482-520,
  `make_mask` :556-580 returns `None` for decode when `max_size == window`,
  `is_trimmable` only while `offset < max_size` :544, `to_quantized` NYI :553);
  `cache.py:1101+` `BatchRotatingKVCache` (server batching of SWA layers);
  `cache.py:339-420` `KVCache` (`step=256` growth, `trim`, `to_quantized`);
  `cache.py:251-336` `QuantizedKVCache`; `cache.py:596-724` `ArraysCache`
  (conv states; `lengths`/`left_padding` for batched prompts; `merge`/`extend`
  /`filter`/`extract`; no `trim`); `cache.py:803-868` `CacheList`;
  `cache.py:43-112` `save_prompt_cache`/`load_prompt_cache` (serializes any
  `state` tuple; classes are looked up in `cache.py` globals, so a model-local
  cache class like `deepseek_v41.py:267` `LayerCache` cannot be saved/loaded).
- Short conv with cached state: `lfm2.py:120-178` `ShortConv` (`nn.Conv1d`
  groups=hidden, state = last `L-1` inputs in `ArraysCache(size=1)`,
  `cache.lengths` handling for right-padded batches, `mx.pad` then
  `padding=0` so the depthwise fast path is taken); `base.py:58-61`
  `create_ssm_mask`.

Generation and serving
- `generate.py:324-494` `generate_step`: chunked prefill (`prefill_step_size`
  default 2048, `mx.eval` of cache state + `mx.clear_cache()` per chunk), KV
  quantization after `quantized_kv_start` (:316-322, :410-430), one-step-ahead
  `mx.async_eval` pipelining (:477-493). No `mx.compile` of the model.
- `generate.py:496-676` `speculative_generate_step`: separate token-fed
  `draft_model` with its own cache (`prompt_cache[len(model.layers):]`),
  verification with `n_predict = num_draft + 1` rows, rewind via
  `trim_prompt_cache` (:608-610) which requires every cache to be
  `is_trimmable()` (:553-558).
- `generate.py:1145-1343` `PromptProcessingBatch`, :1351-1593 `GenerationBatch`
  (`_step` :1438-1497 batched decode with `async_eval`), :1595+ `BatchGenerator`.
  `server.py:474-479`: a model is batchable iff every cache from `make_cache`
  has `merge` and no draft model is loaded; `server.py:766-771` also requires
  `kv_bits is None`. `cache.py:1599+` `LRUPromptCache` and `PromptTrie` back the
  server's prompt cache; nothing in mlx-lm is called a "bundled drafter"
  (`--draft-model` is a separate model, `server.py:407-410`).
- Memory: `generate.py:249-283` `wired_limit` context manager and
  `utils.py:74` `maybe_set_recommended_wired_limit`.

Tests and contribution rules
- `CONTRIBUTING.md`: file name == `model_type`, add a test in
  `tests/test_models.py`, `uvx pre-commit run --all`, PR template
  `.github/PULL_REQUEST_TEMPLATE/new_model.md` (requires an
  `mlx_lm.generate` transcript and, if available, a transformers reference
  transcript; AI-written PR text is prohibited and AI use must be disclosed).
- `tests/test_models.py:633-658` (`test_lfm2`), :2447-2477 (`test_deepseek_v41`)
  show the expected tiny-config tests; `tests/test_prompt_cache.py` covers
  save/load/trim; LoRA keys are discovered generically (`tuner/utils.py:90-120`).
- How similar PRs landed: Kimi K3 (#1626, 2026-09-01, +1574 lines, 9 files,
  5 review rounds by zcbenz: "prefer explicit code", move tokenizer/tool
  logic into helpers, drop a decode kernel that "had not much to gain");
  DeepSeek-V4.1 (#1895, 2026-09-29, +1301 lines, 5 files, one review);
  Muse Spark (#1976, Apple author, +2911 lines, 23 files). Model files of
  1-1.5k lines with a custom `mx.fast.metal_kernel` are accepted when they
  carry tests.
- Installed mlx-lm 0.32.2 already ships gemma4, kimi_k3, muse_spark,
  deepseek_v41, baichuan_m1, lfm2 and all cache classes named above
  (`BatchGenerator` is HEAD-only).

## 3. The shipped checkpoints and the vendored package

### 3.1 What is actually in `mlx-community/Inkling-Small-mlx-{2,3,4}bit`

Weight map (2-bit index, 1470 keys, 18 shards, 88.39 GB):
- `model.llm.embed.weight`, `model.llm.embed_norm.weight`, `model.llm.norm.weight`,
  `model.llm.unembed.weight` (no `.scales`: bf16).
- `model.llm.layers.N.attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud}.weight` (bf16, no
  scales), `attn.{q_norm,k_norm}.weight`, `attn.{k_sconv,v_sconv}.weight`
  (`[C,4,1]` fp32), `attn.rel_logits_proj.proj` (`[16, 512]` on the 35 sliding
  layers, `[16, 1024]` on the 7 global ones), `attn_norm`, `mlp_norm`,
  `attn_sconv`, `mlp_sconv`.
- Layers 0-1: dense `mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`
  (quantized), `mlp.global_scale`.
- Layers 2-41: `mlp.experts.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`
  (`[256, 2048, 4096]`-shaped, quantized), `mlp.shared_experts.*` (`[2, ...]`,
  quantized), `mlp.gate.{weight [258,4096], bias [256], global_scale}` (fp32).
- `model.visual.layers.linear_{0..3}` (3 of 4 quantized), `model.audio.encoder`
  (quantized embedding), their norms. No `model.mtp.*`.

Size check: 257.7B routed weights at 2.5 bits/weight (2-bit + bf16 scale and
bias per 64) = 80.5 GB, plus 3.7 GB bf16 attention, 3.3 GB bf16 embed/unembed,
0.6 GB shared experts: 88.3 GB, matching the index. So the recipe is
`experts_only` exactly as `config.json` says; the README's "non-expert 8-bit"
is not what was shipped. The `mlx_lm.server` snippets on the cards cannot work
because no `mlx_lm/models/inkling*.py` exists.

Per-token weight traffic (decode, B=1), derived from the layout:

| experts | routed+shared+dense | attention | unembed | router | total as shipped | total with 8-bit attn+unembed |
|---|---|---|---|---|---|---|
| 2-bit | 2.64 GB | 3.70 GB (bf16) | 1.65 GB (bf16) | 0.17 | 8.16 GB | 5.66 GB (-31%) |
| 3-bit | 3.70 GB | 3.70 | 1.65 | 0.17 | 9.22 GB | 6.72 GB (-27%) |
| 4-bit | 4.75 GB | 3.70 | 1.65 | 0.17 | 10.27 GB | 7.77 GB (-24%) |

At the 450-520 GB/s this machine sustains on streaming reads (§5.1) the pure
bandwidth ceilings are ~55 / ~75 tok/s (3-bit, as shipped / 8-bit attention)
and ~60 / ~90 tok/s at 2-bit. Real decode sits far below these because of op
count (§5.4).

### 3.2 What the vendored `inkling_mlx` does (and costs)

- Full-history `KVCache` on every layer, including the 35 sliding ones
  (`cache.py:30-49` of the package); sliding is enforced only by the mask, so
  memory grows with context on all 42 layers and the sliding layers read the
  whole history at decode.
- Relative bias built per layer per step as
  `rel @ proj -> swapaxes -> clip(distance) -> broadcast -> take_along_axis -> where`
  on a `[B, H, Lq, Lkv]` array (`attention.py:35-45`), then added to a
  `where(allowed, 0, -1e30)` causal/sliding mask (`attention.py:125-131`), and
  passed to `mx.fast.scaled_dot_product_attention` (`attention.py:118-120`).
- Short conv in fp32 with a `[B,3,C]` state and `mx.conv1d(groups=C, padding=0)`
  (`common.py:24-63`); without a cache it calls `conv1d(padding=k-1)`, which
  misses the depthwise fast path (4x slower, §5.3).
- Router in fp32 with `argpartition`, logsigmoid softmax over the 6 routed
  plus 2 shared logits, `* route_scale * global_scale` (`moe.py:40-85`);
  routed experts through mlx-lm's `SwitchGLU`; shared experts through a second
  `SwitchGLU(…, 2)` with constant indices (`moe.py:88-117`), which is slower
  than a dense GLU at every bit width (§5.3).
- `load.py` re-quantizes by recipe name instead of by the presence of
  `.scales`; with the shipped `experts_only` files that happens to match.
- Conversion (`convert.py`): HF `w13_weight [E, 2I, H]` is gate/up
  **interleaved row-wise** (gate = rows 0::2, up = rows 1::2; a contiguous
  split scrambles them), `*_sconv.weight [C,1,K] -> [C,K,1]`, `model.mtp.*`
  dropped. The logits are `unembed(h / logits_mup_width_multiplier=16)[:, :200058]`.
- mlx-vlm (`upstream/mlx-vlm/mlx_vlm/models/inkling/language.py`) differs in
  that it fuses q/k/v/r into one projection, writes the banded additive mask
  with a custom Metal kernel (`inkling_banded_mask`, :70-136), slices the K/V
  of sliding layers to the window at runtime (:445-456) while still keeping a
  full `KVCache`, and has custom kernels for sconv decode, routing and the
  down-projection combine (:291, :595, :654). Its cache is
  `CacheList(KVCache(), ArraysCache(4))` per layer (:993).

## 4. mlx-lm Inkling design

### 4.1 Files, registration, weights

- `mlx_lm/models/inkling.py` (~550-700 lines) with `TextArgs`,
  `ModelArgs(text_config=...)` (`kimi_k3.py:114` pattern), `Model` exposing
  `model.llm` so parameter paths match the checkpoints exactly
  (`model.llm.layers.N...`, `model.llm.unembed`), plus
  `MODEL_REMAPPING["inkling_mm_model"] = "inkling"` in `utils.py`, or the file
  named `inkling_mm_model.py`.
- `sanitize`: drop `model.visual.*`, `model.audio.*`, `model.mtp.*`; accept both
  the mlx-community layout (already split gate/up, `[C,K,1]` convs) and the HF
  source layout (de-interleave `w13`, swap conv axes, split `shared_w13`,
  rename `w2_md -> down_proj`), so `mlx_lm.convert --hf-path thinkingmachines/Inkling-Small`
  works directly. Same shape of work as `deepseek_v3.py:383-480`.
- `quant_predicate` (property): `mlp.gate` (router) and norms/sconv/`proj`
  never quantized; experts at the requested bits; attention projections and
  `unembed` at 8-bit by default (override via `--quant-predicate`); `embed`
  8-bit (`nn.QuantizedEmbedding`). The loader needs no change: it quantizes
  whatever has `.scales`.
- `cast_predicate`: keep `gate.*`, `global_scale`, `*_sconv.weight`, norms in
  fp32 (`kimi_k3.py:1073-1082` pattern).

### 4.2 Cache

`make_cache()` returns, per layer, `CacheList(kv, ArraysCache(size=4))` with
`kv = KVCache()` on the 7 global layers and a sliding cache on the 35 others.
Two options for the sliding cache:

- `RotatingKVCache(max_size=512)` (zero-copy decode, `BatchRotatingKVCache`
  for the server). The relative bias then needs a per-slot distance: slot j
  holds position `offset-1-((idx-1-j) mod 512)`, so the bias is a gather with
  a rolled index. The index is identical for all 35 sliding layers in a step,
  so compute it once per forward and reuse. Downsides: not trimmable after
  rotation (`cache.py:544`), so no speculative decoding past 512 tokens, and
  `to_quantized` is NYI.
- A new `ShiftingWindowCache` (temporal order kept by `concatenate([k[..., 1:, :], new])`,
  ~2 MB copied per layer per token, 70 MB per token in total, i.e. ~0.15 ms
  at 500 GB/s). The bias then is simply the reversed `rel @ proj` row
  (`mx.flip` on the last axis, no gather at all), the cache is trimmable
  (keep up to 512 + max_draft rows), quantizable, and batchable through a
  `merge` that pads on the left. Recommended: it removes 8-10 ops per layer
  and keeps speculative decoding possible.

The 4 conv states are `[B, 3, C]` arrays (k: 1024, v: 1024, post-attn: 4096,
post-MLP: 4096 channels). For speculative decoding they must be rewindable:
keep the last `3 + num_draft` inputs instead of 3 and make `trim(n)` drop
the newest `n` rows (ArraysCache has no `trim`; a small subclass adds it).
`lfm2.py:165-170` shows the `cache.lengths` handling needed for batched
right-padded prompts.

Masks: global layers use `create_attention_mask(h, cache_global)` semantics
(`"causal"` / `None` / boolean array with `left_padding` for batches) combined
with the additive bias; sliding layers always need an additive array when
`L > 1` (bias) and nothing when `L == 1` with the window cache full
(`RotatingKVCache.make_mask` returns `None`, `cache.py:566-569`).

### 4.3 Layer forward (op-count driven)

Per layer at decode, target ~45 ops instead of the vendored ~100:

1. `rms_norm` -> one fused `qkvr` projection (`[6656, 4096]`, one GEMV, like
   mlx-vlm) -> split views.
2. One short conv over the concatenated k|v channels (2048), state `[B,3,2048]`:
   `concatenate(state, x)` -> `conv1d(groups=2048, padding=0)` -> slice state.
   Keep the conv in the model dtype unless validation shows drift (the
   reference keeps it fp32; measured cost is the same either way, §5.3).
3. Per-head `rms_norm` on q and k (two ops), transpose.
4. Cache update (`update_and_fetch`), bias:
   - sliding: `flip(rel @ proj)` sliced to the current window length;
   - global: `concatenate([zeros(T-1024), flip(rel @ proj)])` (one concat),
     with the log-scaling factor `tau` applied to q and bias only when
     `offset >= 128000`.
5. `mx.fast.scaled_dot_product_attention(q, k, v, scale=1/128, mask=bias)`
   (fused vector kernel, additive mask strided over heads; §5.2).
6. `wo` GEMV -> post-attention short conv -> residual.
7. `rms_norm` -> router (`mx.compile`d function: fp32 GEMV, sigmoid, bias,
   `argpartition`, logsigmoid-softmax, scale; the compiled elementwise chain
   runs at 2.5 us/op instead of ~10) -> routed experts through `SwitchGLU`
   (3 `gather_qmm`) -> **shared experts as one dense GLU** with the two
   experts concatenated (`[4096, 4096]` gate/up, `[4096, 4096]` down,
   `quantized_matmul`), the deepseek_v3 layout, which measured faster than
   `gather_qmm` over 2 experts at decode (0.094-0.101 vs 0.117-0.131 ms) and
   avoids the unsorted `gather_qmv` path at prefill. Weighted sum (routed
   weights and shared gammas come from the same softmax) -> post-MLP short
   conv -> residual. (Folding the shared experts into the routed call as
   experts 256-257 saves ~3 ops but measured as a wash at decode; keep it as
   an option, not the default.)
8. Prefill (`L > 1`): the banded bias `[B, H, L, S]` is produced by a
   `mx.fast.metal_kernel` (port of mlx-vlm's `inkling_banded_mask`, ~60 lines)
   rather than the matmul+gather+where chain (8.6 ms vs ~0.5 ms expected per
   layer at L=2048). Use `prefill_step_size` chunks of 2048; the mask for a
   sliding layer is at most `[32, 2048, 2559]` bf16 (335 MB, transient).
   Dense layers 0-1 use a plain GLU (`gemma3_text.py:114-122` style).

Everything above is expressible with mlx-lm conventions; only the banded-mask
kernel is custom, and kimi_k3/deepseek_v41/plamo3/rwkv7 set the precedent.

### 4.4 What comes for free

- Chunked prefill, prompt-cache save/load (`CacheList` and `ArraysCache`
  serialize through `state`), prompt-cache trie/LRU in the server, wired-limit
  handling, `mlx_lm.generate` / `chat` / `server` / `lm_eval` integration,
  LoRA, `mlx_lm.convert` with mixed recipes, `--kv-bits` on the global layers
  (`KVCache.to_quantized`; the sliding cache needs a `to_quantized` only if
  KV quantization is wanted there, which is pointless at 512 keys).
- Server continuous batching: `CacheList.merge` exists, `BatchKVCache` and
  `BatchRotatingKVCache` exist, `ArraysCache.merge/extend/filter/extract`
  exist; the shifting-window cache needs the same four methods (~60 lines).
- Distributed/pipeline loading is not needed for a 12B-active model that fits
  in one Mac Studio.

### 4.5 Effort and acceptance

- Model file 550-700 lines, cache subclass 80-150 lines, tests ~120 lines
  (tiny-config `model_test_runner` run, cache trim/merge round trips, HF-layout
  vs mlx-layout sanitize, bias equivalence against the vendored reference on
  random weights), convert/quant predicate 40 lines, README/ACKNOWLEDGMENTS
  entries. Roughly 900-1,100 lines; 3-5 days plus a conversion run from the
  532 GB source (the mlx-community files can be loaded directly for the
  `mlx_lm.generate` transcript the PR template demands).
- Maintainers (awni, angeloskath, zcbenz) ask for: explicit code over
  cleverness, no dead paths, tests that fail on main, benchmarks when
  performance is touched, and disclosure of AI use; custom Metal kernels are
  accepted when they are small and justified (K3 review dropped one that
  "had not much to gain").
- Validation plan: greedy transcripts versus the vendored package on the same
  3-bit checkpoint (identical tokens for ~200 steps), then versus the HF
  reference if a CUDA box is available; perplexity on a short text; prompt
  cache save/load; server batch of 4.

### 4.6 MTP as a drafter

The source checkpoint's `model.mtp.layers.{0..7}` are DeepSeek-style nextn
modules: `embed_norm(embed(next_token)) ++ hidden_norm(h) -> input_proj
[8192 -> 4096] -> one dense Inkling transformer block (attention with sconv
and relative bias, dense MLP 4096->16384->4096 with global_scale) -> shared
unembed`; `mtp_config.local_layer_ids = [0,2,4,5,6,7]` says which of the 8
modules use sliding attention. ~279M params per module, ~2.2B in total
(4.5 GB bf16, ~2.4 GB at 8-bit, ~1.25 GB at 4-bit), so drafting k tokens
costs about k x 5% of a target forward in weight traffic plus ~25 ops per
module. `mlx-community` dropped these weights; a converter from the HF source
must keep them.

mlx-lm's `speculative_generate_step` cannot host this directly: its draft
model is token-fed with its own cache. The clean way is a
`mtp_generate_step` (or a `draft_from_hidden` hook) that (1) runs the target
on `y` returning the last hidden state, (2) runs modules 0..k-1 chained on
`(h, embed(t_next))` with their own `CacheList` per module, (3) verifies with
the target on `k+1` rows (MLX's `sdpa_split` kernels for 2-8 query rows,
#4723, target exactly this; not in 0.32.4) and (4) rewinds via
`trim_prompt_cache`, which requires the trimmable caches of §4.2. Expected
gain at batch 1: with acceptance ~70% for 3-4 drafts, 1.8-2.5x fewer target
forwards per token; since the decode step is dispatch-bound the gain is in
tokens per forward, not per byte. This is a separate PR after the base model.

## 5. Micro-benchmarks (bench/R3; raw in bench/R3/results/)

Commands: `source ~/.venvs/inkling/bin/activate; cd bench/R3; RUN=n ./run_all.sh`
(sequential: `bw.py`, `bench_sdpa.py`, `bench_misc.py`, `bench_moe.py`,
`bench_layer.py`; `MAXLOAD=10 RUN=n ./wait_and_run.sh` gates on the 1-minute
load). Per-script logs are `results/<script>_run<n>.log`; every row is also
appended to `results/{bw,sdpa,misc,moe,layer}.jsonl` with a `_sysinfo`
header per run. Columns: `r1` = one synchronous dispatch per op (what a
serial decode loop pays), `r8`/`r4` = eight/four independent copies evaluated
together (per-op time with launch overhead amortized). Medians over 30
iterations, best in parentheses.

Contention: this machine was shared with other jobs for the whole session.
Run 1 (21:55) ran under a llama.cpp build and a Ray benchmark (1-minute load
60-95); run 2 was contaminated by two of my own chains overlapping (moved to
`results/contaminated_run2/`; in the jsonl files its rows sit between the
second and third `_sysinfo` header); run 3 (22:17, load ~9) ran while another
agent's `layer_bench.py` shared the GPU. Short kernels agree within ~20%
across runs; the whole-layer emulation (long graphs) is the most affected.
Numbers below are run 3 unless marked `[run1]`; best-of across runs is the
cleanest estimate. An idle re-run is `RUN=4 ./run_all.sh`.

### 5.1 Memory bandwidth (bw.py)

| op | r1 median | r4/r8 median (best) |
|---|---|---|
| fp16 `a+1`, 1 GiB (read+write) | 460 GB/s | 475 (484) |
| fp16 `a*b`, 1 GiB (2 reads + write) | 461 GB/s | 484 (590); `[run1]` 4 GiB best 609 |
| fp16 `sum(a)` 1 GiB (read) | 472 GB/s | 521 (548) |
| bf16 GEMV 32768x16384 (1 GiB) | 441 GB/s | 464 (489) |
| bf16 GEMV 201024x4096 (1.65 GB, lm_head) | 458 GB/s | 500 (520); `[run1]` 557 |

M5 Max spec is ~614 GB/s; large streaming ops sustain 450-520 GB/s here
(best observed 609 GB/s), 32-90 MB reads (one layer's attention weights, six
experts) sustain 230-450 GB/s depending on the kernel.

### 5.2 Fused SDPA (bench_sdpa.py), 32q/8kv heads x 128, bf16

Decode (1 query), r8 medians per call:

| context | mask=None | additive bf16 `[1,32,1,T]` | bool `[1,1,1,T]` | additive broadcast `[1,1,1,T]` | vendored bias build + sdpa | unfused reference |
|---|---|---|---|---|---|---|
| 512 | 0.040 ms | 0.044 | 0.044 | 0.047 | 0.096 | 0.084 |
| 4096 | 0.097 | 0.130 | 0.093 | 0.130 | 0.206 | 0.153 |
| 32768 | 0.334 | 0.477 (`[run1]` 0.326) | 0.311 | 0.641 | 0.804 (`[run1]` 0.491) | 1.404 |

Reading: at 512 keys the kernel is launch-bound (2 MB of K/V, ~40 us); the
per-head additive mask costs 15-40% at 32K (64 B per key versus 4 KB of
K/V); the boolean window mask lets the kernel skip blocks; the vendored bias
construction adds 0.17-0.33 ms per global layer at 32K. The fused kernel is
3-4x the unfused path at 32K. 42-layer attention per token: 2.3 ms at 4K
context and 4.7 ms at 32K with an additive mask, 2.8 / 7.0 ms with the
vendored bias build.

Prefill (T=2048 queries vs 2048 keys), per layer:

| mask | ms (r4 median, best) | dense-equivalent TFLOPS |
|---|---|---|
| `"causal"` string | 2.21 (1.94) | 31.0 |
| bool causal array | 5.29 (4.85) | 13.0 |
| bool window-512 array | 5.04 (4.83) | 13.6 |
| additive bf16 `[1,32,T,T]` | 4.77 (4.29) | 14.4 |
| vendored bias build only (matmul+gather+where) | **8.58 (8.08)** | - |

With the vendored construction a sliding layer pays 8.6 + 4.8 = 13.4 ms of
attention per 2048-token chunk against 2.2 ms for a bias-free causal call;
over 42 layers 0.56 s versus 0.09 s per chunk. A Metal mask kernel removes
most of the 8.6 ms; a bias-aware SDPA would remove the array altogether and
recover the causal-kernel speed (array masks halve the NAX kernel's rate).

### 5.3 Everything else (bench_misc.py, bench_moe.py)

| op (decode unless stated) | r1 median | r8 median (best) | GB/s at r8 |
|---|---|---|---|
| lm_head bf16 201024x4096 | 3.60 ms | 3.29 (3.10) | 500 |
| lm_head 8-bit g64 | 1.98 | 2.13 (1.65) | 411 (533 best) |
| lm_head 6-bit g64 | 4.60 | **2.76** (2.45) | 242 |
| lm_head 4-bit g64 | 1.56 | 1.44 (1.23) | 321 |
| attention 5 projections bf16 (88 MB) | 0.573 | 0.282 (0.223) | 313 |
| attention 5 projections 8-bit (47 MB) | 0.480 | **0.139** (0.114) | 338 |
| attention 5 projections 4-bit | 0.471 | 0.113 (0.092) | 219 |
| dense GLU 4096->16384->4096 8-bit / 4-bit | 0.79 / 0.59 | 0.436 / 0.280 | 491 / 404 |
| gather_qmm up 4096->2048, 6 experts, 2/3/4-bit | 0.23 / 0.31 / 0.28 | 0.055 / **0.096** / 0.084 | 287 / 229 / 337 |
| SwitchGLU decode (3 gather_qmm), 2/3/4-bit | 0.31 / 0.45 / 0.47 | 0.132 / **0.223** / 0.191 | 358 / 296 / 446 |
| SwitchGLU decode, fused gate+up, 2/3/4-bit | 0.32 / 0.46 / 0.46 | 0.121 / 0.202 / 0.191 | 390 / 328 / 444 |
| shared experts via gather_qmm (E=2), 2/3/4-bit | 0.34 / 0.31 / 0.31 | 0.131 / 0.117 / 0.100 | - |
| shared experts as dense GLU (2I=4096), 2/3/4-bit | 0.28 / 0.28 / 0.29 | **0.094 / 0.101 / 0.087** | - |
| router fp32 (GEMV+sigmoid+argpartition+softmax) | 0.29 | 0.052 (0.044) | - |
| SwitchGLU prefill T=512 sorted, 2/3/4-bit | 11.0 / 22.3 / 22.9 ms per call | - | 21-45 us/token, 7-14 TFLOPS |
| SwitchGLU prefill T=512 **unsorted** | 185 / 190 / 193 ms | - | 9-17x slower than sorted |
| SwitchGLU prefill T=2048 sorted, 2/3/4-bit | 70.0 / 76.4 / 76.4 ms per call | - | 34-37 us/token, 8.1-8.8 TFLOPS |
| sconv decode T=1 conv1d groups=4096 (fp32 / bf16) | 0.34 / 0.27 | 0.059 / 0.053 (0.033-0.039) | launch-bound |
| sconv decode elementwise 4-tap (fp32 / bf16) | 0.37 / 0.30 | 0.067 / 0.068 | launch-bound |
| sconv prefill T=2048 conv1d `padding=0` on padded input (fp32 / bf16) | 0.83 / 0.68 | 0.563 / 0.408 (r4) | 119 / 82 |
| sconv prefill T=2048 conv1d `padding=3` (vendored no-cache path) | 2.48 / 1.99 | 2.29 / 1.81 (r4) | 29 / 19 |
| rms_norm 4096, residual add (8-op graph) | 0.26-0.27 | 0.036-0.042 | - |
| **per-op floor, 200 dependent ops in one graph**: rms_norm / unfused elementwise / compiled elementwise / 4096x4096 bf16 matvec | 6.0 / 11.6 / 2.5 / 50.6 us per op | (best 5.4 / 9.5 / 2.2 / 48.8) | - |

Observations: (1) Six-expert `gather_qmm` sustains 290-450 GB/s; **3-bit is
slower in absolute time than 4-bit** (0.223 vs 0.191 ms per GLU; up-proj
0.096 vs 0.084 ms) and 6-bit `quantized_matmul` is slower than 8-bit
(2.76 vs 2.13 ms on the lm_head): the non-power-of-two unpack paths in the
0.32.4 QMV kernels are the bottleneck, not bytes. 2-bit is the fastest tier
(0.132 ms). A fused gate+up tensor gains 0-10%. (2) The shared experts are
faster as one dense GLU than as a 2-expert gather at every bit width, and
the gather form falls onto the unsorted `gather_qmv` path at prefill.
(3) The sorted expert path is 9-17x faster than unsorted at T=512 (mlx-lm
sorts automatically above 64 indices); prefill expert GEMMs run at 7-14
TFLOPS versus 31 TFLOPS for the dense bf16 attention kernel, so prefill is
expert-GEMM-bound. (4) Depthwise conv1d is launch-bound at decode (33-60 us
whichever way it is written) and 4x faster at prefill when called with
`padding=0` on an explicitly padded input (dedicated kernel) than with
`padding=3` (implicit GEMM), which the vendored no-cache code does. (5) The
per-op floor inside one dependent graph is 6 us for a fast op and ~10 us for
an unfused elementwise op; `mx.compile` brings elementwise chains to 2.5 us
per op but cannot touch matmul/gather/conv/fast ops.

### 5.4 Whole-layer emulation (bench_layer.py) and per-token budget

Measured (run 3, GPU shared with another job's job; 42 layers + unembed,
context 4096, 3-bit experts):

| configuration | ms per token | tok/s |
|---|---|---|
| attention bf16, vendored-style bias | 214 (best 188) | 4.7 (5.3) |
| attention bf16, vendored-style bias, `mx.compile` per layer | 238 | 4.2 |
| attention 8-bit, vendored-style bias | 154 (best 140) | 6.5 (7.2) |
| attention 8-bit, cheap bias / compiled | 214 / 195 | 4.7 / 5.1 |
| one sliding layer, r8 (bf16 / 8-bit attention) | 4.4 / 5.1 (best 4.2 / 4.0) | - |
| prefill, one sliding layer at T=2048 (bf16 / 8-bit attention) | 266 / 224 | 42 layers: 185 / 221 tok/s |

These absolute numbers are a contended lower bound (the same layer's
components sum to ~1.5 ms, not 4.4 ms, and the emulation's shared experts use
the unsorted gather path at prefill), but two conclusions are robust:
`mx.compile` does not help a layer of this shape, and 8-bit attention is
worth ~30% by itself. Re-run on an idle machine with
`python bench_layer.py --experts 2 3 4 --attn bf16 8`.

Per-token budget from the component measurements (r8 medians, run 3,
context 4K; per-op launch cost is inside the measured components):

| design | sliding layer | global layer | 42 layers + dense + head | tok/s |
|---|---|---|---|---|
| vendored structure, 3-bit experts, bf16 attention/unembed (~100 ops/layer) | ~1.45 ms | ~1.6 ms | ~66 ms | ~15 |
| §4.3 design, 3-bit experts, 8-bit attention/unembed (~45 ops/layer) | ~0.80 | ~0.90 | ~37 ms | ~27 |
| same with 4-bit experts | ~0.75 | ~0.85 | ~35 ms | ~29 |
| same with 2-bit experts | ~0.71 | ~0.81 | ~33 ms | ~30 |
| bandwidth limit, 3-bit / 2-bit with 8-bit attention (6.7 / 5.7 GB at 450 GB/s) | - | - | 15 / 13 ms | 67 / 79 |

(Sliding layer, §4.3 design at 3-bit: fused projections 0.139 + k|v conv
0.055 + norms/updates/bias ~0.06 + SDPA 0.044 + post-attn conv 0.055 + router
0.052 + routed GLU 0.223 + shared GLU 0.101 + post-MLP conv 0.055 + residuals
~0.02 = ~0.80 ms. Global layer adds the 4K-context SDPA 0.130 instead of
0.044. Head: 8-bit unembed 2.13 ms, bf16 3.29 ms.)

The gap between ~30 tok/s and the ~70 tok/s bandwidth limit is launch
overhead and small-kernel inefficiency (six 4.7 MB expert matrices per call,
2 MB of K/V per sliding layer, four 4-tap convs per layer). Closing it is
kernel work (MLX core opportunities #3, #5 and #7 in §7) and MTP drafting
(tokens per forward), not quantization.

## 6. MLX core (0.32.4) capabilities and gaps, with versions

- `mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask, sinks, force_fused)`:
  mask is `None`, `"causal"` (lower-right aligned) or an array (bool or
  additive, up to 4-D, broadcast to `[B, N_q, T_q, T_kv]`, `fast.cpp:906-909`);
  bool masks are native in the kernels (`supports_bool_mask`,
  `scaled_dot_product_attention.cpp:1466`). Fused decode kernels
  (`sdpa_vector`, `sdpa_vector_2pass`) run for `T_q <= 8` with head dims
  {64,96,128,192,256,512}, GQA native, 2-pass chosen for `T_kv >= 1024` on
  desktop-class chips (`eval_gpu`, :1577-1579); masks are read with strides
  (`kernels/sdpa_vector.h:28-35,104-122`), so a `[1,1,1,T]` mask is not
  materialized per head. Prefill kernels (steel; NAX variants on M5 for
  D in {64,96,128,256,512}) run for `T_q > 8` with array masks; the unfused
  path is used for D=192/256 prefill and when training with an array mask.
  No sliding-window argument in 0.32.4 (PR #4734 `window_size`, merged
  2026-10-23, is post-0.32.4); no relative-bias/table argument; `sinks`
  supported. The 2-8 query-row `sdpa_split` kernels for speculative
  verification are PR #4723 (2026-10-25, post-0.32.4). GQA 2-pass tuning:
  #4380 (2026-08-28), #4596 (2026-10-03).
- No `mx.fast.quantized_scaled_dot_product_attention`; mlx-lm's quantized KV
  path is two `quantized_matmul` calls plus softmax (`base.py:64-107`).
- `mx.gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose,
  group_size, bits, mode, global_scale, sorted_indices)`: affine bits
  {1,2,3,4,5,6,8}, groups {32,64,128}; `mxfp4`/`mxfp8`/`nvfp4` (groups 32/16).
  Dispatch (`backend/metal/quantized.cpp:1980-2080`): `gather_qmm_rhs` when
  `M == 1 and B >= 16 and sorted and B/E >= 4` (prefill with sorted tokens),
  `gather_qmm` (NAX on M5) when `M >= qmv batch limit` (33/25/13 rows by size
  on gen-17 non-desktop parts, `:89-110`), otherwise `gather_qmv`. Sorted
  `gather_qmm_rhs` landed through #4056 (2026-08-08) with the ragged-K fix
  #4009 (2026-09-07) and the non-transposed NAX variant #4684 (2026-10-14,
  **in** 0.32.4); staged split-k NAX `qmm_t` (#4682, 2026-10-20) is not.
  Measured here: 3-bit and 6-bit QMV/gather_qmv are slower in time than
  4-bit and 8-bit (§5.3).
- `mx.conv1d(groups=C)`: dedicated `depthwise_conv_1d` kernel when
  `groups == C == O`, `padding == 0`, stride/dilation 1 (`conv.cpp:1560-1563`;
  kernel from #2567, 2025-09-08, large-variant fix #3289, 2026-03-23);
  otherwise the grouped implicit-GEMM path (4x slower at these shapes).
- `mx.fast.metal_kernel(name, input_names, output_names, source, header,
  ensure_row_contiguous, atomic_outputs, compile_options)`: JIT Metal, used by
  five mlx-lm model files today.
- `mx.compile(fun, inputs, outputs, shapeless)`: fuses elementwise chains
  only (2.5 vs 10 us per op measured); a compiled function containing
  `gather_qmm`, `conv1d`, `rms_norm`, SDPA just records those as primitives.
- NAX (M5 "neural accelerators" in the GPU) is enabled on this chip
  (`device.cpp:978-996`: macOS >= 26.2 and GPU gen >= 17) and backs the
  quantized GEMM (`quantized_nax.metal`), the prefill attention (31 TFLOPS
  dense-equivalent with the causal string mask) and the sorted expert kernels
  (7-14 TFLOPS at these expert shapes).

## 7. Opportunities, ranked

| # | contribution | target | est. gain | effort | risk |
|---|---|---|---|---|---|
| 1 | **First-class `mlx_lm/models/inkling.py`** with trimmable window cache, fused qkvr, k|v conv, flip-based decode bias, dense shared-expert GLU, Metal banded-mask kernel for prefill, `quant_predicate` (attention/unembed 8-bit) and HF-source `sanitize`; registers `inkling_mm_model`. | mlx-lm | makes the mlx-community checkpoints runnable in mlx-lm/LM Studio; vs the vendored package ~15 -> ~27 tok/s at 3-bit (component sums), 35 sliding layers bounded to 512 keys, 2048-token prefill attention 0.56 s -> ~0.15 s per chunk; server batching and prompt cache for free | 900-1,100 lines, 3-5 days + conversion | medium: numerics must match the reference (fp32 conv/router, interleaved w13); review cycle on a 1k-line model file |
| 2 | **Re-quantization recipe**: `experts_only` + attention/`unembed`/`embed` at 8-bit as the default `quant_predicate`, republished 2/3/4-bit mlx-community checkpoints with corrected cards; recommend the 4-bit (or 2-bit) tier over 3-bit for speed on 0.32.4 (§5.3). | mlx-lm (+ HF cards) | -24 to -31% weight traffic; attention projections 2x faster per layer; ~1.3x at the bandwidth limit | 40 lines + a conversion run (532 GB source; streaming conversion RFC #1842 is open) | low (8-bit attention is near-lossless); needs the source checkpoint and disk |
| 3 | **Relative-position bias table in fused SDPA** (`rel_bias[B,H,T_q,R]` added at `d = q_pos - k_pos` for `0 <= d < R`, zero elsewhere; `sdpa_vector`, 2-pass and steel/NAX kernels). | MLX core | removes the `[B,H,L,S]` mask: prefill attention per 2048 chunk from 13.4 ms to ~2.5 ms per layer (42 layers: 0.56 s -> ~0.1 s), decode global layers -0.2-0.3 ms each at 32K; also serves T5/ALiBi-style biases | 400-700 lines across 3 kernel families + tests | high: maintainer buy-in; API design; must not slow existing paths |
| 4 | **MTP drafter path**: keep `model.mtp.*` in conversion, `mtp_generate_step` with hidden-state-fed chained nextn modules, trimmable caches. | mlx-lm | 1.8-2.5x fewer target forwards per token at batch 1 (dispatch-bound regime makes this the largest decode win after #1) | 400-600 lines, 3-4 days, depends on #1 | medium: acceptance rate unknown until measured; verification rows want #4723-era kernels |
| 5 | **3-bit and 6-bit affine QMV/gather_qmv kernels** brought to 4-bit/8-bit speed (unpack path). | MLX core | 3-bit six-expert GLU 0.223 -> ~0.17 ms (the 128 GB tier's expert time -25%), 6-bit lm_head 2.76 -> ~1.9 ms; benefits every 3/6-bit mlx-community model | kernel work, 1-2 weeks | medium: generic kernels, Apple review; 3-bit packing is inherently awkward |
| 6 | **Op-fusion kernels** via `mx.fast.metal_kernel` inside the model: sconv (concat+conv+residual+state), router (GEMV+sigmoid+top-k+softmax), expert combine. | mlx-lm (model file) | ~15-20 ops per layer removed (~0.7-0.9k ops/token, 5-9 ms at the measured floor), 15-25% of the decode step | 300-400 lines | medium: K3 review dropped a decode kernel with little gain, so each must be benchmarked |
| 7 | **Six-expert decode `gather_qmv` efficiency** (290-450 GB/s of ~600; e.g. all six experts' rows in one launch with better occupancy, or a NAX gather path for M=1) and **prefill `gather_qmm_rhs` efficiency** (7-14 TFLOPS vs 31 for dense). | MLX core | up to 1.4x on the expert part of decode; up to 2x on prefill at 2048-token chunks | kernel work, 2-3 weeks | high: generic kernels, needs Apple review |
| 8 | **Shifting sliding cache as a reusable mlx-lm cache class** (`cache.py`), with `merge/extend/filter/extract/trim/to_quantized`, usable by gemma3/olmo3-style models that want trimmable windows for speculative decoding. | mlx-lm | enables speculative decoding on all SWA models past the window; ~2 MB per layer per token | 150-250 lines + tests | low |
| 9 | **Prefill `conv1d` fast path**: route grouped depthwise conv with non-zero `padding` to the depthwise kernel (pad internally). | MLX core | 4x on sconv prefill for any caller using `padding=k-1` (the vendored code) | ~40 lines | low |
| 10 | **Model card fixes** for mlx-community Inkling repos (quantization recipe, remove non-working `mlx_lm.server` snippets until #1 lands, note the 3-bit speed caveat). | HF (mlx-community) | correctness of user expectations | 1 hour | none |

## 8. Hazards

- Weight format: mlx-community checkpoints are loadable only if parameter
  paths match (`model.llm.*`); any renaming (e.g. mlx-lm-style
  `language_model.model.layers`) forces a re-conversion or a `sanitize` remap
  on both load and save. The vendored package's `w13` de-interleave is a trap
  for anyone converting from the HF source.
- Numerics: the reference keeps the short convs and router in fp32; bf16
  routing flips experts on near-ties (the vendored comment reports incoherent
  output when the router ran in bf16). Attention quantization below 8-bit was
  reported by the vendors as the dominant error source at 4-bit; validate
  before shipping 6-bit attention (which is also slower than 8-bit, §5.3).
- Quantization tiers: on MLX 0.32.4 / M5 Max the 3-bit tier is slower per
  token than 4-bit and 2-bit (kernel unpack cost), so "3-bit for 128 GB" is a
  memory decision, not a speed one, until #5 lands.
- Sliding cache and relative bias must agree on slot positions; a rotating
  cache with a wrong index silently degrades quality rather than crashing.
  Test with the vendored reference on random weights.
- Speculative decoding: `RotatingKVCache.is_trimmable()` is false once rotated
  and `ArraysCache` has no `trim`; `speculative_generate_step` raises on
  untrimmable caches. Prompt-cache files require cache classes defined in
  `cache.py`.
- Server batching requires `merge` on every cache and `kv_bits=None`; the
  window cache needs `BatchRotatingKVCache`-equivalent left-padding logic.
- Memory on a 128 GB Mac Studio: the 3-bit files are 121 GB; attention at
  8-bit saves ~3 GB on disk and in memory, and the window cache saves
  `35 x 2 x 8 x 128 x 2 B = 143 KB per token` of KV versus full history
  (4.7 GB at 32K context). Wired-limit handling in mlx-lm is already in place.
- Benchmarks here are from a shared machine (other jobs' CPU and GPU jobs
  during all runs); short-kernel numbers are consistent across runs, the
  whole-layer emulation is a lower bound, and the per-token budgets in §5.4
  are component sums. Rerun `RUN=4 ./run_all.sh` on an idle machine before
  quoting absolute tok/s externally.
- MLX API drift: `window_size` in SDPA and `sdpa_split` are post-0.32.4; an
  mlx-lm model file must keep working on the released MLX.
