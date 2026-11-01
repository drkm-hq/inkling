# R1: mlx-vlm's Inkling implementation on Apple Silicon (M5 Max, 128 GB)

Campaign: Inkling-Small (276B total / 12B active MoE) speed-ups for Apple Silicon.
Target: `mlx_vlm/models/inkling/*` in Blaizzy/mlx-vlm. Nothing pushed anywhere.

## Summary

1. Installed `mlx_vlm 0.7.11` is byte-identical to upstream HEAD 138d761 for every Inkling-relevant file; only 5 `server/*.py` files differ (plumbing). All citations below are `language.py` (`L`) at that commit.
2. The mlx-community 2-bit checkpoint (headers parsed from the shards already on disk) quantizes only routed/shared experts and the two dense MLPs; attention, router, rel-proj and **the 201024x4096 lm_head are bf16**. Per token that is 8.07 GB of weights, of which bf16 attention (3.70 GB) + lm_head (1.65 GB) are 66% and the 2-bit experts 23%. The model card's "non-expert 8-bit" is wrong for this artifact, and `model.mtp.*` is dropped, so it cannot drive the MTP drafter.
3. Measured on this M5 Max with random weights at exact shapes (3 synthetic layers x 14): the 2-bit tier decodes at a predicted 28-36 tok/s (ctx 0-2K), 22-28 (8K), 13-19 (32K); the 4-bit tier 35-40 / 27 / 14. The bandwidth roofline (490 GB/s GEMV measured) is 59 / 47 tok/s, so decode runs at 55-65% of roofline. Prefill is 640-720 tok/s at 2048-token chunks, MoE-GEMM bound.
4. Per 42-layer token at the 2-bit tier: bf16 attention projections ~12 ms, MoE ~12 ms, lm_head 3.2-3.7 ms, and a kernel-launch floor of ~0.19 ms/layer = ~8 ms (20-28%): about 50 kernels per layer are dispatched per token.
5. Biggest lever is the checkpoint recipe, not code: q8 attention + q8 lm_head cuts 8.07 -> 5.57 GB/token and measured +22% (35.7 -> 43.7 tok/s at ctx 0); all-q4 gives 6.3 GB/token and 37-39 tok/s (q4 GEMV reaches only 295-340 GB/s at these shapes vs 410-450 for q8/bf16).
6. Code levers in mlx-vlm, ranked: launch-count reduction (~5%), bounded sliding-window caches (4.7 GB saved at 32K, 18.8 GB at 128K, ~5-8% at 32K), skipping the no-op log-scaling passes (prefill 1-3%, -12.9 GB transient at 8K chunks), MTP cache transaction without full-cache snapshot + target replay (turns MTP from a net loss into ~1.4x), a prefill short-conv kernel (3-5% prefill), honoring `logits_to_keep`.
7. MLX-core levers: a fused banded attention (bias + window in-kernel: 6-10% prefill, 5% decode at 32K, and removes 1-13 GB mask temporaries); gather_qmm's down-projection tile is 3x slower than up (31 vs 10 ms per layer-chunk at q4) and the 2-bit gather GEMM is 2x slower still (prefill is 56% MoE GEMM).
8. MTP speculative decoding is not a win today: each round pays a full-cache snapshot (6 ms at 2K, 40 ms at 32K), a 5-6 ms lm_head per draft token, and on partial acceptance a restore + full target replay; cost model gives 28 tok/s vs 28-36 AR at 2K and no gain at 32K. PR #2206 removes the replay but drops the Inkling adapter.
9. Hazards found: a float32 `global_scale` promotes the residual stream to fp32 (measured 5.4x slower step; latent for non-mlx-community conversions); `kv_bits` cannot work (QuantizedKVCache returns tuples, `k.shape[2]` at L437 would fail); unbounded KV for 35 sliding layers; with a drafter chunked prefill is disabled and full-vocab logits are computed for every prompt token; 68-70 GB peak for an 8192-token layer prefill.
10. Raw numbers: `~/code/inkling-contrib/bench/R1/results/*.json` (+ `SUMMARY.md` tables); scripts in `bench/R1/`.

## Implementation map

All line numbers refer to `~/code/inkling-contrib/upstream/mlx-vlm` at HEAD 138d761
(2026-10-27). The installed `mlx_vlm 0.7.11` (`~/.venvs/inkling`) is byte-identical to
HEAD for every Inkling-relevant file (`models/inkling/*`, `models/switch_layers.py`,
`moe_offload.py`, `models/cache.py`, `models/base.py`, `utils.py`, `generate/*`,
`speculative/*`, `kernels/*`); only `server/{anthropic,app,compaction,openai,
responses_state}.py` differ (2-26 lines each, request plumbing, not generation).
`language.py` is 993 lines; `L` below means `mlx_vlm/models/inkling/language.py`.

### Model construction and the real config

`TextConfig` (`config.py:25-74`) selects sliding vs global layers through
`layer_types` or `local_layer_ids` (`config.py:61-69`) and dense vs MoE through
`mlp_layer_types` or `i < dense_mlp_idx` (`config.py:71-74`). The real
`mlx-community/Inkling-Small-mlx-2bit/config.json` (cached under
`~/.cache/huggingface/hub`) gives: hidden 4096, 42 layers, 32 q / 8 kv heads x 128
(same for SWA), `local_layer_ids` = every layer except 5, 11, 17, 23, 29, 35, 41
(35 sliding + 7 global), `sliding_window_size` 512, `d_rel` 16, `rel_extent` 1024,
`log_scaling_n_floor` 128000 / `log_scaling_alpha` 0.1, `sconv_kernel_size` 4,
`dense_mlp_idx` 2 with `dense_intermediate_size` 16384, MoE `intermediate_size` 2048,
256 routed experts top-6 + 2 shared (`shared_expert_sink`, sigmoid gate,
`norm_after_topk`, `route_scale` 8), `unpadded_vocab_size` 200058 of 201024,
`logits_mup_width_multiplier` 16, MTP head of 8 layers (`mtp_config`), quantization
`{group_size: 64, bits: 2, recipe: "experts_only"}`.

`LanguageModel` (`L912-993`) = `InklingModel` (`L876-909`: `embed_tokens` +
`embed_norm`, 42 x `InklingDecoderLayer`, final `norm`) + untied `lm_head`
(`L918-919`). `make_cache` gives every layer `CacheList(KVCache(), ArraysCache(4))`
(`L992-993`): one unbounded KV cache (sliding layers included) plus four short-conv
states (k, v, post-attention, post-MLP). The top-level `Model` (`inkling.py:45-300`)
maps the HF names (`model.llm.layers.N.attn.wq_du` ...) onto these modules
(`inkling.py:138-235`), stacks the routed experts into `switch_mlp` (`inkling.py:197-235`),
then `shared_experts_to_dense` (`L704-717`) reshapes the `[2, out, in]` shared-expert
tensors into one dense `[2*out, in]`/`[out, 2*in]` layout and `fuse_qkvr` (`L684-701`)
row-concatenates q/k/v/r into one `qkvr_proj` (`inkling.py:293`).

### What the mlx-community checkpoints actually contain (blob headers, read locally)

Parsing the safetensors headers of the three shards already on disk (first 8 bytes +
JSON header, `python -I`, no download) shows the `experts_only` recipe of the bundled
`inkling_mlx/convert.py:110-129`: only leaves named `gate_proj`, `up_proj`,
`down_proj` (routed experts, shared experts and the two dense MLPs) and the vision
`encoder` are affine-quantized (2-bit, group 64: `U32 [256, 2048, 256]` +
`BF16 [256, 2048, 64]` scales/biases). Attention (`wq_du BF16 [4096,4096]`,
`wk_dv/wv_dv BF16 [1024,4096]`, `wr_du BF16 [512,4096]`, `wo_ud BF16 [4096,4096]`),
the router (`gate.weight BF16 [258,4096]`), `rel_logits_proj BF16 [16,1024|512]`,
the short-conv weights and the norms are bf16; `gate.bias` and `gate.global_scale`
are F32, the dense `mlp.global_scale` is BF16. `embed`/`unembed` are not quantization
targets, so the 201024 x 4096 `lm_head` is bf16 (1.65 GB read per token). The model
card's "non-expert 8-bit" wording is wrong for this artifact. `model.mtp.*` is dropped
by the converter (`convert.py:56-57`), so these checkpoints cannot drive the MTP
drafter at all. The 4-bit and 3-bit tiers come from the same converter, so the same
layout is expected (not verified: not on disk).

mlx-vlm loads them through `load_model_weights` (`utils.py:620-745`): the predicate
(`utils.py:650-680`) ignores the unknown `recipe` key and quantizes exactly the modules
that have a `.scales` tensor (`utils.py:680`) at the config's bits/group, which matches.

Per-token weight traffic implied by this layout (bf16 = 2 B/param; q-bit affine g64 =
bits/8 + 4/64 B/param):

| component | params/token | 2-bit tier | 4-bit tier (same recipe) |
|---|---|---|---|
| attention qkvr+o, 42 layers, bf16 | 42 x 44.0M | 3.70 GB | 3.70 GB |
| router gate, 40 layers, bf16 | 40 x 1.06M | 0.08 GB | 0.08 GB |
| shared experts (2 x 3 x 2048x4096), 40 layers | 40 x 50.3M | 0.63 GB | 1.13 GB |
| routed experts, 6 of 256, 40 layers | 40 x 151M | 1.89 GB | 3.40 GB |
| dense MLP, layers 0-1 | 2 x 201M | 0.13 GB | 0.23 GB |
| lm_head bf16 | 823M | 1.65 GB | 1.65 GB |
| **total** | | **8.07 GB** | **10.2 GB** |

### One decoder layer, decode (B=1, L=1, context N)

`InklingDecoderLayer.__call__` (`L863-873`):

1. `_cache_padding_mask(conv, 1)` (`L220-232`) is `None` without batch padding.
2. `input_layernorm`: `mx.fast.rms_norm` over 4096 (`L852`).
3. `InklingAttention.__call__` (`L422-479`):
   - `qkvr_proj` (`L427`): one `[1,4096] x [4096, 6656]` projection
     (6656 = 32x128 q + 8x128 k + 8x128 v + 32x16 r, `L402-408`); bf16 GEMV in the
     mlx-community checkpoints, `quantized_matmul` if attention is quantized.
   - `k_sconv`, `v_sconv` on the 1024-wide k and v slices (`L429-430`): decode takes
     the fused `inkling_sconv_decode` kernel (`L253-296`, launch `L333-352`; gate:
     cache present, mask None, K==4, L<=8, Metal): grid `(rup(C,32), B, 1)`,
     threadgroup `(32,1,1)`, one thread per channel looping over L; computes the
     4-tap depthwise conv in fp32 over `[state(3 rows fp32); x]`, rounds to bf16,
     adds the input as residual (`HAS_RES` false here), writes the shifted fp32 state
     into `cache[conv_idx]` (`L353`). The k/v inputs are strided views of `qkvr` and
     the kernel keeps the default `ensure_row_contiguous=True`, so each call first
     materializes a contiguous copy.
   - `k_norm` per head (`L432`), reshape/transpose to `[1,8,L,128]`; v likewise (`L433`).
   - `KVCache.update_and_fetch` (`L436`; `cache.py:508-530`): buffers grow in steps of
     256 tokens (`cache.py:501`) by `mx.concatenate` with a zero block, the new k/v
     are slice-assigned, and views `keys[..., :offset, :]` are returned. The same
     unbounded `KVCache` is used for sliding and global layers.
   - q slice + `q_norm` (`L439-443`); r reshaped to `[1,L,32,16]` (`L440`).
   - Sliding layers only (`L445-455`, `_SLIDING_KV_SLICE`): when `S > L + 511` the
     K/V views are cut to the last `L + 511` positions (`j0 = S - L - 511`), so
     decode always sees S_eff = 512. Global layers keep all N keys.
   - `banded_additive_mask` (`L458-467`, `L149-217`): the shape-generic
     `inkling_banded_mask_v2` kernel (`L101-142`) runs whenever `shape_ref` carries S
     and `q_offset == S - LQ` (always true from this call site; the v1 kernel `L69-98`
     is effectively dead code). Grid `(rup(S,8), rup(LQ,8), B*H)`, threadgroup
     `(8,8,1)`, one thread per `(key j, query i, head)`; output `[B, 32, LQ, S]` in
     the activation dtype. Per element: `-1e30` if `dist < 0` or (sliding and
     `dist >= 512`), else a 16-term dot product `rel[b,i,h,:] . proj[:, dist]`
     if `dist < rel_extent` (1024 global / 512 sliding, `L392-394`), else 0.
     `rel_proj.astype(x.dtype)` (`L460`) runs every call.
   - Global layers only: the log-scaling branch (`L468-473`) builds `tau` from the
     query positions (`1 + 0.1*log(max(pos/128000, 1))`, 7 small ops) and applies
     `q = q*tau` and `mask = where(mask > -1e29, mask*tau, mask)`: three extra full
     passes over the `[1,32,LQ,S]` mask. For every context below 128k tokens `tau`
     is exactly 1, so this is pure overhead (2 MB at decode S=32K; 1-4 GB per global
     layer per prefill chunk).
   - `scaled_dot_product_attention` (`L475-477`; `base.py:392-452`) ends in
     `mx.fast.scaled_dot_product_attention(q, k, v, scale=1/128, mask=mask)`
     (`base.py:446`) with the dense additive mask array; GQA 32/8; scale is `1/d`
     because q and k are RMS-normed (`L396`). No window argument, no `"causal"`
     string: the fused kernel must read the full mask. `cache=None` is passed
     (`L476`), so the wrapper's quantized-KV branch (`base.py:433-444`) is unreachable.
   - transpose/reshape and `o_proj` `[4096 x 4096]` (`L478-479`).
4. `attn_sconv(r, cache=conv, residual=x)` (`L868`): the fused kernel again on C=4096
   with `HAS_RES=True`, i.e. the layer residual add is folded in (`L275-282`).
5. `post_attention_layernorm` then `mlp` (`L869`):
   - Dense layers 0-1: `InklingDenseMLP` (`L482-490`) = `SwiGLUMLP` (gate, up
     `[4096 -> 16384]`, compiled `swiglu` `activations.py:7-9`, down) times
     `global_scale` (`L490`, a `[1]` parameter whose dtype decides the dtype of the
     residual stream from here on, see Hazard H1).
   - MoE layers: `InklingSparseMoE.__call__` (`L795-840`). Router logits
     `xf @ gate_weight.T` (`L801`, bf16 `[258, 4096]`). `_route` (`L742-793`): on
     Metal the single `inkling_moe_route` kernel (`L533-600`), grid `(32, N, 1)`, one
     simdgroup per token: sigmoid + correction bias, six rounds of `simd_max` top-k
     with deterministic tie-break, logsigmoid-softmax over the 6 routed + 2 shared
     logits, times `route_scale * global_scale` (`_wscale`, fp32, cached `L748-751`).
     Outputs `idx [N,6] uint32`, `topk_w [N,6]`, and `gamma [N, 2*2048]`: the two
     shared-expert weights pre-broadcast over the expert-major intermediate
     (`L592-593`). The pure-MLX fallback is `L772-793`.
     Routed experts, two paths (`L805-838`):
     * `_DOWN_COMBINE` path when `N <= 8`, Metal, down_proj is affine **q4/g64 with
       biases and input_dims 2048**, scales dtype == x dtype, trivial NVFP4 scales:
       up and gate through `gather_qmm` (`L822`; `switch_layers.py:148-163`), then
       `inkling_moe_down_combine` (`L606-659`): grid `(256, 4096, N)`, 8 simdgroups
       per threadgroup (one per selected expert slot, 6 used), each lane owns one
       64-wide quant group of the 2048 inputs; dequantizes the q4 rows directly and
       folds the top-k weighted sum, so the `[N,6,4096]` intermediate never exists.
     * otherwise (2-bit, 3-bit, 8-bit, nvfp4, N > 8): `InklingSwitchGLU.__call__`
       (`L513-530`) -> trivial scales -> `SwitchGLU.__call__`
       (`switch_layers.py:249-316`) with 2-D `xf`, which skips the
       `DECODE_BLOCK_SIZE` branch (needs `x.ndim == 3`, `L252`), expands to
       `[N,1,1,4096]`, sorts only when `indices.size >= 64` (`L298`, i.e. N >= 11),
       and issues three `gather_qmm` calls (up, gate, down); the caller then does
       `(out * topk_w[..., None]).sum(-2)` (`L838`).
     Shared experts: `InklingSharedExpertsDense` (`L667-681`): gate and up
     `[4096 -> 4096]` (2 experts x 2048 concatenated), compiled
     `silu(gate)*up*gamma` (`L662-664`), down `[4096 -> 4096]`; quantized at the
     tier's bits in the checkpoints. Finally `(yr + ys).reshape(B,L,D).astype(x.dtype)`
     (`L840`).
6. `mlp_sconv(r, cache=conv, residual=h)` (`L870`): fused kernel, C=4096, residual folded.
7. `conv.advance(1)` (`L871-872`; `cache.py:1278-1282`) only moves padding counters.

Head (`L921-930`, called from `L947-949`): final `rms_norm`, `h / 16.0`, `lm_head`
`[4096 -> 201024]` (bf16 GEMV, 1.65 GB), `logits[..., :200058]`. `generate_step`
(`generate/ar.py:365-417`) takes `logits[:, -1, :]`, runs processors, computes
`logprobs = logits - logsumexp(logits)` over the 200058-wide row, samples, and
pipelines token n+1 with `mx.async_eval` before waiting on token n (`ar.py:541-565`);
cache states are force-evaluated every 50 tokens (`ar.py:553-554`) and
`mx.clear_cache()` runs every 256 tokens (`ar.py:557-558`).

Kernel count per MoE layer at decode (from the ops above): about 30 on the attention
side (GEMV, 2 contiguous copies + 2 sconv, 2 norms, 2 transposes/copies, KV
slice-updates, mask, SDPA, o_proj, plus ~9 more on global layers for tau) and about 20
on the MoE side (router GEMV, route kernel, 3 gather_qmm or 2 + down-combine, 3 dense
GEMV + compiled swiglu, adds/casts, sconv). Measured: a same-graph tiny-shape model
takes 0.56 ms per token for 3 layers + head (`kernels_bench.json`), i.e. ~0.19 ms of
launch/latency per layer.

### One decoder layer, prefill (T tokens)

`generate_step` chunks the prompt at `prefill_step_size` = 2048 (`common.py:28`,
loop `ar.py:476-515`) whenever there is more than one token and no drafter
(`_chunked_prefill_enabled` returns `draft_model is None`, `common.py:132`: with an
MTP drafter the entire prompt is one chunk). Inkling's `LanguageModel.__call__`
(`L932-955`) ignores `logits_to_keep` and `n_to_process` (they land in `**kwargs`), so
each chunk's graph contains the full `[T, 201024]` lm_head; the chunk loop evaluates
only the cache states (`ar.py:505`), so MLX prunes it, but the unchunked drafter path
and `_step` on a multi-token input (`ar.py:517`, `ar.py:389`) do compute it.

Per layer and chunk of L tokens with S cached positions:
- attention: `qkvr_proj` GEMM `[L,4096]x[4096,6656]`; two **unfused** sconvs
  (`L355-369`: cast to fp32, `mx.concatenate([state, x])`, `nn.Conv1d` depthwise, add,
  cast; plus `_next_conv_state`'s concatenate+slice `L235-250`); k_norm; KV
  slice-update (and a 256-step reallocation + concatenate of the whole cache when the
  buffer is exceeded); sliding layers cut K/V to S_eff = L + 511; the v2 mask kernel
  writes `[1,32,L,S_eff]` bf16 (L=2048,S=8192: 1.07 GB; L=S=8192: 4.3 GB); global
  layers then run the three tau passes over it; `mx.fast.scaled_dot_product_attention`
  with the dense mask computes all L x S_eff scores (for sliding layers only 512 of
  each row's L+511 keys are unmasked, so ~80% of the score work at L=2048 is wasted);
  `o_proj` GEMM.
- MoE: router GEMM; route kernel with N=L; `SwitchGLU` sorted path
  (`switch_layers.py:298-316`): `_gather_sort` (`L86-91`: argsort of the 6L indices,
  inverse argsort, gather of x rows -> `[6L,1,4096]`, i.e. a 6x copy of the chunk's
  activations, 100 MB at L=2048), `gather_qmm(..., sorted_indices=True)` for up and
  gate, compiled swiglu, `gather_qmm` down, `_scatter_unsort`, weight-and-sum.
  Shared experts: three dense GEMMs `[L,4096]x[4096,4096]`.
- dense layers: three GEMMs with 16384.

### MTP drafter path (bundled or split `inkling_mtp`)

`InklingMTPDraftModel` (`speculative/drafters/inkling_mtp/inkling_mtp.py:82-385`):
8 blocks (`num_nextn_predict_layers` 8; `mtp_config.local_layer_ids` [0,2,4,5,6,7]
so blocks 1 and 3 are global), each `embed_norm`/`hidden_norm` -> concat ->
`input_proj [8192 -> 4096]` -> a reused `InklingDecoderLayer` configured with
`mlp_layer_types=["dense"]*n` (`inkling_mtp.py:106-115`), i.e. a dense 16384 MLP.
Drafting (`draft_block`, `L252-295`) is sequential: block `min(round, 7)` per draft
token (`L218-224`) and a **full `lm_head` call per draft token** (`_block_logits`,
`L197-201`). Each round also snapshots the drafter's caches (`L269`). The target side
(`L957-986`): `speculative_verify_hidden` copies the state of **every** target cache
(`_snapshot_cache_state`, `L34-54`: `mx.array(value)` per tensor, evaluated) before
each verify forward, and on any partial acceptance `rollback_speculative_cache`
(`L974-986`) restores that copy and re-runs the target on the accepted tokens
(`L985`). The round loop is `speculative/mtp.py:587-742`; `commit_speculative_round`
(`cache_state.py:216-228`) calls the rollback only when `accepted < block_size - 1`.
The verify forward uses the normal decode path with N = block_size tokens (<= 8 keeps
the sconv kernel and the q4 down-combine path). Bundled drafters are loaded through
`load_drafter` (`drafters/__init__.py:201-227`) and the splitter registry
(`mtp_split.py:326-343`); batch > 1 acceptance raises (`inkling_mtp.py:348-350`).

## Measurements

### Method

Machine: MacBook Pro M5 Max, 128 GB, macOS 26.5.1, MLX 0.32.4
(`max_recommended_working_set_size` 115.4 GB). Two model downloads and other jobs'
benchmarks were running during this session (a `test_oracle_vs_transformers.py` process
and `bench/R3/bw.py` overlapped some runs), so run-to-run variance reached 1.5x on
some rows. Each table states which run it comes from; "best of runs" takes the fastest
clean run because contention only inflates times. Every number is a median over
`reps` synchronous `mx.eval` calls after warm-up; per-layer decode costs are measured
as a chain of 12 dependent layer calls inside one eval (amortizing the ~0.14-0.3 ms
command-buffer latency that single-op timing carries, see `bw_probe.py` vs
`bw_probe2.py`). Random weights, exact Inkling-Small shapes, real `mlx_vlm` classes.

Scripts (all under `~/code/inkling-contrib/bench/R1/`, run with
`source ~/.venvs/inkling/bin/activate`):

| script | what | output |
|---|---|---|
| `bw_probe2.py` | dispatch-amortized GB/s of bf16/q8/q4/q2 GEMV and gather_qmm at Inkling shapes | `results/bw_probe2.json` |
| `layer_bench.py --experts-bits {2,4} [--attn-bits 8|4] [--head-bits 8|4] [--skip-prefill]` | 3-layer synthetic model (L0 dense+SWA, L1 MoE+SWA, L2 MoE+global) via `mlx_vlm.models.inkling.language.LanguageModel`; per-layer decode at ctx 0/2K/8K/32K, full-loop decode pipelined like `generate_step`, prefill 512/2048/8192 + 2048-chunks at offsets 2048/6144; 42-layer extrapolation | `results/layer_bench_<tag>.json` |
| `kernels_bench.py` | each custom Metal kernel in isolation vs its MLX alternative; SDPA with/without mask; MoE block decode/prefill paths; launch-overhead floor (tiny-shape model, same graph) | `results/kernels_bench.json` |
| `sdpa_probe.py` | fused-SDPA behaviour with array/bool/causal masks, peak-memory deltas, `force_fused` | `results/sdpa_probe.json` |
| `kv_probe.py` | does `KVCache.update_and_fetch` copy the buffer per token (sync / pipelined / chained); RotatingKVCache(512) alternative | `results/kv_probe.json` |
| `mtp_probe.py` | verify forward for N=1/3/5/8, `_snapshot_cache_state`/restore cost, draft-step cost | `results/mtp_probe.json` |
| `diag_loop.py` | Hazard H1 reproduction (fp32 `global_scale`) | `results/diag_loop.json` |
| `summarize.py` | regenerates `results/SUMMARY.md` (all tables) | `results/SUMMARY.md` |
| `run_all{,2,3,4}.sh` | the sequential runs (logs next to the json) | `results/*.log` |

Extrapolation: 42 layers = 2 x L0 + 33 x L1 + 7 x L2 (+ embed + final norm + lm_head +
argmax). "parts" sums the chained per-layer medians; "loop" adds (mult-1) x layer time
to the measured pipelined 3-layer loop (so it includes the real head and dispatch).

### Achievable bandwidth (bw_probe2.json)

| op (M=1) | ms/op | MB | GB/s |
|---|---|---|---|
| gemv 4096x4096 bf16 / q8 / q4 / q2 | 0.074 / 0.043 / 0.032 / 0.022 | 33.6 / 17.8 / 9.4 / 5.2 | 452 / 415 / 296 / 237 |
| gemv 6656x4096 (qkvr) bf16 / q8 / q4 | 0.121 / 0.071 / 0.045 | 54.5 / 29.0 / 15.3 | 452 / 411 / 338 |
| dense MLP up+down 16384 bf16 / q8 / q4 / q2 | 0.561 / 0.317 / 0.182 / 0.136 | 268 / 143 / 75 / 42 | 478 / 450 / 414 / 309 |
| lm_head 201024x4096 bf16 / q8 / q4 / q2 | 3.35 / 1.80 / 0.94 / 0.53 | 1647 / 875 / 463 / 257 | 492 / 486 / 493 / 483 |
| SwitchGLU gather_qmm N=1 K=6 q2 / q4 (up+gate+down) | 0.169 / 0.256 | 47 / 85 | 280 / 331 |
| 1 GB bf16 copy (read+write) | 5.47 (1 GB) | 2147 | 393 |

Practical roofline for this machine: ~480 GB/s for large bf16/q8 streams; q4 small
projections reach 300-340 GB/s and q2 240-310 GB/s (dequantization-bound at M=1).

### Decode, per layer and per token (layer_bench)

Best of runs (`e2_abf16_hbf16` first clean run and `_decode2`; `e4_abf16_hbf16` first
clean run), ms per call; GB/token from the table above; the final canary pass
(`*_final`) is appended at the end of this section when available.

2-bit tier as shipped (experts q2, attention bf16, lm_head bf16):

| ctx | L0 dense+SWA | L1 MoE+SWA (attn / mlp / sconv) | L2 MoE+global (attn / mlp / sconv) | 3-layer loop async / sync / graph-build | 42-layer tok/s (parts / loop) | GB/tok | roofline @480 |
|---|---|---|---|---|---|---|---|
| 0 | 0.638 | 0.569 (0.284 / 0.288 / 0.023) | 0.606 (0.310 / 0.266 / 0.032) | 4.93 / 5.97 / 0.39 | 35.7 / 36.5 | 8.07 | 59.5 |
| 2048 | 0.665 | 0.742 (0.359 / 0.381 / 0.026) | 0.781 (0.452 / 0.370 / 0.023) | 4.15 / 6.00 / 0.37 | 27.8 / 29.1 | 8.21 | 58.5 |
| 8192 | 0.743 | 0.863 (0.418 / 0.372 / 0.036) | 1.051 (0.618 / 0.360 / 0.032) | 4.73 / 5.54 / 0.32 | 22.7 / 25.4 | 8.38 | 57.3 |
| 32768 | 0.946 | 1.104 (0.751 / 0.390 / 0.035) | 1.781 (1.459 / 0.447 / 0.047) | 4.74 / 6.90 / 0.29 | 17.4 / 19.3 | 9.09 | 52.8 |

head: lm_head 3.15-3.67 ms (1.65 GB, 450-520 GB/s), norm + lm_head + argmax 3.3 ms,
embed 0.14-0.33 ms. Residual stream verified bf16 after 3 layers.

4-bit tier as shipped (experts q4, attention bf16, lm_head bf16; first clean run):

| ctx | L0 | L1 (attn / mlp / sconv) | L2 (attn / mlp / sconv) | 3-layer loop async / sync / build | 42-layer tok/s (parts / loop) | GB/tok | roofline @480 |
|---|---|---|---|---|---|---|---|
| 0 | 0.476 | 0.516 (0.245 / 0.283 / 0.026) | 0.559 (0.268 / 0.287 / 0.024) | 4.41 / 4.82 / 0.27 | 39.4 / 40.4 | 10.19 | 47.1 |
| 2048 | 0.532 | 0.589 (0.292 / 0.294 / 0.028) | 0.632 (0.354 / 0.299 / 0.029) | 4.50 / 4.78 / 0.31 | 35.2 / 36.1 | 10.32 | 46.5 |
| 8192 | 0.643 | 0.754 (0.393 / 0.372 / 0.035) | 1.133 (0.670 / 0.436 / 0.032) | 4.60 / 4.87 / 0.48 | 26.6 / 27.6 | 10.50 | 45.7 |
| 32768 | 1.623 | 1.475 (0.961 / 0.499 / 0.051) | 2.095 (1.470 / 0.470 / 0.050) | 8.77 / 23.97 / 0.41 | 14.3 / 14.2 | 11.20 | 42.9 |

(The 4-bit `_decode2` repeat ran during another GPU job and was 1.5x slower
across context-independent ops; discarded. Both 4-bit runs show the q4 MoE block at
0.28-0.29 ms, identical to the q2 block, because q2 gather_qmm reaches only 280 GB/s.)

Hypothetical recipes (same session as the 2-bit `_decode2` run, `--skip-prefill`):

| recipe | GB/tok | ctx 0 | ctx 2048 | ctx 8192 | ctx 32768 | lm_head ms |
|---|---|---|---|---|---|---|
| experts q2, attention q8, lm_head q8 (`e2_a8_h8_decode2`) | 5.57 | 43.7 / 46.6 | 32.3 / 34.5 | 24.0 / 25.2 | 17.1 / 18.4 | 2.28 |
| all q4 (`e4_a4_h4_decode2`) | 6.34 | 37.2 / 39.0 | 31.6 / 33.1 | 26.8 / 28.5 | 14.6 / 15.3 | 1.19 |
| experts q2, attention q4, lm_head q4 (`e2_a4_h4`, no warm-up run) | 4.23 | 22.7 / 32.4 | 21.0 / 30.1 | 14.5 / 18.0 | 14.6 / 18.5 | 2.97 |

Final canary pass (`run_all4.sh`, `*_final` tags, decode only, GPU warm-up before each
context): the end-of-run canaries show the GPU was shared with another process for
most of it (1 GB bf16 copy 8.5-9.0 ms vs 5.5 ms when quiet in `bw_probe2`; the 2-bit
run collapsed to 5 tok/s and is discarded). The one quiet run (`e4_a4_h4_final`,
canary 4.58 ms, lm_head 0.97-1.14 ms) corroborates the all-q4 row: 30.9 / 29.9 / 28.0
/ 17.7 tok/s at ctx 0 / 2K / 8K / 32K (parts), 31.2 / 30.2 / 29.4 / 18.9 (loop).
`e4_a8_h8_final` (canary 5.74 ms) gave 24.3 / 25.1 / 15.4 / 17.0 tok/s for q4 experts
+ q8 attention + q8 head (7.68 GB/token), a lower-confidence point. All per-row data is
in `results/SUMMARY.md`.

Per-token budget at the 2-bit tier, ctx 2048 (from the rows above): lm_head + norm +
argmax 3.3 ms (9%); dense layers 2 x 0.67 = 1.3 ms; 33 sliding MoE layers x 0.74 =
24.5 ms (attention 0.36 x 33 = 11.8 ms: 88 MB of bf16 weights at ~250 GB/s effective
plus ~0.1 ms of small ops; MoE 0.38 x 33 = 12.6 ms: 63 MB of q2 weights at ~170 GB/s
effective); 7 global layers x 0.78 = 5.5 ms; total ~35 ms = 28 tok/s against a 17 ms
(58 tok/s) byte roofline. The launch floor measured with the tiny-shape model (0.56
ms per 3 layers + head, `kernels_bench.json`) accounts for ~8 ms of the gap; the rest
is the q2 gather_qmm efficiency (280 GB/s) and the many small ops between GEMVs.

### Prefill (layer_bench, first clean runs), ms per layer call

2-bit tier:

| T | L0 (attn/mlp/sconv) | L1 (attn/mlp/sconv) | L2 (attn/mlp/sconv) | L2 attn without tau | 42-layer ms | tok/s |
|---|---|---|---|---|---|---|
| 512 | 14.0 (3.6/11.5/0.5) | 15.4 (2.3/12.9/0.4) | 25.5 (7.9/18.3/0.5) | 3.4 | 714 | 717 |
| 2048 | 56.4 (22.5/39.9/7.4) | 76.6 (25.1/54.6/1.3) | 79.6 (22.6/55.5/3.2) | 19.7 | 3197 | 641 |
| 8192 (one chunk) | 284 (137/191/7) | 362 (181/230/6) | 460 (227/200/6) | 177 | 15718 | 521 |

Steady-state 2048-token chunk at an offset: L1 70.9 ms (offset 2048) / 63.8 (6144);
L2 89.2 / 107.6 ms -> ~3.3 s per chunk at 6-8K context (~620 tok/s).

4-bit tier: T=512 497 tok/s (L1 24.7 ms: mlp 21.7), T=2048 711 tok/s (L1 69.1:
attn 14.7 / mlp 52.6 / sconv 1.3; L2 70.4: attn 19.9 / mlp 52.7), T=8192 652 tok/s
(L2 395.6: attn 234 / mlp 205); chunk at offset 6144: L1 73.4, L2 102.1 ms.

Peak MLX memory of these runs: 68.6 GB (2-bit) / 70.2 GB (4-bit) with 7.7-11 GB of
weights: the 8192-token global-layer chunk alone materializes a 4.3 GB mask plus three
mask-sized tau temporaries (12.9 GB) plus the MoE sorted-path intermediates.

### Custom kernels and primitives in isolation (kernels_bench.json, sdpa_probe.json)

Decode (chained x8; S = keys seen by the kernel):

| | S=512 (SWA) | 2048 (global) | 8192 | 32768 |
|---|---|---|---|---|
| `inkling_banded_mask_v2` [1,32,1,S] | 0.106 | 0.037 | 0.057 | 0.125 |
| SDPA with the additive mask | 0.119 | 0.044 | 0.090 | 0.315 |
| SDPA mask=None | 0.064 | 0.040 | 0.074 | 0.287 |
| mask + SDPA as in `InklingAttention` | 0.038 | 0.056 | 0.111 | 0.470 |
| tau `where()` pass over the mask (L473) | - | 0.028 | 0.028 | 0.034 |

(The S=512 row is at the chained-timing noise floor; sub-0.05 ms numbers are not
resolvable.) `sdpa_probe.json`: at LQ=1 the fused kernel is used for every mask kind
(no peak-memory delta), array vs none costs +0.02-0.06 ms up to S=32K.

Prefill:

| LQ x S | mask_v2 write | SDPA + array mask | SDPA "causal" | tau pass | SDPA none |
|---|---|---|---|---|---|
| 2048 x 2048 (SWA) | 2.27 ms (0.27 GB) | 1.86 | 0.90 | - | 1.57 |
| 2048 x 2559 (SWA after slice) | 2.70 | 2.23 | 1.27 | - | 1.83 |
| 2048 x 2048 (global) | 2.83 | 1.86 | 0.89 | 3.95 | - |
| 2048 x 8192 (global) | 7.35 (1.07 GB) | 7.38-7.85 | 5.92 | 15.98 | 6.50 |
| 8192 x 8192 (global) | 28.5 (4.29 GB) | 27.4-55.0 | 10.6-36.9 | 62.2 | 81.8 |

`force_fused=True` changes nothing at LQ<=2048 and is 2.3x slower at 8192x8192; the
fused kernel is already used with array masks (peak-memory delta 0.00 GB in every
case; the unfused reference adds 0.34-1.24 GB and is 2.5-3x slower). A bool mask is
30-100% slower than the additive one.

Short conv (chained): fused decode kernel C=4096 0.023 ms vs unfused 0.069 ms (3x);
C=1024 0.129 vs 0.132 (noise floor); prefill T=2048 conv1d path 0.45 ms (C=1024),
1.49 ms (C=4096) per site.

Route: `inkling_moe_route` 0.142 ms (N=1) / 0.143 (N=8) / 0.161 (N=2048) vs the
pure-MLX chain 0.234 / 0.202 / 0.261; these single-op timings sit on the ~0.14 ms
eval floor, so the kernel's true cost is well below 0.05 ms; the router GEMV alone
(2 MB) times the same 0.14 ms.

MoE block (chained x6, includes router, route kernel, shared experts, routed experts,
weighted sum):

| | N=1 | N=3 | N=8 | N=2048 |
|---|---|---|---|---|
| q4, `_DOWN_COMBINE` kernel | 0.282 | 0.532 | 1.167 | 30.5 (sorted gather_qmm) |
| q4, gather_qmm down (`_DOWN_COMBINE=False`) | 0.299 | 0.609 | 1.163 | - |
| q2 (always gather_qmm) | 0.388 | 0.711 | 1.904 | 79.5 |

The down-combine kernel buys 6% at N=1 and nothing at N=8 on this machine/MLX
version. Prefill N=2048 components: q4 up_proj sorted 10.4 ms (unsorted 568 ms!),
down_proj sorted 31.3 ms, dense reference `[2048,4096]x[4096,2048]` bf16 2.6 ms / q4
3.6 ms; q2 up 21.6 ms, down 22.1 ms, reference q2 3.0 ms.

Launch-overhead floor: a tiny-shape 3-layer model with the identical op graph decodes
at 0.564 ms per token (pipelined loop; Python graph build 0.165 ms of it), i.e. about
0.19 ms per layer -> ~8 ms per 42-layer token.

### KV cache update (kv_probe.json), ms per one-token update of one layer

| ctx | A sync (one update per eval) | B pipelined like generate_step | C chained x8 (floor amortized) | D update + SWA slice + SDPA (sync) | E RotatingKVCache(512) update + SDPA (sync) |
|---|---|---|---|---|---|
| 2048 | 0.246 | 0.139 | 0.093 | 0.296 | 0.283 |
| 8192 | 0.266 | 0.179 | 0.123 | 0.294 | 0.275 |
| 32768 | 0.287 | 0.320 | 0.334 | 0.334 | 0.229 |

The update is not a whole-buffer copy in the sync pattern (A grows 0.04 ms from 2K to
32K), but the chained/pipelined patterns grow by 0.18-0.24 ms per layer at 32K (the
134 MB K+V buffer cannot be donated while the previous graph still references it);
over 42 layers that is 8-10 ms per token at 32K, consistent with the loop-based
numbers above. A bounded 512-entry rotating cache is 0.1 ms cheaper per sliding layer
at 32K (E vs D).

### MTP cost inputs (mtp_probe.json, 2-bit experts, 3 synthetic layers)

| | ctx 2048 | ctx 32768 |
|---|---|---|
| verify forward (hidden only) N=1 / 3 / 5 / 8 | 3.04 / 3.13 / 3.58 / 4.38 | 2.83 / 3.74 / 4.88 / 8.79 |
| + argmax head | 6.52 / 6.39 / 6.91 / 8.44 | 6.33 / 7.43 / 9.10 / 17.5 |
| `_snapshot_cache_state` (3 layers) | 0.43 (25 MB) | 2.86 (403 MB) |
| `_restore_cache_state` | 0.31 | 1.12 |
| draft step: dense layer + lm_head + argmax | 6.13 | - |
| lm_head + argmax alone | 5.15 | - |

Scaled to 42 layers (x14 for the cache copies): snapshot 6 ms at 2K / 40 ms at 32K per
round, restore 4 / 16 ms, verify(bs=3) = token time + ~2 ms, draft token ~5-6 ms
(dominated by the bf16 lm_head).

### Hazard H1 reproduction (diag_loop.json, 3-layer 2-bit model, ctx 0)

| `InklingDenseMLP.global_scale` dtype | residual dtype after layer 0 | 3-layer loop (pipelined) | head on model output | model without head |
|---|---|---|---|---|
| bf16 (as in the mlx-community checkpoint) | bfloat16 | 4.38 ms | 3.50 ms | 1.96 ms |
| float32 | float32 | 23.7 ms | 16.3 ms | 8.04 ms |

## Opportunities, ranked

Baseline for the arithmetic: 2-bit tier, ctx <= 2K, ~35 ms/token (28 tok/s) of which
attention bf16 projections ~12 ms, MoE ~12 ms, lm_head 3.3 ms, launch floor ~8 ms;
prefill 3.2 s per 2048-token chunk (640 tok/s) of which MoE gather GEMMs ~1.8 s,
attention ~1.0 s.

1. **Quantize attention and lm_head in the checkpoint recipe** (q8 attention + q8 or q6
   lm_head; experts unchanged). Bytes 8.07 -> 5.57 GB/token (-31%); measured in the
   same session 35.7 -> 43.7 tok/s at ctx 0 (+22%), 27.8 -> 32.3 at 2K (+16%); the
   4-bit tier goes 10.2 -> 7.7 GB. All-q4 (6.3 GB) measured 37-39 tok/s, no better
   than q8 because q4 GEMV at 4096-6656 rows runs at 296-338 GB/s vs 411-452 for q8,
   and the bundled converter's own note says attention dominates 4-bit error, so q8
   is the safe choice. Files: the artifact recipe (`inkling_mlx/convert.py:110-129`
   `_RECIPES`, a new `experts_only_q8attn`) or an Inkling-aware mixed predicate in
   `mlx_vlm/convert.py:68-95` (today only `down_proj`-based `mixed_*` recipes). Risk:
   quality, needs a perplexity check on real weights; engineering risk nil.
   Contribution: mlx-community checkpoints / mlx-vlm convert.
2. **Fewer kernels per layer** (mlx-vlm, `language.py`). The launch floor is ~0.19
   ms/layer = ~8 ms/token (22-28%). Per MoE layer: drop the two contiguous copies in
   front of the k/v sconv kernels by declaring `_sconv_kernel` with
   `ensure_row_contiguous=False` and strided indexing like the v2 mask kernel
   (`L291-296`, `L333-352`); pre-cast `rel_proj` at load instead of `L460`; skip the
   nine-op tau branch when `offset + L <= log_floor` (`L468-473`); fold the top-k
   weighted sum into the down projection for every bit width (generalize
   `_DOWN_COMBINE`, `L805-838`, which today only serves q4/g64) and fold `yr + ys`,
   the final `astype` (`L840`) and the `h / mup` (`L922`, fold into `norm.weight` at
   load). Roughly 10 of ~50 kernels per layer: ~1.5-2 ms/token (5%). Low risk.
3. **Bounded sliding-window caches** (mlx-vlm): `make_cache` (`L992-993`) gives the 35
   sliding layers an unbounded `KVCache` although `L445-455` never reads beyond
   L+511 keys. Memory 42 x ctx x 4 KB: 5.6 GB at 32K (4.7 GB of it wasted), 22 GB at
   128K, on top of 88 GB of 2-bit weights under a 107.5 GiB wired limit. Speed: 0.1
   ms per sliding layer per token at 32K (kv_probe E vs D) = 3.5 ms/token (5-8%), and
   no 256-step reallocations. Needs temporal order for the distance-based bias (use
   a shift-left window of 512+block entries rather than the circular
   `RotatingKVCache._update_in_place`, or apply `_temporal_order`), `trim` for the
   MTP rollback, and the batch path (`BatchRotatingKVCache`, `ar.py:974-977`).
   Medium effort, low-medium risk.
4. **MTP without full-cache snapshot and target replay** (mlx-vlm). Per round today:
   snapshot 6 ms (2K) / 40 ms (32K), restore 4 / 16 ms, and on partial acceptance a
   second target forward (~35 ms); plus ~5-6 ms per draft token for the bf16
   lm_head. Cost model, bs=3, 60% partial-acceptance probability, E[tokens]=2.3:
   2K: 10 + 37 + 6 + 0.6 x 40 = 77 ms -> 30 tok/s vs 28-36 AR (no gain); 32K:
   10 + 39 + 40 + 0.6 x 55 = 122 ms -> 19 tok/s vs 17-19 AR. Replace the snapshot with
   `KVCache.trim` (offset rollback, `cache.py:584-587`) plus a 4 x [1,3,C] conv-state
   copy per layer (8 MB total), and record conv states per verified position
   (`ArraysCache.record_speculative_states`/`update_window`, `cache.py:933-1060`,
   already used by other families) so no replay is needed: 2K round -> ~50 ms ->
   ~45 tok/s (1.4x) if per-draft acceptance is >= 60%. Also quantize the lm_head
   (item 1) since every draft token pays it. Risk: medium; `test_speculative.py` has
   Inkling greedy-invariance tests. PR #2206 implements this transaction model
   generically but removes the Inkling adapter; writing an adapter for its contract
   is the forward-compatible route.
5. **Skip the log-scaling passes when tau == 1** (mlx-vlm, 3 lines at `L468`): tau is
   exactly 1 for positions < 128000. Prefill saves 4 ms (S=2K) to 16 ms (S=8K) per
   global layer per 2048 chunk = 28-112 ms per chunk (1-3%), 62 ms per global layer
   on an 8192 chunk, and 3 mask-sized temporaries (12.9 GB at 8192x8192). Decode
   0.2 ms/token. Zero risk.
6. **Fused banded attention** (MLX core preferred; could be an mlx-vlm Metal kernel):
   compute `rel . proj[:, dist]` and the window inside a flash-attention kernel
   instead of materializing `[1,32,L,S]`. Prefill per 2048 chunk: 42 x (mask 2.3-2.8
   + SDPA-with-mask 1.9-2.3 - windowed SDPA ~0.9-1.3) ~ 150 ms (5%) at short context;
   at 8K context 7 x (7.3 + 16 + 7.4 - 1) + 35 x (2.7 + 2.2 - 0.4) ~ 370 ms (10%), and
   it removes the 1-4.3 GB mask temporaries (the 68-70 GB peaks). Decode: 0.02-0.18
   ms per global layer, ~2-3 ms/token at 32K (5%). High effort, needs its own test
   suite; the v2 mask kernel also writes at only 146 GB/s (7.35 ms for 1.07 GB), so a
   cheaper interim step is a faster mask kernel (precompute `rel @ proj` as
   `[1,32,L,extent]` then gather) at ~2x.
7. **gather_qmm efficiency for prefill** (MLX core): q4 `down_proj` sorted gather GEMM
   is 31.3 ms vs 10.4 ms for `up_proj` at identical FLOPs/bytes (K=2048 -> N=4096
   tile shape); q2 up/down are 21.6 / 22.1 ms. MoE GEMMs are 56% of a 2048-token
   chunk (2-bit); matching down to up would save ~20 ms x 40 layers = 0.8 s per chunk
   at q4 (711 -> ~980 tok/s). Effort high (steel kernels).
8. **Prefill short-conv kernel** (mlx-vlm): the fused kernel is gated to L <= 8
   (`L325`); the unfused path costs 0.45 ms (C=1024) / 1.49 ms (C=4096) per site at
   T=2048, i.e. 2 x 0.45 + 2 x 1.49 ~ 3.9 ms per layer = 160 ms per chunk (5%) plus
   the fp32 temporaries. A 2-D (channel, position) grid version of the same kernel.
   Low risk.
9. **Honor `logits_to_keep`** in `LanguageModel.__call__` (`L932-955`): any multi-token
   `_step` (unchunked prefill with a drafter, `ar.py:450-459` + `common.py:132`;
   prompt processing in the server) computes `[T, 201024]` logits: 3.4 TFLOP and
   0.8 GB per 2048 tokens (~0.2 s), 3.3 GB at 8192. Three lines, no risk.
10. **Batching** (usage; code exists): the 6.2 GB of non-expert bytes per step are
    independent of batch size; at B=4 the step reads ~6.2 + ~4 x 1.9 GB (worst case)
    = 13.8 GB for 4 tokens -> ~2.3x throughput at ~1.7x the latency. `BatchKVCache` +
    `ArraysCache.left_padding` are wired for Inkling (`ar.py:946-981`, `L214-216`);
    MTP batch > 1 is not (`inkling_mtp.py:348-350`).
11. Not worth it: `mx.compile` of the decode step (shape changes every token through
    the cache offset; graph build is only 0.3-0.5 ms per 3 layers and is hidden by
    `async_eval`); `kv_bits` (broken, see H2, and KV reads are <= 0.94 GB/token even
    at 32K); the `_DOWN_COMBINE` kernel as a speed feature (6% of the MoE block at
    N=1, 0 at N=8 on M5 Max / MLX 0.32.4).

Memory footprint for a 128 GB Mac: the 2-bit tier (88.4 GB on disk) fits with ~19 GB
for KV and transients; 3-bit (120.9 GB) does not fit under the 107.5 GiB wired limit;
4-bit (153.5 GB) needs `moe_offload` (PR #2538 measured 0.7 -> 1.6 tok/s for a
358 GB corpus; not a speed path). Items 3 and 5 are what keep long contexts inside
the 2-bit budget: at 32K context the unbounded caches add 5.6 GB and an 8192-token
drafter prefill adds up to 15 GB of transient masks per global layer.

## Hazards and open bugs

- **H1 fp32 `global_scale` promotes the residual stream** (`L490`,
  `super().__call__(x) * self.global_scale`): with a float32 scalar every later layer
  runs in fp32 and every bf16 weight (attention, lm_head) is upcast per call; measured
  4.38 -> 23.7 ms per 3-layer token, head 3.5 -> 16.3 ms (`diag_loop.json`). The
  mlx-community artifact stores `mlp.global_scale` BF16 and `mlx_vlm convert` casts
  floats to the config dtype (`convert.py:343-349`), so it is latent, but any
  conversion that keeps the reference's fp32 scalars hits it silently. Fix:
  `self.global_scale.astype(x.dtype)` in `InklingDenseMLP.__call__` or cast in
  `sanitize`. The same multiply exists in the MTP blocks (dense layers).
- **H2 `kv_bits` cannot work for Inkling**: `QuantizedKVCache.update_and_fetch`
  returns `(w, scales, biases)` tuples (`cache.py:371-377`); `InklingAttention` reads
  `k.shape[2]` (`L437`) and slices `k[:, :, j0:, :]` (`L451`) on the raw return and
  passes `cache=None` to the SDPA wrapper (`L476`), so the quantized branch
  (`base.py:433-444`) is unreachable. Expected result: `AttributeError` on the first
  decode step past `quantized_kv_start` (by code reading; not executed).
- **H3 unbounded sliding-window caches** (`L992-993` vs `L445-455`): 4.7 GB wasted
  at 32K, 18.8 GB at 128K; see opportunity 3.
- **H4 MTP**: the mlx-community checkpoints have no `model.mtp.*`; `convert --mtp`
  drops `q_mode` (#2526, non-affine drafters come out affine); with a drafter
  `_chunked_prefill_enabled` is false (`common.py:132`) so an 8192-token prompt builds
  `[1,32,8192,8192]` masks (4.3 GB + 12.9 GB tau temporaries per global layer) and
  full-vocab logits (H5); per-round snapshot + replay makes it a net loss (opportunity
  4); batch > 1 raises (`inkling_mtp.py:348-350`).
- **H5 `logits_to_keep` ignored** (`L932-955`): see opportunity 9.
- **H6 prefill peak memory**: 68-70 GB measured for the 3-layer synthetic model
  (7.7-11 GB of weights) when a layer processes 8192 tokens; with 2048-token chunks the
  transient is 1-3 GB per layer, but an unchunked prompt (drafter path, API users
  calling the model on a whole prompt) next to 88 GB of weights exceeds the wired
  limit.
- **H7 mask kernel bandwidth**: `inkling_banded_mask_v2` writes at 146 GB/s
  (gathered `proj` loads, 16 per element); 7.35 ms per global layer per 2048 chunk at
  8K context, 28.5 ms at 8192x8192.
- **H8 unsorted gather_qmm cliff**: the sorted path is taken only when
  `indices.size >= 64` (`switch_layers.py:298`); unsorted `gather_qmm` at N=2048 is
  568 ms vs 10.4 ms (q4). Fine for Inkling-Small (top-6), but any configuration or
  future code path that feeds >= 11 tokens unsorted is 18-55x slower.
- **H9 2-bit kernels under-deliver**: q2 GEMV/gather_qmm reach 237-310 GB/s vs
  411-492 for q8/bf16, so the 2-bit tier's MoE block costs the same 0.28-0.39 ms as
  the 4-bit one at N=1 and its prefill MoE is 2.6x slower (79.5 vs 30.5 ms per
  2048-token chunk).
- Dead code: `_mask_kernel` v1 (`L69-98`) is never selected from `InklingAttention`
  (`q_offset == S - LQ` always holds at `L456`).

## Open questions

1. Real-weight validation: the download of `mlx-community/Inkling-Small-mlx-2bit` was
   still in progress; the predictions (28-36 tok/s short context, 640 tok/s prefill)
   need confirming with `python -m mlx_vlm.generate` once it lands, including the
   dtype of every loaded parameter (`H1`) and peak memory.
2. Are the 3-bit/4-bit mlx-community artifacts laid out like the 2-bit one (bf16
   attention and lm_head)? The 49 tok/s on an M3 Ultra for the 4-bit tier (lablup/
   mlxcel #2479) is 61% of its 819 GB/s roofline for 10.2 GB/token, consistent with
   the same recipe.
3. Quality of q8/q6 attention and lm_head for Inkling (needed to ship opportunity 1).
4. Acceptance rate of the 8-layer Inkling MTP head on real text (sets the payoff of
   opportunity 4); only measurable with a drafter split from the bf16 source.
5. GPU contention: other benchmarks overlapped several runs (the `*_final`
   pass canaries prove it: 1 GB copy 8.5-9.0 ms vs 5.5 ms quiet). A re-run of
   `run_all4.sh` on a quiet machine (check `ps` for other MLX/torch processes first)
   would tighten the decode tables to +-10%.
6. Whether a newer MLX adds a window/bias argument to `mx.fast.scaled_dot_product_attention`
   (would make opportunity 6 a 30-line change in mlx-vlm).
