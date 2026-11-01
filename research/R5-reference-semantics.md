# R5: Inkling reference semantics, cross-implementation audit, oracles and kernel contracts

Date: 2026-10-28. Machine: MacBook Pro M5 Max 128 GB, macOS 26.5.1, venv `~/.venvs/inkling`
(mlx 0.32.4, mlx-vlm 0.7.11, transformers 5.21.0, torch 2.15.0 CPU, numpy 2.5.4).
Sources read in full are archived under `research/R5-sources/` (transformers `modeling_inkling.py`
/ `modular_inkling.py` / `configuration_inkling.py` at huggingface/transformers main; vLLM
`vllm/models/inkling/{configs.py,nvidia/*.py,nvidia/ops/*.py}` + tests; SGLang
`python/sglang/srt/models/inkling.py`, `inkling_common/{attn,sconv,moe,dense_mlp,norm}.py`,
`srt/configs/inkling.py`, JIT kernels; mlx-vlm `models/inkling/{language,config,inkling}.py` and
`speculative/drafters/inkling_mtp/*`; llama.cpp PR #25731 head `3582c5ee` `src/models/inkling.cpp`,
`ggml/src/ggml-cuda/fattn-banded.cu{,h}`, `ggml.c`/`ggml.h`, `ggml-cpu/ops.cpp`, `conversion/inkling.py`).
Real checkpoint tensor names/shapes/dtypes were read from the safetensors headers of
`thinkingmachines/Inkling-Small` (range requests only; no weights downloaded):
`research/R5-sources/hfhub/Inkling-Small.tensor_shapes.json`.

Oracles: `research/oracle/inkling_ref.py` (pure numpy float64). Tests:
`research/oracle/test_oracle_vs_transformers.py`, `research/oracle/test_oracle_vs_mlx_vlm.py`;
logs `results_transformers.log`, `results_mlx_vlm.log`, `results_mlx_vlm_tf32off.log`, JSON next to them.

## 1. Summary

* The forward pass is pinned down and the numpy oracle reproduces transformers' `InklingForCausalLM`
  (the reference) to float32 rounding on every op, in full-sequence, cached-decode and chunked-prefill
  modes, and is self-consistent to 1e-15 between stateful decode and full forward. transformers upcasts
  RMSNorm and sconv to float32 internally even with float64 weights, so agreement is 1e-7 relative, not 1e-15.
* vLLM, SGLang, llama.cpp and mlx-vlm agree with transformers on every op of the **main model**
  (attention incl. band/window/log-scaling, sconv, router, experts, shared sinks, head). mlx-vlm on CPU
  matches the oracle at 1e-6 relative; on GPU every layer is off by ~2.5e-3 because MLX 0.32.4 runs
  float32 GEMMs (M > 1) at TF32 precision on the M5 GPU **by default** (`MLX_ENABLE_TF32=0` restores
  exact fp32, after which the GPU path incl. the Metal sconv/banded-mask/route kernels matches at 8e-7) --
  not an mlx-vlm bug, but it matters for every Metal correctness test in this campaign.
* **MTP is where implementations disagree.** Four implementations, three different semantics:
  token-embedding norm (vLLM/SGLang apply the backbone `embed_norm`; transformers `MtpModel` and
  mlx-vlm feed the raw embedding), which target hidden state is chained (vLLM/SGLang/transformers:
  post-final-norm; mlx-vlm: pre-final-norm), and whether a norm precedes the shared head (mlx-vlm
  applies the target's final norm; the other three apply none). On random weights the variants pick a
  different greedy token 25-46% of the time. transformers additionally runs Inkling's sliding-window
  MTP depths (0,2,4,5,6,7) with **full** attention because `masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["hybrid_sliding"]`
  maps the sliding slot to `create_causal_mask` (bug; demonstrated in test [5]/[6]).
* The transformers config field `rms_norm_eps_moe_gate` ("RMS normalization applied inside the MoE
  router") is dead: no implementation applies a router input norm. Do not add one.
* Precision-only divergences worth knowing for a Metal port: router logits are kept in fp32 by
  vLLM/SGLang/llama.cpp but rounded to bf16 by transformers and mlx-vlm (near-tie expert selection can
  differ); mlx-vlm rounds `tau` to bf16 and double-rounds the sconv output.
* The "band" is two different things: visibility is the 512-token window on local layers and
  unbounded causal on global layers; the relative bias is non-zero only for distance < rel_extent
  (= 512 on local layers, 1024 on global layers) and is exactly zero for older keys on global layers.

## 2. Model configuration (Inkling-Small, from `config.json`)

hidden 4096, 42 layers, 32 q heads / 8 kv heads / head_dim 128 (swa identical), sliding_window_size
512, d_rel 16, rel_extent 1024, log_scaling_n_floor 128000, log_scaling_alpha 0.1, rms_norm_eps 1e-6,
use_embed_norm true, local_layer_ids = all except {5,11,17,23,29,35,41}, dense_mlp_idx 2 (layers 0-1
dense, intermediate 16384), use_sconv true, sconv_kernel_size 4, 256 routed experts, 6 per token,
2 shared, shared_expert_sink true, moe intermediate 2048, route_scale 8.0, use_gate_bias true,
gate_activation sigmoid, norm_after_topk true, use_global_scale true, q_bias/o_bias false,
logits_mup_width_multiplier 16.0, vocab 201024 padded / 200058 unpadded, final_logit_softcapping null,
model_max_length 1048576. `mtp_config`: num_nextn_predict_layers 8, chain_hidden_post_norm false,
local_layer_ids [0,2,4,5,6,7]. No `model.mtp.chain_norm` tensor exists in `mtp.safetensors`.

Name mapping between the checkpoint ("OG"), transformers and the others:
`intermediate_size` (OG, MoE width 2048) = transformers `moe_intermediate_size`; `dense_intermediate_size`
(OG) = transformers `intermediate_size`; `sconv_kernel_size` = transformers `conv_kernel_size`
(attribute_map, configuration_inkling.py:102-111).

### Checkpoint tensor layout (bf16 unless noted; `research/R5-sources/hfhub/Inkling-Small.tensor_shapes.json`)

| tensor | shape | notes |
|---|---|---|
| `model.llm.embed.weight` / `model.llm.unembed.weight` | [201024, 4096] | separate, never tied; rows >= 200058 are padding |
| `model.llm.embed_norm.weight`, `model.llm.norm.weight` | [4096] | |
| `layers.i.attn.wq_du.weight` | [4096, 4096] | q = x W^T (nn.Linear layout), no bias |
| `layers.i.attn.wk_dv.weight`, `wv_dv.weight` | [1024, 4096] | 8 kv heads x 128 |
| `layers.i.attn.wr_du.weight` | [512, 4096] | 32 heads x d_rel 16 |
| `layers.i.attn.wo_ud.weight` | [4096, 4096] | |
| `layers.i.attn.q_norm.weight`, `k_norm.weight` | [128] | per-head RMSNorm |
| `layers.i.attn.k_sconv.weight`, `v_sconv.weight` | [1024, 1, 4] | depthwise Conv1d `[C, 1, K]`; tap K-1 = current token |
| `layers.i.attn.rel_logits_proj.proj` | [16, 512] local / [16, 1024] global | `[d_rel, rel_extent]` |
| `layers.i.attn_norm.weight`, `mlp_norm.weight` | [4096] | pre-norms |
| `layers.i.attn_sconv.weight`, `mlp_sconv.weight` | [4096, 1, 4] | |
| `layers.{0,1}.mlp.w13_dn.weight` | [32768, 4096] | rows interleaved `[g0,u0,g1,u1,...]` |
| `layers.{0,1}.mlp.w2_md.weight` | [4096, 16384] | |
| `layers.{0,1}.mlp.global_scale` | [1] bf16 | dense MLP output scale |
| `layers.i>=2.mlp.gate.weight` | [258, 4096] | rows 0-255 routed, 256-257 shared sinks |
| `layers.i>=2.mlp.gate.bias` | [256] **f32** | selection-only bias (`e_score_correction_bias`) |
| `layers.i>=2.mlp.gate.global_scale` | [1] **f32** | multiplies all routing weights |
| `layers.i>=2.mlp.experts.w13_weight` | [256, 4096, 4096] | `[E, 2*2048 interleaved, hidden]` |
| `layers.i>=2.mlp.experts.w2_weight` | [256, 4096, 2048] | `[E, hidden, inter]` |
| `layers.i>=2.mlp.shared_experts.shared_w13_weight` | [2, 4096, 4096] | interleaved like w13 |
| `layers.i>=2.mlp.shared_experts.shared_w2_weight` | [2, 4096, 2048] | |
| `model.mtp.layers.k.{embed_norm,hidden_norm}.weight` | [4096] | per-depth |
| `model.mtp.layers.k.input_proj.weight` | [4096, 8192] | `[hidden | embed]` -> hidden |
| `model.mtp.layers.k.transformer_block.*` | as a dense layer | rel proj [16,512] on local depths |

De-interleave (transformers `Interleave(dim)` op, core_model_loading.py:228-247; mlx-vlm
`_split_gate_up`, inkling.py:29-36; llama.cpp `_deinterleave_w13`): `gate = w13[..., 0::2, :]`,
`up = w13[..., 1::2, :]`. transformers' `experts.gate_up_proj` is `cat(gate, up)` along the output dim
(gate = first half), `InklingExperts.forward` chunks it (modeling_inkling.py:335).

## 3. Per-op math, in forward order (transformers = reference; cross-checked)

Notation: `B` batch, `L` tokens, `H=32`, `Hkv=8`, `D=128`, `C` channels; linear layers are
`y = x @ W^T`, no biases anywhere in the text model.

### a. Embedding, embed norm, muP

```
x0 = rmsnorm(E[ids], w_embed_norm)        rmsnorm(x, w) = w * x / sqrt(mean(x^2) + 1e-6)
```
No multiplier on the input side. `embedding_multiplier` in the HF config is only an alias of
`logits_mup_width_multiplier` (configuration_inkling.py:103). Multimodal embeddings are scattered in
after the norm (towers carry their own norm). Agreement: transformers `InklingNormedEmbedding`
(modeling_inkling.py:596-602); vLLM `embed_rmsnorm` (model.py:298-329); SGLang (models_inkling.py:792-795);
mlx-vlm `InklingModel.embed` (language.py:891-895); llama.cpp `tok_norm` (inkling.cpp:604-612).

### b. Block structure, residuals, short convolutions

```
h  = x + attn_sconv( Attn( rmsnorm(x, w_attn_norm) ) )
x' = h + mlp_sconv ( MLP ( rmsnorm(h, w_mlp_norm) ) )
```
Four sconvs per layer: `k_sconv`, `v_sconv` (inside attention, on the flat K and V projections
`[B, L, Hkv*D]`, **before** the head reshape and before `k_norm`), `attn_sconv` (on the `o_proj`
output, `[B, L, 4096]`), `mlp_sconv` (on the MLP/MoE output). Every sconv is

```
sconv(x)[t, c] = x[t, c] + sum_{k=0}^{3} w[c, k] * x[t-3+k, c]      x[<0] = 0
```
depthwise per channel, kernel 4, **no bias, no activation, built-in residual**, causal left
zero-padding (K-1 = 3 zeros). Weight layout `[C, 1, 4]` with `w[c, 0, 3]` on the current token
(torch cross-correlation; vLLM `source_position = position - (W-1) + tap` with weight index `tap`,
qkvr_prep.py:320-371 and sconv.py:104-138; SGLang causal_conv1d.cuh:1-13; llama.cpp `ggml_ssm_conv`
on `concat(state, x)` with kernel ne0 = K, inkling.cpp:307-338; mlx-vlm pads K-1 then `nn.Conv1d`
with weight transposed to `[C, 4, 1]`, language.py:355-369, inkling.py:39-42).
The outer residual is then added again by the layer, so the residual stream receives
`x + (attn_out + conv(attn_out))`. All five implementations agree (transformers
`InklingShortConvolution.forward` modeling_inkling.py:509-544 and `InklingDecoderLayer.forward` 568-593;
vLLM `_sconv_add_norm` model.py:68-123 with `use_residual=True`; SGLang `ShortConvolution(use_residual=True)`
sconv.py:36-99; llama.cpp inkling.cpp:616-636; mlx-vlm language.py:843-873).

**Decode state = the last 3 inputs of each of the 4 streams** (not outputs): mlx-vlm `ArraysCache(4)`
holding `[B, 3, C]` per stream; llama.cpp `n_embd_r = (K-1) * (kw + vw + 2*n_embd)` packed `[k|v|attn|mlp]`
(inkling.cpp:47-49, 205-209); vLLM stores conv inputs at every absolute position in a paged
sliding-window cache of `kernel_size` positions (sconv_swa_attn.py:3-15, 156-167); transformers keeps the
last 4 inputs (cache_utils.py:1221-1224 with `conv_kernel_size=4`, one more than needed). Per sequence:
3 x (1024 + 1024 + 4096 + 4096) = 30720 values per layer, 1.29 M values (2.6 MB bf16) for 42 layers.
Precision: transformers runs the conv in fp32 (`.float()`, weights kept fp32 by `_keep_in_fp32_modules_strict`,
modeling_inkling.py:620) and casts once after the residual; vLLM/SGLang accumulate fp32 and round once
(qkvr_prep.py:373-374). The checkpoint weights are bf16, so "fp32 weights" carry no extra information.

### c. Attention

```
q = x Wq^T  -> [B, L, H, D]          r = x Wr^T -> [B, L, H, 16]
k = sconv_k(x Wk^T), v = sconv_v(x Wv^T)  -> [B, L, Hkv, D]        (no biases: q_bias = o_bias = false)
q = rmsnorm_head(q, w_q_norm);  k = rmsnorm_head(k, w_k_norm)      (per head over D; v is not normed)
KV cache <- (k, v)   (k stored post-norm, v post-sconv)
rel_logits[b, i, h, e] = sum_d r[b,i,h,d] * proj[d, e]            e in [0, E), E = 512 (local) / 1024 (global)
bias[b,h,i,j] = rel_logits[b,i,h, i-j]  if 0 <= i-j < E   else 0   (i, j absolute positions)
tau[i] = 1 + 0.1 * ln(max((i+1)/128000, 1))                        global layers only; 1 on local layers
s[b,h,i,j] = tau[i] * ( (1/D) * <q[b,h,i], k[b, h // (H/Hkv), j]> + bias[b,h,i,j] )
mask: j > i -> -inf;  local layers: i - j >= 512 -> -inf            (512 visible keys including self)
out = softmax_j(s) v  -> o_proj
```
* Scale is `1/head_dim`, not `1/sqrt(head_dim)` (q,k are unit-RMS; modeling_inkling.py:199-200,
  vLLM attention.py:91-92, SGLang attn.py:314, llama.cpp inkling.cpp:467/481, mlx-vlm language.py:396).
* The bias is a **learned bank of 16 bias-vs-distance profiles** (`proj`, `[d_rel, rel_extent]`),
  mixed per token and head by the 16-dim relative state `r = x Wr^T`. It is not a function of
  (R_i, R_j): only the *query* token's relative state enters; the key enters only through the distance.
  transformers `InklingRelativeLogits` (modeling_inkling.py:121-144); vLLM `RelLogitsProj` + FA4
  `score_mod` (attention.py:54-65, fa4_rel_attention.py:37-68); SGLang attn.py:59-90, 181-251;
  llama.cpp CUDA (fattn-banded.cu:110-118), CPU (ggml-cpu/ops.cpp:8988-8995), non-flash gather path with a
  zero column at index E (inkling.cpp:480-497); mlx-vlm mask kernels (language.py:69-134, 199-212).
* On local layers `rel_extent := sliding_window_size` (modeling_inkling.py:198; attention.py:131;
  attn.py:361-366; llama.cpp `rel_extent_swa` written as `sliding_window` by conversion/inkling.py:138;
  mlx-vlm language.py:392-394), and the checkpoint's `rel_logits_proj.proj` is `[16, 512]` there. So every
  visible key of a local layer has a bias; on global layers keys older than 1024 are attended with bias 0.
  The window (512) and the extent (1024) never interact: no layer has both.
* Log scaling: transformers multiplies `q` and `bias` by `tau` in fp32 (modeling_inkling.py:255-263),
  equivalent to scaling the logits. vLLM/SGLang/llama.cpp do the same (qkvr_prep.py:254-257/275-277,
  attn.py:93-97/872-879, inkling.cpp:131-143/371-374/386-388). `tau` depends on the absolute query position
  (`positions + 1`), so a decode step at position p uses `tau(p)`; cached keys are never rescaled.
* Window: visible iff `0 <= i - j < 512` in all five: transformers `kv_idx > q_idx - sliding_window`
  (masking_utils.py:88-96); vLLM FA `window_size = (511, 0)` (attention.py:150-152); SGLang
  `sliding_window_size = local_extent - 1` (attn.py:402-411); llama.cpp `LLAMA_SWA_TYPE_STANDARD`
  "visible iff pos_q - pos_k < n_swa (includes self)" (inkling.cpp:20-22); mlx-vlm `dist >= SLIDING -> -1e30`.
* GQA: head h reads kv head `h // 4` (`repeat_kv` = repeat_interleave; CUDA `ih / (n_head_q/n_head_kv)`).
* What the KV cache must retain: global layers everything; local layers the last 511 keys/values (plus
  the current token): transformers keeps `sliding_window - 1` (cache_utils.py:243-246), vLLM a
  `SlidingWindowSpec(sliding_window=512)` (attention.py:185-193), mlx-vlm slices to `L + 511`
  (language.py:445-455), llama.cpp an SWA cache of `n_swa`. Nothing about `rel_extent` needs caching
  (`rel_logits` is a per-query quantity). The sconv states (3 inputs per stream) are separate (b).
* KV-cache contents: K **after** `k_norm` and sconv, V after sconv, in model dtype. Verified in the oracle
  (`attn_sublayer_ref`) against transformers' `DynamicCache` decode and chunked prefill (test [3]).

### d. MoE (layers 2-41) and dense MLP (layers 0-1)

Router input is the `mlp_norm` output; **no extra router norm** (`rms_norm_eps_moe_gate` is unused by
every implementation).
```
logits = x @ Wg^T                       [N, 258]   (rows 0..255 routed, 256..257 shared sinks)
sel    = sigmoid(logits[:, :256]) + gate.bias        (bias affects selection only)
idx    = top-6(sel)                                  (ties -> lowest index in vLLM; see 5.)
active = [ logits[:, idx] , logits[:, 256:258] ]     8 raw logits
w      = softmax( logsigmoid(active) ) * 8.0 * gate.global_scale
       = sigmoid(active_j) / sum_j' sigmoid(active_j') * 8 * global_scale
topk_w = w[:, :6];  gamma = w[:, 6:8]
y = sum_j topk_w[j] * W2[e_j] ( silu(Wg_e x) * (Wu_e x) )  +  sum_s W2s[s] ( silu(Wgs x) * (Wus x) * gamma[s] )
```
The shared experts are always on; they do not take part in the top-k but their raw logits enter the
normalisation as "sinks" (`shared_expert_sink`), so they soak up routing mass; `norm_after_topk` is
this renormalisation over the 6+2 selected logits. `use_gate_bias` is the DeepSeek-V3 style
selection-only correction bias, `use_global_scale` the learned scalar. Routed weights are applied after the
down projection (transformers modeling_inkling.py:338; vLLM FusedMoE; llama.cpp inkling.cpp:575-576),
the sink gammas before it (modeling_inkling.py:402; vLLM `sink_silu_mul_epilogue` moe.py:337-341;
llama.cpp inkling.cpp:590-593; mlx-vlm `_swiglu_scaled` language.py:663-681) -- mathematically identical.
Experts are SwiGLU with separate gate/up (interleaved rows in the checkpoint). Agreement: transformers
`InklingTopkRouter`/`InklingExperts`/`InklingSharedExperts` (modeling_inkling.py:304-426), vLLM
`_inkling_gate_select_kernel` (moe.py:80-149), SGLang `InklingGate.forward` + `_renorm_topk_logits`
(moe.py:137-159, 399-484), llama.cpp `build_moe` (inkling.cpp:528-602), mlx-vlm `_route` (language.py:533-600, 742-793).

Dense MLP (layers 0-1 and every MTP block): `global_scale * W2 ( silu(Wg x) * (Wu x) )` with the learned
scalar `mlp.global_scale` (modeling_inkling.py:287-301; vLLM mlp.py; llama.cpp `ffn_gscale`; mlx-vlm
`InklingDenseMLP`).

### e. Final norm, head

```
h = rmsnorm(x_42, w_norm);  logits = (h / 16.0) @ unembed^T;  logits = logits[..., :200058]
```
No softcapping (`final_logit_softcapping` null). vLLM folds `1/16` into the GEMM alpha
(logits_processor.py); llama.cpp scales by `f_logit_scale = 1/16` and masks padded rows with `-inf` instead
of slicing (inkling.cpp:153-163, 652-663; logits tensor keeps 201024 rows); SGLang divides before the head
(models_inkling.py:1139-1143); mlx-vlm `_logits_from_norm` (language.py:921-930).

### f. MTP

Per depth k (8 depths; the checkpoint carries 8, a drafter uses the first `num_speculative_tokens`):
```
e  = embed_norm_k( E )                        E = embed[next_token]   (normed or raw: see below)
hN = hidden_norm_k( h_prev )
h  = input_proj_k( [hN | e] )                 mtp_hidden_states_first = true (hidden first; MtpLayer.forward)
h  = InklingBlock_k(h)                        dense MLP; attention sliding (depths 0,2,4,5,6,7, window 512,
                                              rel_extent 512) or full (depths 1,3; rel_extent 1024, log scaling);
                                              its own 4 sconv states and KV cache
h  = chain_norm(h)  if chain_hidden_post_norm (false for Inkling-Small; no tensor)
logits_k = (h / 16) @ unembed^T [:200058]     shared embed/unembed with the target
h_prev <- h  for depth k+1 (the raw block output when chain_hidden_post_norm is false)
```
Drafting (all implementations): depth k predicts token t+k+2 from the target's hidden state at t and
token t+1; the drafted token is fed as the next depth's `next_token` with position +1. Verification is the
standard draft-then-verify loop with the target: vLLM EAGLE-style proposer (`spec_step_idx` selects depth,
mtp.py:238-258); SGLang one `InklingForConditionalGenerationMTP` instance per depth
(`--enable-multi-layer-eagle` required for banded depths, models_inkling.py:1792-1798) with the sconv
"intermediate windows" committed after verify for the accepted prefix (sconv.py:222-244,
`commit_conv_state_after_mtp_verify`); mlx-vlm `InklingMTPDraftModel` snapshots/restores its caches around
each draft block and re-runs the accepted tokens (inkling_mtp.py:252-325, language.py:963-986); transformers
`MtpModel` (modeling_layers.py:361-600, assisted decoding) rolls `input_ids` by one per depth. llama.cpp has no MTP
(conversion skips `model.mtp.*`, conversion/inkling.py:16). Acceptance is verified by the target forward on
the drafted tokens (greedy: token equality; sampling: standard rejection), nothing Inkling-specific.

Where they disagree (evidence in section 5): which `E` (raw vs backbone-normed embedding), which `h_prev`
for depth 0 (target's pre- vs post-final-norm state), and whether a norm precedes the head.

## 4. Discrepancy table

| # | Aspect | transformers | vLLM | SGLang | llama.cpp | mlx-vlm | Severity |
|---|---|---|---|---|---|---|---|
| 1 | MTP depth-0 input hidden | post-final-norm (`MtpModel.forward`: `last_hidden_states` = model output after `norm`) | post-final-norm (model.py:361-367 returns `norm(...)`; proposer feeds it) | post-final-norm (models_inkling.py:1145-1158 passes the normed hidden as `hidden_states_before_norm`; test_inkling_aux_hidden_states.py:64-65,114) | n/a | **pre-final-norm** (language.py:947,963-972 `skip_final_norm=True`; inkling_mtp.py:213) | logic |
| 2 | MTP token embedding | **raw** (modeling_layers.py:412-416 `embed_norm = nn.Identity()`) | backbone `embed_norm` applied (mtp.py:163-173, 203-236; comment: raw drops MTP1 acceptance 0.85 -> 0.70) | backbone `embed_norm` applied (models_inkling.py:1686-1689, 1737-1739) | n/a | **raw** (inkling_mtp.py:152, 211 uses `inner.embed_tokens`, the un-normed `nn.Embedding`) | logic |
| 3 | Norm before MTP head | none (`use_post_norm=False`, `_project_to_logits`, modeling_layers.py:386-390, 425-434) | none (mtp.py:315-344) | none (models_inkling.py:1759-1765) | n/a | **target's final norm** (inkling_mtp.py:197-201; split.py:11,36,45 copies `model.llm.norm.weight`) | logic |
| 4 | Sliding mask on `hybrid_sliding` MTP depths / generate | **full causal** (masking_utils.py:1488 maps `"hybrid_sliding" -> {"sliding_attention": create_causal_mask}`; MtpModel.create_masks_for_mtp_layer and create_masks_for_generate use it; main-model forward builds its own masks, modeling_inkling.py:713-717) | window 512 | window 512 (`mtp_local_extent`) | n/a | window 512 | bug (transformers) |
| 5 | Router logits precision | bf16 (`F.linear` in model dtype, modeling_inkling.py:360) | fp32 (`_linear_with_fp32_out`, moe.py:60-77) | fp32 (moe.py:101-122) | fp32 (`GGML_PREC_F32_PEDANTIC`, inkling.cpp:534-535) | bf16 (language.py:798-801) | precision (near-tie routing) |
| 6 | Router top-k tie-break | torch.topk (unspecified) | lowest index (moe.py:119-130) | gate_topk | argsort_top_k | lowest *lane* (`j % 32`), language.py:559-568 | precision edge |
| 7 | `tau` precision | fp32, product rounded (modeling_inkling.py:255-263) | fp32 (qkvr_prep.py:254-257, 275-277) | fp32 | fp32 input tensor (inkling.cpp:131-143) | **tau cast to model dtype** (bf16) before use (language.py:468-473) | precision (<= 2^-8 rel on logits past 128k) |
| 8 | sconv rounding | fp32 conv + residual, one cast (modeling_inkling.py:509-543) | fp32 accumulate, one cast (qkvr_prep.py:373-374; sconv.py:138-148) | fp32 accumulate, one cast (causal_conv1d.cuh:1-13) | F32 | conv output rounded to model dtype **before** the fp32 residual add (language.py:367-368; kernel 272-281) | precision |
| 9 | Router input norm (`rms_norm_eps_moe_gate`) | field exists (configuration_inkling.py:62-63,146), unused | none | none | none | none | dead config field |
| 10 | Conv state size | last 4 inputs (cache_utils.py:1221-1224) | positions window of `kernel_size` (sconv_swa_attn.py:158-159) | K-1 (`[pool, W-1, D]`) | K-1 (inkling.cpp:48) | K-1 (`[B, 3, C]`) | none (3 suffice) |
| 11 | Local-layer KV retention | 511 (cache_utils.py:243-246) | 512-window spec | SWA pool, window 511 | `n_swa` 512 | slice to L+511 | none |
| 12 | Padded vocab | slice to 200058 | slice (`org_vocab_size`) | slice | `-inf` mask on rows >= 200058 (inkling.cpp:153-163, 659-661) | slice | none |
| 13 | MTP support | generic `MtpModel` (with #2, #4) | yes | yes (per-depth instances) | **none** | yes (with #1-#3) | feature |
| 14 | fp32 GEMM precision on the Mac GPU | n/a | n/a | n/a | n/a | MLX 0.32.4 on M5: TF32-like for M>1 GEMMs (see 6.3) | validation hazard |
| 15 | `SGLang hidden_states_before_norm` | - | - | name means "before the muP division"; it is post-norm | - | - | naming trap |

All other ops (embedding+norm, block order, four sconvs with residual, tap convention, q/k norm,
`1/D` scale, bias support `0 <= dist < E` with E=512 local/1024 global, window 512 inclusive of self, GQA
`h // 4`, `tau` formula on global layers only, router/sink/global-scale math, SwiGLU experts with
interleaved checkpoint rows, dense MLP global_scale, final norm / `/16` / unpadded slice) are identical
across the five implementations by code reading, and transformers vs mlx-vlm(CPU) vs oracle by test.

## 5. Oracle and test results

`research/oracle/inkling_ref.py` exports `rmsnorm_ref, sconv_ref, rel_logits_ref, rel_bias_ref,
log_scaling_tau_ref, attention_ref, attn_sublayer_ref, router_ref, experts_ref, shared_experts_ref,
moe_ref, dense_mlp_ref, decoder_layer_ref, model_forward_ref, mtp_step_ref` (+ layout helpers
`deinterleave_w13`, `w13_to_transformers_gate_up`, `sconv_weight_2d`). Parameters are keyed by the
transformers state_dict names so `{k: v.double().numpy() for k, v in model.state_dict().items()}` is the
input. `mtp_step_ref(variant=...)` implements the three MTP semantics ("vllm"/"sglang", "transformers", "mlx").

Tiny config used by both tests: hidden 64, 2 q heads / 1 kv head (GQA 2:1), head_dim 32, d_rel 4,
rel_extent 8, window 4, 8 routed experts top-2 + 2 shared, 3 layers (sliding, global, sliding), layer 0
dense (96) / layers 1-2 MoE (32), route_scale 8, log scaling n_floor 4 (active from position 3),
alpha 0.1, mup 16, vocab 64 / unpadded 60, 1 MTP depth (local). Every parameter (norm weights, global
scales, correction bias, conv taps, rel proj) is random and non-trivial (`random_fill`, seed 1234).
Inputs: B=2, L=12 random tokens (seed 7); L > window and L > rel_extent so both band edges and the
log-scaling floor are exercised.

### 5.1 transformers (float64 model, eager attention and eager experts) vs oracle

```
source ~/.venvs/inkling/bin/activate && cd ~/code/inkling-contrib/research/oracle && python test_oracle_vs_transformers.py
```
Output (`results_transformers.log`):
```
[1] full-sequence forward (transformers float64 vs oracle float64), B=2 L=12
  embedding + embed_norm                                     max_abs=3.751e-07  max_rel=9.458e-08
  layer 0 output (hybrid_sliding, dense)                     max_abs=8.161e-07  max_rel=1.268e-07
  layer 1 output (hybrid, sparse)                            max_abs=4.015e-06  max_rel=1.666e-07
  final norm output (last_hidden_state)                      max_abs=1.718e-06  max_rel=4.195e-07
  logits (sliced to unpadded vocab)                          max_abs=6.908e-07  max_rel=4.217e-07
[2] per-op checks (isolated module vs oracle)
  sconv_ref (prefill, no state)                              max_abs=2.670e-07  max_rel=6.226e-08
  sconv_ref (chunked with carried state) vs prefill          max_abs=2.670e-07  max_rel=6.226e-08
  rel_bias_ref (global layer, q offset 3, 8 keys)            max_abs=4.441e-16  max_rel=1.295e-16
  router_ref: identical expert sets for all 9 tokens: True
  router_ref topk weights (index-sorted)                     max_abs=0.000e+00  max_rel=0.000e+00
  router_ref shared gammas                                   max_abs=2.220e-16  max_rel=6.266e-17
  moe_ref (router + experts + shared)                        max_abs=6.078e-07  max_rel=4.311e-08
  dense_mlp_ref (global_scale)                               max_abs=4.441e-16  max_rel=1.481e-16
  attention_ref full (GQA 2:1 + bias + causal)               max_abs=1.146e-07  max_rel=5.866e-08
  attention_ref sliding window 4                             max_abs=2.109e-07  max_rel=8.694e-08
[3] cached decode (transformers DynamicCache prefill 7 + 5 x 1 token) vs oracle stateful decode vs full forward
  transformers cached decode vs transformers full forward    max_abs=4.937e-07  max_rel=3.014e-07
  oracle stateful decode vs oracle full forward              max_abs=1.887e-15  max_rel=1.152e-15
  oracle stateful decode vs transformers cached decode       max_abs=6.908e-07  max_rel=4.217e-07
  layer 0 (hybrid_sliding): oracle retains 3 keys, sconv states [3, 3, 3, 3]
  layer 1 (hybrid): oracle retains 12 keys, sconv states [3, 3, 3, 3]
  layer 2 (hybrid_sliding): oracle retains 3 keys, sconv states [3, 3, 3, 3]
  chunked prefill (5,3,1,1,1,1): transformers vs full        max_abs=2.587e-07  max_rel=1.579e-07
  chunked prefill (5,3,1,1,1,1): oracle vs full              max_abs=2.387e-15  max_rel=1.457e-15
[4] MTP depth step (torch reference built from transformers modules, vLLM/SGLang semantics) vs mtp_step_ref
  mtp_step_ref chain hidden                                  max_abs=1.459e-06  max_rel=2.060e-07
  mtp_step_ref logits                                        max_abs=4.840e-07  max_rel=1.284e-07
[5] transformers MtpModel (generic MTP head, Inkling weight mapping) vs mtp_step_ref
  [local depth] MtpModel: use_post_norm=False use_shared_post_norm=False embed_norm=Identity mtp layer_types=['hybrid_sliding']
  [local depth] MtpModel attention mask row for query 4 (window 4): visible keys = [0, 1, 2, 3, 4]
  [local depth] MtpModel chain hidden vs mtp_step_ref variant=transformers max_abs=3.288e+00  max_rel=4.692e-01
  [local depth] MtpModel logits vs mtp_step_ref variant=transformers max_abs=1.106e+00  max_rel=3.426e-01
  [local depth] MtpModel chain hidden vs mtp_step_ref variant=vllm max_abs=2.510e+00  max_rel=3.955e-01
  [local depth] MtpModel logits vs mtp_step_ref variant=vllm max_abs=1.292e+00  max_rel=4.208e-01
  [local depth] MtpModel chain hidden vs mtp_step_ref variant=mlx max_abs=3.288e+00  max_rel=4.692e-01
  [local depth] MtpModel logits vs mtp_step_ref variant=mlx  max_abs=1.453e+00  max_rel=7.678e-01
  [local depth] MtpModel block with sliding mask forced vs variant=transformers max_abs=9.487e-07  max_rel=1.354e-07
  [global depth] MtpModel: use_post_norm=False use_shared_post_norm=False embed_norm=Identity mtp layer_types=['hybrid']
  [global depth] MtpModel chain hidden vs mtp_step_ref variant=transformers max_abs=6.840e-07  max_rel=1.382e-07
  [global depth] MtpModel logits vs mtp_step_ref variant=transformers max_abs=4.151e-07  max_rel=1.418e-07
  [global depth] MtpModel chain hidden vs mtp_step_ref variant=vllm max_abs=1.979e+00  max_rel=3.738e-01
  [global depth] MtpModel logits vs mtp_step_ref variant=vllm max_abs=6.789e-01  max_rel=2.293e-01
  [global depth] MtpModel chain hidden vs mtp_step_ref variant=mlx max_abs=6.840e-07  max_rel=1.382e-07
  [global depth] MtpModel logits vs mtp_step_ref variant=mlx max_abs=1.404e+00  max_rel=6.841e-01
[6] generate() path: first-step logits vs plain forward (masks from create_masks_for_generate)
  create_masks_for_generate['sliding_attention'] row for query 8 (window 4): visible keys = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  forward with generate-style mask dict vs plain forward logits max_abs=1.656e+00  max_rel=1.011e+00
  generate() first-step logits vs oracle last-position logits max_abs=4.298e-07  max_rel=2.832e-07
WORST max_abs over all same-semantics checks: 4.015e-06  (float32 internals in transformers norms/sconv)
OK
```
Reading: every op matches at float32 rounding (transformers' `InklingRMSNorm` and sconv compute in
float32 regardless of parameter dtype; the oracle's own decode/prefill consistency is 1e-15). [5] shows the
`MtpModel` block equals the oracle once the sliding mask is supplied (9.5e-7) and equals it outright on a
global depth (6.8e-7), isolating the `hybrid_sliding` mask bug; [6] shows `create_masks_for_generate`
produces the same full mask under the `"sliding_attention"` key (feeding it to the model changes the logits
by 100%), while `generate()` with the default `DynamicCache` does not take that path (it is only used for
compileable caches, generation/utils.py:723-728; Inkling's linear-attention cache layers are not
compileable, cache_utils.py:1251) and therefore still matches the oracle (4.3e-7).

### 5.2 mlx-vlm (float32) vs oracle, GPU (Metal kernels) and CPU

```
python test_oracle_vs_mlx_vlm.py
```
Output (`results_mlx_vlm.log`):
```
[gpu] mlx-vlm LanguageModel float32 vs oracle float64 (B=2, L=12)
  gpu: full forward logits                                               max_abs=9.176e-03  max_rel=5.602e-03
  gpu: embedding + embed_norm                                            max_abs=3.708e-07  max_rel=9.348e-08
  gpu: layer 0 output (hybrid_sliding, dense)                            max_abs=1.647e-02  max_rel=2.560e-03
  gpu: layer 1 output (hybrid, sparse)                                   max_abs=4.837e-02  max_rel=2.007e-03
  gpu: layer 2 output (hybrid_sliding, sparse)                           max_abs=9.740e-02  max_rel=2.837e-03
  gpu: cached decode (7 + 5x1) logits                                    max_abs=9.927e-03  max_rel=6.060e-03
  gpu: chunked prefill (5,3,1,1,1,1) logits                              max_abs=9.231e-03  max_rel=5.635e-03
[cpu] mlx-vlm LanguageModel float32 vs oracle float64 (B=2, L=12)
  cpu: full forward logits                                               max_abs=1.355e-06  max_rel=8.270e-07
  cpu: embedding + embed_norm                                            max_abs=4.068e-07  max_rel=1.026e-07
  cpu: layer 0 output (hybrid_sliding, dense)                            max_abs=2.362e-06  max_rel=3.671e-07
  cpu: layer 1 output (hybrid, sparse)                                   max_abs=1.283e-05  max_rel=5.324e-07
  cpu: layer 2 output (hybrid_sliding, sparse)                           max_abs=2.238e-05  max_rel=6.520e-07
  cpu: cached decode (7 + 5x1) logits                                    max_abs=1.153e-06  max_rel=7.038e-07
  cpu: chunked prefill (5,3,1,1,1,1) logits                              max_abs=1.519e-06  max_rel=9.270e-07
[cpu] mlx-vlm InklingMTPDraftModel vs mtp_step_ref variants
  cpu: mlx drafter chain hidden vs oracle variant=mlx (pre-norm hidden, raw embed) max_abs=3.285e-06  max_rel=3.869e-07
  cpu: mlx drafter logits vs oracle variant=mlx (final_norm before head) max_abs=9.150e-07  max_rel=4.178e-07
  cpu: mlx drafter logits vs oracle variant=transformers (post-norm hidden, raw embed, no head norm) max_abs=2.744e+00  max_rel=6.875e-01
  cpu: mlx drafter logits vs oracle variant=vllm (post-norm hidden, backbone embed_norm, no head norm) max_abs=2.521e+00  max_rel=6.690e-01
  greedy-token agreement of the mlx drafter with: mlx-semantics 1.00 | transformers-semantics 0.67 | vllm-semantics 0.54
[gpu] ... (same at GPU precision: mlx-semantics 0.96 | transformers 0.67 | vllm 0.58)
  (vllm vs transformers semantics greedy agreement on the same inputs: 0.75)
```
Reading: the mlx-vlm main model is logically identical to the reference (CPU: 1e-6 relative on full
forward, cached decode -- which exercises the Metal sconv-decode and route kernels on GPU -- and chunked
prefill). The GPU-only 2.5e-3 deviation was bisected to MLX's float32 GEMM (section 6.4): the plain
`qkvr_proj` Linear already differs GPU-vs-CPU by 8.7e-4, the banded-mask Metal kernel equals its numpy
fallback bit-for-bit, and the Metal SDPA on identical inputs differs by 1e-3. With `MLX_ENABLE_TF32=0`
the GPU run matches the oracle like the CPU run (`results_mlx_vlm_tf32off.log`):
```
MLX_ENABLE_TF32=0 python test_oracle_vs_mlx_vlm.py
  gpu: full forward logits                                               max_abs=1.302e-06  max_rel=7.949e-07
  gpu: layer 0 output (hybrid_sliding, dense)                            max_abs=2.035e-06  max_rel=3.163e-07
  gpu: layer 1 output (hybrid, sparse)                                   max_abs=1.527e-05  max_rel=6.334e-07
  gpu: layer 2 output (hybrid_sliding, sparse)                           max_abs=2.225e-05  max_rel=6.482e-07
  gpu: cached decode (7 + 5x1) logits                                    max_abs=1.399e-06  max_rel=8.543e-07
  gpu: chunked prefill (5,3,1,1,1,1) logits                              max_abs=1.302e-06  max_rel=7.949e-07
  gpu: mlx drafter chain hidden vs oracle variant=mlx                    max_abs=3.084e-06  max_rel=3.632e-07
  gpu: mlx drafter logits vs oracle variant=mlx                          max_abs=6.718e-07  max_rel=3.067e-07
```
The drafter implements the "mlx" semantics exactly (4e-7) and disagrees with the two other semantics on
33-46% of greedy tokens on this random model; vLLM-vs-transformers semantics (embedding norm only) disagree on 25%.

## 6. Kernel contracts (testable against the oracle)

### 6.1 Banded relative-bias attention (Metal/MLX), prefill and decode

Inputs (per layer; `B` sequences, `Lq` new queries, `Lk` keys incl. cache):
* `q` `[B, H, Lq, D]` model dtype, **already** `q_norm`'ed; the kernel applies `tau` itself (pass `q_pos`
  and `(n_floor, alpha)` or a precomputed fp32 `tau[Lq]`; `tau = 1` on local layers).
* `k` `[B, Hkv, Lk, D]` (post-sconv, post-`k_norm`), `v` `[B, Hkv, Lk, D]` (post-sconv), from the KV cache.
* either `rel_logits` `[B, Lq, H, E]` (= `r @ proj`, fp32 or bf16) or, better, `r` `[B, Lq, H, 16]` plus
  `proj` `[16, E]` so the bias is formed in-register (16 MACs per score; mlx-vlm's mask kernel already does
  this but materialises an `[B, H, Lq, Lk]` mask, which is O(Lk) per query and untenable at 1M context).
* `q_pos[Lq]`, `k_pos[Lk]` absolute positions (contiguous in practice: `k_pos = arange(Lk)`,
  `q_pos = Lk - Lq + arange(Lq)`; the decode case is `Lq = 1`).
* `E` (= 512 local / 1024 global), `window` (512 local / 0 global), `scale = 1/D`.
Semantics: for each `(b, h, i, j)` with `dist = q_pos[i] - k_pos[j]`:
```
visible  = dist >= 0 and (window == 0 or dist < window)
bias     = (0 <= dist < E) ? rel_logits[b, i, h, dist] : 0           (zero bias beyond E on global layers)
s        = tau[i] * (scale * dot(q[b,h,i], k[b, h // (H/Hkv), j]) + bias)
out[b,h,i] = sum_j softmax_j(s) * v[b, h // (H/Hkv), j]        (softmax over visible j, fp32)
```
Output `[B, H, Lq, D]` (consumer transposes to `[B, Lq, H*D]` for `o_proj`). Visibility is NOT
"window + rel_extent": a local layer sees exactly the 512 most recent positions (bias on all of them); a
global layer sees all `i+1` keys and the bias is zero for the ones older than 1024. Precision: scores,
max/sum and the P@V accumulation in fp32; `bias` may be bf16 (all implementations store it in model
dtype); `tau` in fp32 (do not round it to bf16). GQA by integer division of the head index. Masked
scores are `-inf` (or skipped), never a large finite number added to a scaled score.
Oracle: `attention_ref(q, k, v, rel_bias_ref(r, proj, q_pos, k_pos), q_pos, k_pos, 1/D, sliding_window, log_scaling)`
(`attn_sublayer_ref` covers the whole sublayer incl. cache update). Suggested test cases: Lq in
{1, 8, 37}, Lk up to 3*window, both layer kinds, `tau` active (`n_floor` small), GQA 4:1, bf16 inputs
compared at <= 1e-2 relative and fp32 inputs at <= 1e-5 (on CPU or with `MLX_ENABLE_TF32=0`).

### 6.2 Short-convolution decode kernel

Inputs: `x` `[B, L, C]` (L small, typically 1-8; the stream input in model dtype), `state` `[B, 3, C]`
(the previous 3 inputs of this stream; bf16 is lossless since the inputs are bf16 values), `w` `[C, 4]`
(= checkpoint `[C, 1, 4]` squeezed; `w[:, 3]` on the current token). Optional: `residual` `[B, L, C]`
(the layer's residual stream, for the attn/mlp streams) and a following RMSNorm weight.
```
xp = cat(state, x)  along time                                   # [B, 3+L, C]
y[t] = x[t] + sum_{k=0}^{3} w[:, k] * xp[t + k]                  # fp32 accumulate, one rounding
new_state = xp[L:]                                               # last 3 inputs
(optional) h = residual + y ;  n = rmsnorm(h, w_norm)           # the fusion vLLM/SGLang use
```
Four streams per layer with C = 1024, 1024, 4096, 4096 (K, V, attn-out, mlp-out). K/V streams run before
`k_norm` and the KV-cache write, so the natural Mac fusion is `{k/v sconv + q/k norm + tau + KV write}`
(vLLM `fused_qkvr_prep`) and `{sconv + residual add + next rmsnorm}` for the two hidden-width streams.
State per sequence: 30720 values per layer. Oracle: `sconv_ref(x, w, state) -> (y, new_state)`; a prefill
of length L then a decode must equal `sconv_ref` on the concatenation (test [2]/[3]).

### 6.3 MoE route kernel

Inputs: `logits` `[N, 258]` **fp32** (compute `x @ Wg^T` with an fp32 output, as vLLM/SGLang/llama.cpp do;
do not round to bf16), `corr_bias[256]` fp32, `global_scale` fp32 scalar, `route_scale = 8`, `K = 6`,
`S = 2`.
```
sel = sigmoid(logits[:, :256]) + corr_bias ;  idx = top-6(sel), ties -> lowest index
active = [logits[n, idx[n, :]], logits[n, 256:258]]            # 8 raw logits
lp = -softplus(-active) ;  w = exp(lp - logsumexp(lp)) * 8 * global_scale
topk_w = w[:, :6] ; gamma = w[:, 6:8]
```
Outputs: `idx` `[N, 6]` int32, `topk_w` `[N, 6]`, `gamma` `[N, 2]` (fp32 or model dtype). `topk_w`
multiplies each routed expert's down-projection output; `gamma[s]` multiplies shared expert `s`'s SwiGLU
activation before its down-projection (or its output; same math). Oracle: `router_ref` (returns the
index set; compare index-sorted) and `moe_ref` for the full sparse MLP.

### 6.4 Validation hazard on this Mac

MLX 0.32.4 on the M5 Max runs float32 matmuls with M > 1 at TF32 precision on the GPU by default
(`libmlx.dylib` env var `MLX_ENABLE_TF32`; unset and `=1` behave the same, `=0` disables it): 64x64x64 and
2048^3 fp32 GEMMs are 7-9e-4 relative off a float64 reference (CPU: 2e-7), while M = 1 GEMV (5e-7),
elementwise and reductions are exact. Probe:
```
MLX_ENABLE_TF32=<unset> matmul 64x64x64: gpu fp32 vs f64 rel=7.14e-04   256x4096x256: 7.79e-04   1x4096x4096: 4.49e-07
MLX_ENABLE_TF32=1       matmul 64x64x64: gpu fp32 vs f64 rel=7.14e-04   256x4096x256: 7.79e-04   1x4096x4096: 4.49e-07
MLX_ENABLE_TF32=0       matmul 64x64x64: gpu fp32 vs f64 rel=2.05e-07   256x4096x256: 8.66e-07   1x4096x4096: 4.49e-07
```
Any fp32 "exact" comparison of a Metal kernel against the oracle must therefore set `MLX_ENABLE_TF32=0`
(or run on CPU / M = 1); with it the whole mlx-vlm GPU path matches the oracle at 8e-7 (section 5.2). bf16
inference is unaffected in practice (bf16 inputs carry 8 mantissa bits), but benchmark numbers taken with
fp32 activations on the GPU are TF32 numbers unless the variable is set.

## 7. Open questions

1. **Which MTP semantics were trained?** The vendor's `mtp_model.py` is not in the public repos
   (only config/tokenizer files are). vLLM reports measured MTP1 acceptance 0.85 with the backbone
   `embed_norm` applied vs 0.70 raw (mtp.py:163-167), which also implies the post-final-norm hidden and no
   head norm were what they validated. mlx-vlm's three choices (raw embedding, pre-norm hidden, target's
   final norm before the head) have not been validated against anything we can see; its drafter should be
   benchmarked for acceptance rate against the vLLM semantics once weights are local (the oracle
   `mtp_step_ref(variant=...)` gives both). transformers' `MtpModel` (raw embedding) likely inherits the
   DeepSeek convention rather than a measurement.
2. **transformers bug reports to file**: (a) `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["hybrid_sliding"]`
   should use `create_sliding_window_causal_mask` (affects `MtpModel` local depths now, and `generate()` with
   any compileable cache); (b) `rms_norm_eps_moe_gate` is documented but unused; (c) `MtpModel` strips
   the embedding norm for Inkling, contradicting vLLM/SGLang.
3. **Router logits precision**: transformers (the reference) and mlx-vlm round the gate logits to bf16;
   the vendor-backed CUDA stacks keep fp32. On exact ties mlx-vlm's lane-based tie-break can also pick a
   different expert than lowest-index. Recommend fp32 logits + lowest-index ties for the Mac kernel.
4. **llama.cpp MMA banded path**: only the reference warp-per-row CUDA kernel (`fattn-banded.cu`) and
   the CPU path were read; the tensor-core path (`ggml_cuda_flash_attn_ext_banded_use_mma` in `fattn.cu`)
   was not reviewed and llama.cpp has no Metal implementation of `GGML_OP_FLASH_ATTN_EXT_BANDED`
   (only CPU and CUDA backends reference it at that SHA), i.e. llama.cpp on Mac falls back to the gather
   path (`ggml_get_rows` of `rel` with a zero column, `inkling.cpp:480-511`).
5. **MLX TF32 default** (resolved): `MLX_ENABLE_TF32` is on by default on the M5 and also affects
   `mx.fast.scaled_dot_product_attention` for fp32 inputs (1e-3 on identical inputs; exact with `=0`).
   Open: whether this default is intended by MLX for all M5 GPUs and whether a Python-level switch exists.
6. The mlx-vlm sconv double rounding and bf16 `tau` are small but systematic; worth a one-line fix
   upstream when touching those kernels.
