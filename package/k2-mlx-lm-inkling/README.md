# K2: a first-class Inkling text model for mlx-lm

Status: implemented and validated locally; nothing pushed. Branch `inkling` in
`upstream/mlx-lm` (base `1f7057c`, head = the four-commit series in `k2-mlx-lm-inkling.patch`).

## What

`mlx_lm/models/inkling_mm_model.py` (+ a `key_distances()` helper on `RotatingKVCache` /
`BatchRotatingKVCache` in `cache.py`) makes Thinking Machines' Inkling-Small run in mlx-lm:
`mlx_lm.generate`, `mlx_lm.chat`, `mlx_lm.server` (continuous batching, verified with
`BatchGenerator` on a tiny model: identical tokens and logprobs to single-sequence runs),
prompt-cache save/load, `mlx_lm.convert` from the HF source layout, and LM Studio's MLX engine
(which loads mlx-lm model files by `model_type`). It loads the existing
`mlx-community/Inkling-Small-mlx-{2,3,4}bit` checkpoints directly (`model.llm.*` paths).

Per-token design (decode, one layer): one fused q/k/v/r projection; one Metal kernel for the
k|v short convolution (fp32 accumulate, single rounding, state update included) and one each for
the post-attention and post-MLP convolutions with the residual folded in; per-head q/k RMS norm;
`RotatingKVCache(512)` on the 35 sliding layers (bounded memory) and `KVCache` on the 7 global
ones; the relative bias bank stored reversed so a token's bias is a slice of `r @ proj` on global
layers and one gather with a slot index shared by all sliding layers; `mx.fast.scaled_dot_product_attention`
with the additive bias; fp32 router GEMV + one Metal kernel for top-6 selection and the
log-sigmoid softmax over the 6 routed + 2 shared logits; the 2 shared experts appended to the 256
routed ones so one `SwitchGLU` (`gather_qmm`) call serves all 8; the log-scaling factor only
applied once positions reach 128000. Prefill builds the banded additive mask (window / bias extent
/ causal) with a 20-line Metal kernel; every kernel has a pure-MLX path for non-Metal devices and
the tests check kernel and fallback against numpy.

Measured per-token dispatch (tiny-width 42-layer models, same machine state): graph build
2.0 ms vs 3.0 ms for mlx-vlm; the short-conv kernel removed ~3.5 ms of launches per token and
the route kernel ~4.5 ms (bench/logs/K2-dispatch-profile.log, K2-router-dispatch.log).

## Deviations from research/R3-mlx-lm.md section 4 (and why)

- File name: `inkling_mm_model.py` (the checkpoints' `model_type`) per CONTRIBUTING, instead of
  `inkling.py` plus a remap; a flat `inkling_text` config is accepted too.
- Sliding cache: `RotatingKVCache` instead of the proposed shifting-window cache. Both were
  implemented and benchmarked (bench/logs/K2-cache-bench*.log): at Inkling-Small shapes the
  rotating cache was never slower (bests 1.63-1.80 vs 1.74-1.86 ms per decode step over the 35
  layers), its batched variant already exists upstream, and the shifting cache's selling point
  (trimming past the window for speculative decoding) does not materialize while the conv states
  live in an `ArraysCache`, which is never trimmable. The rotated slot order is handled by the
  new `key_distances()` and tested across rotations.
- Conv states: `ArraysCache(3)` (k|v share one conv), not 4.
- Two more small kernels than R3 planned (short conv for <= 8 tokens, expert selection): the
  dispatch profile showed the compiled-op versions costing more launches per token than the rest
  of the layer; each kernel is <= 50 lines, has a pure-MLX fallback and a test.
- The bias bank parameter is stored reversed (`rel_logits_proj.proj_reversed`), a load-time
  transform like the fused projections; converted checkpoints carry the transformed names.
- The router keeps fp32 logits (vLLM / SGLang / llama.cpp do; transformers and mlx-vlm round
  them to bf16), so greedy tokens can differ from mlx-vlm on near-tie routing decisions; the
  equivalence section quantifies this with a teacher-forced comparison and a router audit.

## Numerics caveats

- The router's fp32 logits hold at decode (a GEMV is exact); at prefill the fp32 GEMM runs at
  TF32 precision on the M5 by default (~7.5e-4 relative) unless `MLX_ENABLE_TF32=0` is set before
  importing MLX, so near-tie expert selections at prefill are TF32-resolved.
  The repo tests therefore compare the tiny fp32 model at 1e-2 on the GPU and at 1e-5 on the CPU
  device (`test_cpu_exactness`); with TF32 off the GPU comparisons hold at 1e-6 too.
- The short-conv kernel and the compiled taps agree bit for bit after bf16/fp16 rounding; their
  fp32 accumulation orders differ at 5e-8.
- Prefill materializes a `[B, 32, L, S]` bf16 mask per layer: 4.6 GB transient per global layer
  at 32K context with 2048-token chunks (9 GB at 64K, 18 GB at 128K); `prefill_step_size` bounds it.

## Limitations

- `--kv-bits` is a no-op (the per-layer `CacheList` has no `to_quantized`), as in mlx-vlm; the
  single-stream generate path warns once about it.
- Not trimmable (`ArraysCache`), so no speculative decoding and no LRU prompt-cache trimming;
  prefix reuse still works.
- Text only: the vision and audio towers of the multimodal checkpoint are dropped by `sanitize`.
- The stock CLI's one-shot `mx.eval` of 82 GB: see the CLI finding below.

## How to run

```console
source ~/.venvs/inkling/bin/activate          # mlx_lm resolves to upstream/mlx-lm (editable)
mlx_lm.generate --model mlx-community/Inkling-Small-mlx-2bit -p "The secret to baking a good cake is" -m 200
python -m mlx_lm.server --model mlx-community/Inkling-Small-mlx-2bit
```

Loading the 82 GB checkpoint: the stock CLI's one-shot `mx.eval` of all parameters worked on this
machine (37 s wall for load + 200 tokens; mlx-vlm's CLI hit the Metal watchdog for M1 under
memory pressure); `mlx_lm.load(..., lazy=True)` plus a chunked eval (`bench/K2/gpulock.py:chunked_eval`,
~4 GiB per `mx.eval`) takes 1-4 s + 7-11 s and is what every benchmark here used. Decoding from a
raw Python loop needs `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])`
(as `stream_generate` does): without it the driver pages the weights and decode drops below 1 tok/s.

Tests (in `upstream/mlx-lm`): `python -m unittest tests.test_inkling tests.test_models.TestModels.test_inkling_mm_model`;
`uvx pre-commit run --all-files`.

## Measured numbers (M5 Max 128 GB, mlx 0.32.4, mlx-community/Inkling-Small-mlx-2bit; `load1` = 1-min load average)

### Equivalence with mlx-vlm 0.7.11 (the reference, run from the untouched wheel in `~/.venvs/inkling-base`)

Greedy tokens are not expected to match: on this model implementations that differ only by bf16
rounding diverge within a few tokens (routing near-ties are amplified layer by layer), so the
evidence is a teacher-forced comparison: the reference's greedy tokens are fed to both models and
the per-step output distributions are compared, against the noise floor of mlx-vlm compared with
itself under a different prefill chunk size (2048 vs 128). Full table: `bench/logs/K2-equivalence.md`.

| run | prompt tok | noise floor KL mean / median / max, top-1 agree | mlx-lm vs mlx-vlm KL mean / median / max, top-1 agree |
|---|---|---|---|
| p_short raw | 226 | 0.021 / 0.002 / 0.26, 95.3% | 0.032 / 0.002 / 0.54, 93.0% |
| p_2k raw | 2072 | 0.057 / 0.007 / 1.21, 92.2% | 0.088 / 0.008 / 5.13, 91.4% |
| p_8k raw | 8218 | 0.017 / 0.000 / 0.29, 95.3% | 0.018 / 0.000 / 0.32, 96.9% |
| p_short chat, effort none | 239 | 0.034 / 0.002 / 0.86, 94.5% | 0.035 / 0.002 / 0.66, 95.3% |
| p_short chat, effort low | 241 | 0.029 / 0.002 / 0.53, 94.5% | 0.039 / 0.001 / 1.28, 96.9% |

Residual stream on p_short (cache-free forward): identical after the embedding norm, 4-6e-3
relative after the two dense layers (bf16 rounding), then the routed expert sets start to differ
(3% of tokens at layer 2, 9% at layer 5, 70-85% in the last layers) and the deviation grows with
them. Router audit inside mlx-lm (fp32 vs bf16 router logits, prompt + 128 tokens): 1.4-1.5% of
the (token, layer) selections flip, every one with a 6th/7th selection-score gap <= 0.0013 (median
0.0002; bf16 resolves 0.0039 near 1; the median gap over all decisions is 0.010): near-ties only.
The one large step (p_2k, max KL 5.1 nats at step 47) is a near-tie router flip: A2's per-layer
dumps on identical cache state put the first differing routed expert at layer 12 with a 6th/7th
selection-score gap of 0.0004-0.0009 (below bf16 resolution), amplified through the remaining
layers; mlx-vlm against itself shows the same signature (its own max 1.2 nats on that prompt).
Both models answer the question correctly on every prompt.

### Speed: the robust results first

- **8-bit attention/embed/unembed (the recipe this model file enables through its `quant_predicate`,
  `bench/K2/make_attn8.py`)**: in A2's same-hold pair (load 4.2-4.5) p_short decode 27.8 vs 22.6 tok/s
  for the bf16-attention build (+23%), peak 79.9 vs 83.0 GiB (-3.1 GiB), 79.3 vs 82.4 GiB on disk,
  and KL 0.028 (median 0.003, max 0.48) / top-1 95.3% against the bf16-attention build's logprobs on
  the same tokens: near-lossless and faster. In my hold 3 (load 4-6): 25.4 vs 22.2 tok/s decode,
  239 vs 196 tok/s prefill, 1.00 vs 1.22 s TTFT, KL 0.069 / top-1 95.3%.
- **Against mlx-vlm 0.7.11 (same checkpoint, same hold, back to back)** the bf16-attention port is
  not faster on decode: -3 to -23% across two holds (hold 2 at load 7-14: 35.3 vs 45.6 tok/s at
  p_short, 32.0 vs 40.6 at p_2k, 32.0 vs 38.2 at p_8k; A2's hold at load 4: 22.6 vs 23.5). Long-prompt
  prefill is within variance (-19 to -29% in hold 2, within +-5% beyond 2K in A2's hold), and short-
  prompt TTFT is within variance too (hold 2 favoured mlx-lm 2.5x on a 148 tok/s mlx-vlm outlier,
  A2's pair favoured mlx-vlm by 15%). Peak memory is 2.5-7.6 GiB lower at 2K-8K (the bounded
  sliding caches), 15 GiB lower at 32K (90.1 vs 105 GiB in contended runs). Absolute tok/s on this
  shared machine swung 1.6-2x between holds (the identical build: 35.3 tok/s in hold 2, 25.5 in
  hold 3), so only within-hold pairs are quoted.

Hold 2 pair (load 7-14, chat template at reasoning effort none, 128 new tokens, median of 3;
raw rows `bench/logs/K2-same-hold-table.md`, `*same-hold*.jsonl`):

| prompt | mlx-vlm 0.7.11 prefill tok/s / TTFT / decode tok/s / peak GiB | mlx-lm prefill / TTFT / decode / peak |
|---|---|---|---|
| p_short (239) | 148 / 1.62 s / 45.6 / 83.2 | 373 / 0.64 s / 35.3 / 83.0 |
| p_2k (2085) | 700 / 2.98 s / 40.6 / 87.8 | 568 / 3.67 s / 32.0 / 85.6 |
| p_8k (8231) | 758 / 10.9 s / 38.2 / 93.5 | 537 / 15.3 s / 32.0 / 86.5 |

Where the decode time goes is in the profile section below. Earlier, contended runs (hold 1,
load 16-30) are in `bench/logs/K2-hold1-table.md`; they are not comparable across engines.

### Hold 3 (03:14-03:21 local, load 4-6): fold vs dense shared experts, attn8, CLI

The machine was in a slower state than in hold 2 (the identical fold build: 25.5 tok/s decode at
p_short here vs 35.3 there), so only within-hold comparisons are read:

| prompt | fold build (`cce0834`) prefill / TTFT / decode / peak | dense build (`00cdad1`) prefill / TTFT / decode / peak | attn8 (dense build) |
|---|---|---|---|
| p_short | 209 tok/s / 1.14 s / 25.5 (25.2-25.8) / 83.0 GiB | 196 / 1.22 s / 22.2 (21.7-23.9) / 83.0 | 239 / 1.00 s / 25.4 (24.3-26.3) / 79.9 |
| p_8k | 448 / 18.4 s / 23.3 (22.5-23.6) / 86.5 | 411 / 20.0 s / 20.5 (19.9-21.0) / 86.4 | - |

- Tried and reverted: the dense shared-expert layout (commit `00cdad1`, mlx-vlm's layout) was
  12-13% slower at decode and 6-8% slower at prefill than the fold in this A/B, so my prefill
  hypothesis was wrong and the fold (one `gather_qmm` chain over the 258 experts) is the committed
  default; the shared-expert layout is not the cause of the gap to mlx-vlm.
- 8-bit attention (`bench/K2/make_attn8.py`: fused qkvr, wo_ud, embed and unembed at 8 bits /
  group 64, 86 modules, 14 s to save, 79.3 GiB on disk vs 82.4): +22% prefill, +14% decode,
  -3.1 GiB peak at p_short; against the bf16-attention build on the same 128 tokens KL mean 0.069
  / median 0.002 / max 3.3 nats, top-1 agreement 95.3% (the mlx-vlm self-noise floor on this prompt
  is 0.021 / 0.002 / 0.26, 95.3%); the six disagreements are near-ties (top-2 gaps 0.25-1.0) in
  the repetitive tail of the raw-prompt continuation.
- Stock CLI: `mlx_lm.generate --model <2-bit> -p "The secret to baking a good cake is" -m 200`
  ran with the default one-shot load, 37 s wall in total, 17.9 tok/s, peak 88.6 GB
  (`bench/logs/K2-cli-generate.log`; the transcript is in PR.md).

### Where the decode time goes (real widths, context 525, same hold): `bench/logs/K2-profile-decode*.log`

Synchronous `mx.eval` of one decode token, real weights, medians (dense-layout build for mlx-lm,
profiled minutes apart in a drifting machine state, so compare within a row):

| | mlx-lm | mlx-vlm 0.7.11 |
|---|---|---|
| whole step (42 layers + head) | 41.0 ms (best 39.9), later 50.7 (49.3) | 46.9 ms (best 44.7) |
| head: final norm + /16 + unembed + slice | 3.8-7.3 ms | 4.8 ms |
| one sliding MoE layer / global MoE layer / dense layer (chained x10) | 1.82-1.99 / 1.24-1.33 / 0.84-1.09 ms | 1.83 / 1.33 / 0.86 ms |
| qkvr GEMV / wo GEMV | 0.12-0.13 / 0.07-0.09 ms | 0.12 / 0.08 ms |
| short convs (k\|v, attn, mlp) | 0.05-0.07 + 0.04-0.07 + 0.04 ms (3 kernels) | 0.10 + 0.07 + 0.06 ms (4 kernels) |
| bias + SDPA (512 keys) | 0.06-0.08 (matmul + gather) + 0.05-0.06 | 0.05 (mask kernel) + 0.06 |
| router GEMV + selection | 0.04-0.06 + 0.05-0.08 | 0.05 + 0.07 |
| experts | 0.26 (8 experts in one gather_qmm chain) / 0.32 + 0.13 (dense build) | 0.22 (6) + 0.11 (dense shared) |
| decode loop, 64 tokens after the p_short prefill | `generate_step` pipelined 20.9 tok/s, synchronous `mx.eval` per token 20.3 tok/s | (hold 2, same state as its benchmark: 45.6 tok/s pipelined vs a 46.9 ms synchronous step measured in hold 3) |

Reading: per-layer and per-op costs are at parity with mlx-vlm (the fused k|v conv and the
bounded sliding cache are slightly cheaper, the 8-expert gather slightly dearer than 6 + dense),
and the synchronous step is at parity or faster. The end-to-end decode gap comes from the
generation loop: with this model mlx-lm's `generate_step` gains only 3% from `async_eval`
pipelining (20.9 vs 20.3 tok/s), i.e. the per-token CPU work (graph build plus the encoding of
~1,300 kernel launches) is as long as the GPU work and is not hidden, whereas mlx-vlm's
`stream_generate` hides most of it (45.6 tok/s against a ~47 ms synchronous step, 1.5-2x). The
tiny-width dispatch benchmarks (graph build 2.0 vs 3.0 ms/token, launch cost at parity) say the
difference is not in Python-side graph building; it is in what each loop lets overlap, which this
campaign did not resolve (candidates: per-kernel encoding cost at real widths, buffer donation
blocked by the one-step-ahead graph, the command-buffer split at 50 ops). Fewer kernels per
token (fusing the ~10 small norm/gather/elementwise ops per layer) is the lever either way.

## Files

- `PR.md`: facts for the pull request (mlx-lm requires a human-written description).
- `k2-mlx-lm-inkling.patch`: `git format-patch` of the branch.
- `tests/`: the tests added (also in the patch).
- `bench/`: scripts and raw logs (equivalence, benchmarks, cache microbenchmarks, attn8).
