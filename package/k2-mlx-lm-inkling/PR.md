# Pull request draft: Inkling (`inkling_mm_model`) for mlx-lm

Target: `ml-explore/mlx-lm`, branch `main` (base `1f7057c`). Patch series: `k2-mlx-lm-inkling.patch`.

> mlx-lm's CONTRIBUTING.md prohibits AI-written pull-request descriptions and requires the
> author to understand every line and to disclose how AI was used. The submitter writes the PR text
> himself; this file collects the facts and measurements for it. The code, tests and this draft
> were written with Claude (Anthropic) under human direction; the disclosure line below states that.

## Requirements (template)

- ☑️ I understand it is strictly prohibited to use AI to write PR description
- AI usage disclosure: the model file, the cache helper, the tests and the benchmark scripts were
  written with Claude (Anthropic, Claude Code) from the transformers reference implementation and
  the mlx-vlm port, under human direction and review; the numerical validation (numpy oracle of the
  reference semantics, greedy-token equivalence against mlx-vlm on the real checkpoint) was run on
  the author's machine. The author takes responsibility for every line.

## Model information

- Name: Inkling-Small (Thinking Machines), 276B total / 12B active parameters, hybrid
  sliding-window (35 layers, window 512) / global (7 layers) attention with a hidden-state
  conditioned relative position bias, short convolutions on k/v and around both sublayers,
  256-expert sigmoid-routed MoE with 2 always-on shared experts (top-6 + 2 sinks), dense MLPs
  on layers 0-1, muP logit scaling and an untied unembedding.
- Homepage: https://huggingface.co/thinkingmachines/Inkling-Small
- Reference implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling

## Supported checkpoints

- https://huggingface.co/mlx-community/Inkling-Small-mlx-2bit (loaded directly; `model.llm.*` paths)
- https://huggingface.co/mlx-community/Inkling-Small-mlx-3bit and -4bit (same layout)
- https://huggingface.co/thinkingmachines/Inkling-Small via `mlx_lm.convert` (the HF source layout:
  interleaved `w13` gate/up rows are de-interleaved, `[C,1,K]` conv weights swapped, MTP / vision /
  audio weights dropped)

The vision and audio towers of the multimodal checkpoint are not loaded; this is the text model.

## What the port does differently from the vendored / mlx-vlm implementations

- q/k/v/r are one fused projection (`attn.qkvr_proj`, fused at load; row-concatenation is exact
  for quantized weights too); k and v share one short convolution with a single cached state.
- The three short convolutions of a layer run as one Metal kernel each for up to 8 tokens
  (fp32 accumulation, one rounding, residual and state update included); longer inputs use a
  compiled tap sum, bit-identical to the kernel.
- The relative bias bank is stored reversed, so a single token's bias against its keys is a slice
  of `r @ proj` (global layers) or one gather with a slot index shared by all sliding layers
  (`RotatingKVCache.key_distances()`, new); prefill builds the banded additive mask with a
  20-line Metal kernel (pure-MLX fallback).
- The router runs in fp32 and its selection (sigmoid + bias top-6, log-sigmoid softmax over the
  6 routed + 2 shared logits, times route_scale * global_scale) is one Metal kernel; the 2 shared
  experts are appended to the 256 routed ones so routed and shared experts go through one
  `SwitchGLU` (`gather_qmm`) call.
- Sliding layers use `RotatingKVCache(512)` (bounded memory, `BatchRotatingKVCache` for the
  server); global layers `KVCache`; the three conv states live in an `ArraysCache(3)`.
- `quant_predicate` keeps attention, embedding and unembedding at 8 bits when converting
  (attention is the dominant error source below 8 bits); `cast_predicate` keeps the router in fp32.
- The log-scaling factor is only applied once positions reach `log_scaling_n_floor` (128000).
- Every custom kernel has a pure-MLX path for non-Metal devices and a test against numpy. The
  short-conv kernel and the compiled taps agree bit for bit after bf16/fp16 rounding (their fp32
  accumulation orders differ at the 5e-8 level).

Numerics and memory caveats:

- The router's "fp32 logits" is exact at decode (GEMV); at prefill the fp32 GEMM runs at TF32
  precision (~7.5e-4 relative) on the M5 unless `MLX_ENABLE_TF32=0` is set before MLX is
  imported. Expert selection differs from bf16 logits only at near-ties (see the router audit).
- Prefill materializes one `[B, 32, L, S]` bf16 additive mask per layer (window / bias extent /
  causal), as mlx-vlm does: with `prefill_step_size` 2048 that is up to 335 MB per sliding layer
  and 4.6 GB per global layer at 32K context (9 GB at 64K, 18 GB at 128K), transient, on top of
  the 82 GiB of weights; a smaller `prefill_step_size` bounds it (512 for 64K+ prompts on 128 GB).
- `--kv-bits` has no effect on this model (its per-layer `CacheList` has no `to_quantized`);
  the single-stream path now warns once instead of staying silent. The caches are not
  trimmable (conv states), so there is no speculative decoding and no prompt-cache trimming;
  prefix reuse works.

## Verification

Run mlx-lm command (the stock CLI, default one-shot load of the 82 GB checkpoint: 37 s wall,
no watchdog timeout on this machine; the 2-bit checkpoint, chat template applied by the CLI):

```console
mlx_lm.generate --model mlx-community/Inkling-Small-mlx-2bit -p "The secret to baking a good cake is" -m 200
```

Output:

```
==========
<|content_thinking|>The user is asking "The secret to baking a good cake is" which seems like they want to know the secret or key to baking a good cake, but it's phrased as a fill-in-the-blank or asking for a specific answer. However, it's somewhat ambiguous - they might be asking for general tips, or they might be referencing a specific saying/phrase.

Actually, looking at it again: "The secret to baking a good cake is" - this could be a prompt asking me to complete the sentence, or asking for advice. But there's no specific context. I should provide a comprehensive answer covering the main "secrets" or key principles.

Key things for baking a good cake:
1. Precision/measurement (not just "a bit")
2. Temperature of ingredients (room temperature eggs/butter)
3. Don't overmix (especially with flour)
4. Oven temperature/accuracy
5. Fresh ingredients
6. The "secret" often cited: "the secret is
==========
Prompt: 23 tokens, 3.517 tokens-per-sec
Generation: 200 tokens, 17.885 tokens-per-sec
Peak memory: 88.618 GB
```

Equivalence against mlx-vlm 0.7.11 on the same checkpoint (teacher-forced, the reference's greedy
tokens fed to both, 128 steps, per-step output distributions compared; the noise floor is mlx-vlm
against itself with prefill chunks 2048 vs 128): KL(mlx-vlm || mlx-lm) mean 0.018-0.088 nats
(floor 0.017-0.057), medians equal to the floor, top-1 agreement 91-97% (floor 92-95%) on
p_short / p_2k / p_8k and the chat template at reasoning effort none / low. Greedy sequences
diverge after 0-17 tokens at near-tie steps (top-2 logit gaps <= 1.0 on the reference side),
as any bf16 implementation of this model does. The largest teacher-forced step (p_2k, 5.1 nats at
step 47) is a near-tie router flip at layer 12 (6th/7th selection-score gap 0.0004-0.0009, below
bf16 resolution) amplified through the stack, not an error; mlx-vlm against itself shows the same
signature. Full table: README.md / bench/logs/K2-equivalence.md.

Tests: `python -m unittest tests.test_inkling tests.test_models.TestModels.test_inkling_mm_model`
(tiny-config runner incl. batch 2 and deepcopy; banded mask vs numpy at shapes crossing the
window and the bias extent; short-conv state round trip (prefill == decodes, bit for bit);
router vs numpy; HF vs mlx-community layout sanitize; exactness of the quantized shared-expert
fold; `key_distances` across rotations; right-padded batched prompts through merged caches;
prefill/decode consistency). `uvx pre-commit run --all-files` passes.

## Reference output (if you have access to CUDA hardware)

Not run (no CUDA hardware). The numpy float64 oracle of the reference semantics
(`research/oracle/inkling_ref.py`, validated against transformers at ~1e-7) agrees with this
implementation at ~1e-6 relative on a tiny random model (full forward, cached decode, chunked
prefill, batch of 2), with `MLX_ENABLE_TF32=0`.

## Benchmarks (M5 Max 128 GB, mlx 0.32.4)

The result worth leading with is the quantization recipe the model file enables through its
`quant_predicate` (attention projections, embedding and unembedding at 8 bits; `bench/K2/make_attn8.py`
re-quantizes the mlx-community 2-bit checkpoint in 14 s, 79.3 vs 82.4 GiB on disk): in a same-hold pair on
p_short, decode 27.8 vs 22.6 tok/s for the bf16-attention build (+23%), peak 79.9 vs 83.0 GiB, and
KL 0.028 / top-1 95.3% against the bf16-attention build's logprobs on the same tokens (near-lossless).

Against mlx-vlm 0.7.11 on the same checkpoint, same hold, back to back (chat template at reasoning
effort none, greedy, 128 new tokens, median of 3), the bf16-attention port is not faster on decode
(-3 to -23% across two holds: hold 2 below at load 7-14, A2's pair at load 4: 22.6 vs 23.5 tok/s),
long-prompt prefill is within variance (-19 to -29% in hold 2, within +-5% in A2's pair), short-prompt
TTFT is within variance, and peak memory is 2.5-7.6 GiB lower at 2K-8K (15 GiB at 32K). Absolute
tok/s swung 1.6-2x between holds on this shared machine, so only within-hold pairs are quoted.

| prompt | mlx-vlm 0.7.11 prefill tok/s / TTFT / decode tok/s / peak GiB | mlx-lm prefill / TTFT / decode / peak |
|---|---|---|---|
| p_short (239 tok) | 148 / 1.62 s / 45.6 / 83.2 | 373 / 0.64 s / 35.3 / 83.0 |
| p_2k (2085) | 700 / 2.98 s / 40.6 / 87.8 | 568 / 3.67 s / 32.0 / 85.6 |
| p_8k (8231) | 758 / 10.9 s / 38.2 / 93.5 | 537 / 15.3 s / 32.0 / 86.5 |

Synchronous per-token work is at parity (41-51 vs 47 ms per decode step at context 525, per-layer
costs equal within noise, bench/logs/K2-profile-decode*.log); the gap is in how much of the
per-token CPU work the generation loop overlaps with the GPU (mlx-lm's `generate_step` with this
model: 20.9 tok/s pipelined vs 20.3 synchronous).

Hold 3 (03:14-03:21 local, load 4-6, a slower machine state than hold 2: the identical fold
build measured 25.5 tok/s decode at p_short here vs 35.3 in hold 2), fold build vs dense-shared-expert
build back to back, p_short and p_8k, 3 reps each:

| prompt | fold build prefill / TTFT / decode / peak | dense build prefill / TTFT / decode / peak |
|---|---|---|---|
| p_short | 209 tok/s / 1.14 s / 25.5 (25.2-25.8) / 83.0 GiB | 196 / 1.22 s / 22.2 (21.7-23.9) / 83.0 |
| p_8k | 448 / 18.4 s / 23.3 (22.5-23.6) / 86.5 | 411 / 20.0 s / 20.5 (19.9-21.0) / 86.4 |

Tried and reverted: the dense shared-expert layout (mlx-vlm's) was 12-13% slower at decode and
6-8% slower at prefill than the fold, so the fold (one `gather_qmm` chain for routed and shared
experts) is the committed default; the shared-expert layout is not the cause of the gap.
The 8-bit-attention variant (`bench/K2/make_attn8.py`, 86 modules re-quantized in 14 s, 79.3 GiB
on disk vs 82.4): p_short prefill 239 tok/s, TTFT 1.00 s, decode 25.4 tok/s (24.3-26.3), peak
79.9 GiB, measured in the same hold against the dense bf16-attention build's 196 / 1.22 s / 22.2 /
83.0 GiB; its agreement with the bf16-attention run is in README.md.
