# A2 adversarial review: K2's `inkling_mm_model` for mlx-lm

Reviewer A2, 2026-10-29. Independent of the author. Branch `inkling` in `upstream/mlx-lm`
(base `1f7057c`). The branch moved four times while the review ran: `cce0834` (the state when
the review started; shared experts folded into the routed gather), `00cdad1` (02:21, dense
shared experts), `cc6a6fe` (02:55, kv_bits warning, CPU exactness test, acknowledgment) and
`dc0a8b1` (03:26, the fold restored: its model file is the `cce0834` text plus two docstring
lines). Every verdict below was re-checked at the head `dc0a8b1` (section 3.9); the real-model
hold ran the `00cdad1` model code and K2's own hold-2/3 logs cover the fold. Experiments live on
branch `a2-review` in `upstream/mlx-lm-a2` (a worktree rebased onto `dc0a8b1`, venv
`~/.venvs/inkling-a2`: mlx 0.32.4, editable `mlx-lm-a2`); nothing was pushed. Raw logs are
under `review/`.

Method: `research/00-method.md` section 3 with the revised end-to-end criterion (teacher-forced
KL / top-1 against a measured noise floor, not token equality). Exact comparisons use the
float64 numpy oracle `research/oracle/inkling_ref.py` (validated against transformers at 1e-7)
on its own tiny random model, run with `MLX_ENABLE_TF32=0` because fp32 GEMMs with M > 1 run at
TF32 precision on this M5 by default (section 3.7).

## 1. Verdicts

| # | Claim (K2 README / PR.md) | Verdict | Evidence |
|---|---|---|---|
| 1 | The model implements the reference semantics (every op of R5 section 3) | CONFIRMED | 15/15 adversarial oracle tests pass at <= 1e-6 relative in fp32 on CPU and GPU (section 4.3): full forward, decode, chunked prefill at chunk sizes 1/3/5/7 and mixed, multi-token steps after rotation, left- and right-padded batches, prompt-cache save/load, `generate_step`, `BatchGenerator`, cache contents (K post-sconv post-norm, V post-sconv, conv state = last 3 inputs) |
| 2 | Both checkpoint layouts load to the same parameters (HF `w13` de-interleave gate=0::2, `[C,1,K]` conv swap, visual/audio/mtp dropped, idempotent on a converted checkpoint) | CONFIRMED | `test_sanitize_layouts_agree_and_idempotent`; both layouts also match the oracle |
| 3 | Fused q/k/v/r and k\|v conv are exact, including quantized rows | CONFIRMED | oracle tests through `sanitize`; `test_quantized_tiny` (8-bit attention rows + 4-bit experts vs the dequantized twin: 2e-6) |
| 4 | Shared experts exact in both layouts: folded as experts 256-257 of one gather (cce0834, the head dc0a8b1) and as a dense SwiGLU (00cdad1 only) | CONFIRMED | K2's `test_sanitize_quantized_fold` / `test_sanitize_quantized_shared_experts`; my oracle suite passes on all four commits; I=2048 is a multiple of the 64-group so expert blocks never share a group |
| 5 | `RotatingKVCache(512)` + `key_distances()` reproduces the window/bias semantics across rotations, chunking and batching | CONFIRMED | section 4.3 (40-token sequences through a 4-slot window, multi-token updates after rotation, batched rotation under left padding) |
| 6 | Short-conv Metal kernel is "bit-identical" to the compiled taps | PLAUSIBLE (overstated) | bit-identical for bf16/fp16 outputs at every L tested; fp32 differs by 4.5e-8 relative at L=1 (FMA/association), so say "identical after rounding" |
| 7 | fp32 router logits (vs bf16 in transformers / mlx-vlm) | PLAUSIBLE (decode only) | exact fp32 at decode (M=1 GEMV, 2e-7); at prefill (M > 1) the fp32 GEMM runs at TF32 precision by default on the M5 (7.5e-4 relative at the router's shapes), i.e. the prefill router is ~5x more precise than bf16 but not fp32 unless `MLX_ENABLE_TF32=0`; undeclared |
| 8 | Teacher-forced KL / top-1 vs mlx-vlm within the noise floor (p_short 0.032/93.0 %, p_2k 0.088/91.4 %, p_8k 0.018/96.9 %, chat 0.035/0.039) | CONFIRMED | my re-run on the same tokens against the same saved reference logprobs: p_short 0.045/94.5 %, p_2k 0.055/95.3 %, p_8k 0.019/98.4 % (floors 0.021 / 0.057 / 0.017); the reference re-run against itself is exactly 0 (section 5.1) |
| 9 | p_2k 5-nat outlier at step 47 is benign | CONFIRMED (explained) | per-layer dumps of both implementations on identical cache state: first routed-expert difference at layer 12 at a selection-score gap of 0.0004-0.0009 (below bf16 resolution), then a flip cascade through the MoE stack; 1.3 nats on the dense tree, 5.1 on the fold tree, 0.14 in the reference's own noise floor; neighbouring steps unaffected (section 5.2) |
| 10 | Router audit: ~1.4 % of selections differ fp32 vs bf16, all at gaps <= 0.0013 | PLAUSIBLE | consistent with the logs; the "fp32" side of the audit ran at TF32 precision during prefill (claim 7), which does not change the conclusion that the flips are near ties |
| 11 | bf16 fidelity equals mlx-vlm's | CONFIRMED (tiny model) | like-for-like on the tiny model: at bf16 both are 0.43-0.68 from the oracle (MoE amplification of bf16 near ties on a random model) while agreeing with each other to 1.7e-2 with identical greedy tokens at all 32 positions; at fp16 both are 2e-3 (`review/a2-tiny-bf16.log`) |
| 12 | Server batching works (`BatchGenerator`, right-padded chunked prefill, merge/extend/filter/extract) | CONFIRMED | `test_batch_generator` (prompts of 9/4/13 tokens with max_tokens 6/3/5 plus a 4th inserted mid-run, prefill chunk 5), `test_batched_right_padded_prefill_chunked`, `test_batched_decode_left_padding`; real model: `mlx_lm.server` completions and chat completions 200 with coherent text, `mlx_lm.generate` CLI exit 0 (section 5.4) |
| 13 | Prompt-cache save/load works | CONFIRMED (tiny) | `test_prompt_cache_save_load` after rotation |
| 14 | `--kv-bits` is a silent no-op, caches are not trimmable | CONFIRMED | `review/a2-tests-tf32off.log` (`test_trim`), kv-bits probe in section 4.4 |
| 15 | Speed: "not faster than mlx-vlm on this machine: decode ~20 % slower, prefill beyond 2K 19-29 % slower, wins on short-prompt TTFT (2.5x) and peak memory" (README after hold 2) | PLAUSIBLE in direction, magnitudes not robust | my same-hold pair: decode -3 to -8 %, prefill within +-5 % beyond 2K, p_short TTFT 15 % in mlx-vlm's favour, peak -2.5 to -7.1 GiB; the 2.5x TTFT claim came from an abnormal mlx-vlm first prefill and should go; the dense layout of `00cdad1` was a 12-13 % decode regression vs the fold in K2's own hold-3 pair and `dc0a8b1` restores the fold (section 6) |
| 16 | mlx-lm test suite and pre-commit pass | CONFIRMED for the Inkling tests and pre-commit; the full suite has 19 pre-existing failures | section 4.2 |
| 17 | 8-bit attention/embed/unembed variant: near-lossless and faster | CONFIRMED | p_short KL 0.028 / top-1 95.3 % vs the bf16-attention build on the same tokens; same-hold decode 27.8 vs 22.6 tok/s (+23 %), peak 79.9 vs 83.0 GiB (section 5.5) |

Overall: no BROKEN claim. The implementation is semantically correct (oracle at 1e-6, real
model at the reference's noise floor, server and CLI work), its numerics are declared except
for the TF32 prefill router (7) and the "bit-identical" wording (6; fixed in the docstring at
`cc6a6fe`). The one performance regression found (the dense shared experts of `00cdad1`) was
reverted by K2 in `dc0a8b1` before this report was finished; what remains to fix is text: the
README's 2.5x TTFT sentence, the TF32 caveat, and the PR placeholders. The attn8 conversion
recipe (+23 % decode at KL 0.028) is the result worth leading with.

## 2. What was reviewed and how

- Code: `mlx_lm/models/inkling_mm_model.py` (813 lines at the head `dc0a8b1`, 829 at
  `00cdad1`), `mlx_lm/models/cache.py` (+21: `key_distances` on `RotatingKVCache` /
  `BatchRotatingKVCache`), `mlx_lm/generate.py` (+7 at `cc6a6fe`: the kv_bits warning),
  `tests/test_inkling.py` (439), `tests/test_models.py` (+33), `ACKNOWLEDGMENTS.md` (+1). Read against R5 section 3/6 (semantics), R3 section 2/4
  (conventions and the proposed design) and the mlx-lm sources the model relies on
  (`cache.py` rotation, batching and prompt-cache code; `generate.py` prefill, batch and server
  paths; `utils.py` loader order sanitize -> quantize -> load_weights; `convert.py` predicates).
- Tests: K2's tests, the whole mlx-lm suite, `uvx pre-commit run --all-files`, then 15
  adversarial tests of my own against the oracle (`tests/test_inkling_a2.py` on `a2-review`,
  copy in `review/`), a like-for-like bf16 comparison with the stock mlx-vlm 0.7.11 wheel
  (`review/a2_tiny_bf16.py`) and probes (TF32, kv-bits).
- Real model: one bounded GPU hold under the queue protocol (10 minutes, 03:23-03:33):
  teacher-forced runs on both implementations with per-layer dumps, p_32k, the stock CLI and
  server, back-to-back benchmarks and the attn8 variant (`review/a2_hold.sh`,
  `review/a2_forced.py`, `review/a2_outlier_compare.py`); results in sections 5 and 6.

## 3. Code review against the reference semantics (R5 section 3) and mlx-lm conventions

Line numbers refer to `inkling_mm_model.py` at `00cdad1`, the tree the real-model hold ran. The
head `dc0a8b1` is the `cce0834` file plus two docstring lines: its `select_experts` / `MoE` /
`sanitize` sit at 565-609 / 710-791 (no `SharedExperts` module; the shared experts are experts
256-257 of one 258-expert `SwitchGLU`, fused by `sanitize`'s `fuse`), everything else is at the
same lines.

### 3.1 sanitize (`Model.sanitize`, 724-810)
- Drops `model.visual.*`, `model.audio.*`, `model.mtp.*` by prefix (729). Correct for the
  mlx-community files (no mtp tensors) and the HF source (mtp present).
- HF layout: `w13_dn` / `experts.w13_weight` / `shared_experts.shared_w13_weight` are
  de-interleaved with `reshape(..., rows/2, 2, H)` and taken at `[..., 0, :]` (gate) and
  `[..., 1, :]` (up) = rows `0::2` / `1::2` (732-734), matching R5 (`gate = w13[..., 0::2, :]`);
  `w2_md` / `w2_weight` / `shared_w2_weight` -> `down_proj`. `[C,1,K]` conv weights are swapped
  to `[C,K,1]` only when `shape[1] == 1 and shape[2] > 1` (739), so the mlx-community `[C,4,1]`
  layout passes through; the oracle test confirms the tap orientation (tap K-1 = current token).
- `mlp.gate.weight` is upcast to fp32 (763); `rel_logits_proj.proj` is stored reversed as
  `proj_reversed` (765) so the decode bias is a slice. Re-sanitizing a converted checkpoint is a
  no-op (my idempotence test), so `mlx_lm.convert` round trips.
- Fusions: `qkvr_proj` from `[wq_du, wk_dv, wv_dv, wr_du]` and `kv_sconv` from
  `[k_sconv, v_sconv]` by row concatenation of `weight`/`scales`/`biases` (`fuse`, 770-788);
  exact for quantized rows because affine groups run along the input dimension. Mismatched
  quantization raises. Only fused when the unfused names exist, so converted files load.
- Shared experts: at `cce0834` and the head `dc0a8b1` they are appended to the routed experts
  (`fuse` of `experts.{gate,up,down}_proj` with `shared_experts.*` along the expert axis, exact
  for quantized tensors since each expert keeps its own groups; mismatched quantization
  raises). `00cdad1` laid them out as one dense SwiGLU instead: gate/up `[S*I, H]`
  (`reshape(-1, H)`), down `[H, S*I]` (`transpose(1,0,2).reshape(H, -1)`, 806), also exact
  because I=2048 is a multiple of the 64-group - R3 section 4.3's recommended layout, reverted
  in `dc0a8b1` for speed (section 6).
- Finding (minor): with `.scales` present but `.biases` absent (an `mxfp4`/`nvfp4` conversion),
  `fuse` only concatenates the leaves that exist, which is right, but the dense shared-expert
  transform would silently reshape a `global_scale` leaf of ndim 1 (skipped by `ndim != 3`) -
  fine - and the loader's `_takes_global_scale` path was not exercised. Not a defect for the
  affine checkpoints this PR targets.

### 3.2 Short convolutions (`ShortConv`, 183-250; kernel 135-165; compiled taps 113-132)
- Semantics `y[t] = x[t] + sum_k w[k] x[t-(K-1)+k]`, zero left padding, state = the last `K-1`
  inputs (`xp[:, L:]`, 245) - confirmed against `sconv_ref` (prefill, chunked, state round trip)
  and by the cache-contents test (conv states equal the oracle's at 3e-7).
- Rounding: fp32 accumulate, one rounding to the activation dtype, then the layer residual is
  added and rounded again (`_short_conv_taps_residual`, 131; kernel 151-154). This is exactly
  the transformers sequence (fp32 conv + built-in residual, one cast; then the bf16 residual
  add), not mlx-vlm's extra rounding of the conv output before its built-in residual (R5 #8).
  Declared in the code comment.
- k and v share one conv over the concatenated channels (`kv_sconv`, 366) before `k_norm`
  and before the cache write (374-378): matches R5 c (K cached post-sconv post-norm, V post-sconv),
  verified by the cache-contents test. State `[B, 3, 2*kv_dims]` in slot 0 of `ArraysCache(3)`.
- Kernel path only for `L <= 8`, no padding mask and `cache.lengths is None` (209-216);
  otherwise the compiled taps. Kernel == fallback: bit-identical for bf16/fp16 at L in
  {1,2,5,8,9,20}; fp32 differs at the 4.5e-8 level (claim 6).
- Batched right-padded prompts: the state is gathered at `clip(lengths, 0, L)` (241-243), which
  also keeps the old state for sequences whose prompt ended in an earlier chunk (`lengths <= 0`);
  verified by the chunked right-padded batch test (prefill chunk 4 over 11/5 tokens).

### 3.3 Attention (`Attention.__call__`, 371-427), relative bias, window, tau
- `q`, `k|v`, `r` from one `qkvr_proj` (373); per-head `q_norm`/`k_norm` on `head_dim` (377-378);
  v not normed; scale `1/head_dim` (353); GQA by `mx.fast.scaled_dot_product_attention` (head
  `h // (H/Hkv)` implicit) - all as R5 c.
- Bias: `rel_logits = r @ proj_reversed` -> `[B, H, L, E]` listing the bias against the keys
  `E-1..0` positions back (391). Prefill (`L > 1`): `banded_mask` (291-320, Metal kernel
  253-273 with a pure-MLX fallback) builds `[B, H, L, S]` with `-inf` for `dist < 0` and, on local
  layers, `dist >= window`; the bias for `0 <= dist < E`; `0` for `dist >= E` (global layers).
  Verified against `rel_bias_ref` + masking at 12 shapes including L=1/8/37/600/1300 against
  S=548/1030/1100/1111/1300/1500/2000 (window 512 / extent 512 and global 1024) and the tiny
  shapes, kernel and fallback (both exact).
- Decode on a local layer: one gather over the rotated slots with
  `rel_index = E - 1 - cache.key_distances()` (396-398, 429-432), computed once for the first
  sliding layer and shared (671-685) because all sliding caches move in lockstep. The visible
  set is exactly the 512 slots (window inclusive of self); after a multi-token update the
  rotating cache keeps `max_size - 1` old keys plus the new ones, which is the "visible iff
  `0 <= i - j < 512`" rule. Global layers: a slice `rel_logits[..., E-S:]` for `S <= E`, zero-padded
  on the left beyond E (399-403). All verified by the oracle decode/chunk tests through many
  rotations (window 4, extent 8, 40 tokens).
- `tau` only on global layers (`log_scaling_n_floor` is `None` on sliding layers, 354) and only
  when `offset + L > n_floor` (407-421): per query position in fp32,
  `1 + alpha * log(max((pos+1)/n_floor, 1))`, applied to `q` and to the additive mask, both rounded
  to the model dtype (420-421) - the transformers sequence (R5 #7: mlx-vlm rounds tau itself to
  bf16). Verified with `n_floor = 4` (positions 3..39 active) and in the batched case where
  `offset` is a per-row array (414-415).
- Masks for batched caches: `cache.make_mask(L)` is taken before the update (384), which is the
  convention of `BatchKVCache` / `BatchRotatingKVCache` (they describe the cache after the
  upcoming update); the bias is then `where(valid, bias, -inf)` (404-405). Verified for left
  padding (merge after prompts of 7 and 3) and right padding (`prepare`/`finalize`).
- Quantized KV: `scaled_dot_product_attention` from `base.py` would switch on `cache.bits`, but a
  `CacheList` never gets quantized (3.6).

### 3.4 Router and experts (`Router` 454-468, `select_experts` 560-575, kernels 486-544, `MoE` 600-624)
- Logits `x.fp32 @ W.fp32.T` over `[256 routed | 2 shared]` rows (616); selection
  `sigmoid(routed) + bias` top-6 (ties to the lowest index in the kernel: `simd_min` over tied
  lanes with the lowest tap first); weights `softmax(logsigmoid([6 routed, 2 shared]))` x
  `route_scale` x `global_scale` (R5 d / 6.3). Verified against `router_ref` at N = 1/37/300,
  R = 256, including an exact tie at the 6th slot (kernel picks the lowest index like the
  oracle; the compiled `argpartition` fallback picks the other member of the tie on that one
  row - unspecified order, acceptable since it is the non-Metal fallback) and a 1e-4 near tie
  (both correct). Weights agree to 4e-7.
- Routed experts through `SwitchGLU` (sorted `gather_qmm` above 64 tokens, mlx-lm convention);
  weights applied after the down projection, rounded to the activation dtype (621). Shared
  experts at the head (`dc0a8b1`, as at `cce0834`): experts 256-257 of one 258-expert gather,
  their gammas applied after the down projection like the routed weights - mathematically the
  reference's "gamma before the down projection" (linear), verified by the oracle suite.
  `00cdad1` used a dense SwiGLU with the gamma applied before the down projection
  (`SharedExperts.__call__`, 592-597); both exact (3.9).
- Dense layers 0-1: `DenseMLP` with `* global_scale` (449-451). Final norm, `* (1/16)` (686),
  `unembed`, slice to `unpadded_vocab_size` (688-689). Exact (1/16 is a power of two).
- Finding (claim 7): at prefill the router GEMM has M > 1 and MLX 0.32.4 runs it at TF32
  precision on the M5 by default (measured at the router's shapes: M=1 2.3e-7, M=2 7.5e-4,
  M=2048 7.6e-4 relative, `review/` probe in section 4.4). The README's "the router keeps fp32
  logits (vLLM / SGLang / llama.cpp do)" is true at decode and ~5x better than bf16 at prefill,
  but not fp32 there. Cheap fix if wanted: none at the model level (TF32 is an MLX env switch);
  the honest fix is a sentence in the README/PR.

### 3.5 Model, registration, config
- `model_type` `inkling_mm_model` = file name (CONTRIBUTING); `ModelArgs(text_config=...)` with
  a flat `inkling_text` config accepted (98-110); unsupported knobs (`q_bias`, `o_bias`,
  softcapping, non-sigmoid gate, ...) raise at construction (69-82) - explicit, good.
- `Model.layers`, `make_cache` (`CacheList(RotatingKVCache(512) | KVCache(), ArraysCache(3))`,
  713-722), `input_embeddings` support, deepcopy: `model_test_runner` passes (K2's test).
- `cast_predicate` keeps only `.mlp.gate.` in fp32 (812-817). R3 4.1 suggested also norms,
  conv taps and global scales; the loader does not cast at all (only `convert --dtype` does),
  the checkpoint values are bf16-representable, and the convs/router accumulate in fp32, so
  nothing is lost - but a `convert --dtype float16` would put bf16-valued taps into fp16 (fine
  for these magnitudes). Not a defect.
- `quant_predicate`: attention projections, `embed`, `unembed` at 8-bit/g64, the rest at the
  requested bits (820-828). Verified on the tiny model (`quantize_model` + `dequantize_model`).

### 3.6 Cache design and its consequences
- `RotatingKVCache` instead of R3's shifting cache (declared, with K2's microbenchmarks). The
  new `key_distances()` (cache.py 533-545 / 1256-1263) is correct for `keep = 0` in every state
  I could construct (growth in steps, temporal order after a concat update, rotation, right
  after `_trim`); the `keep > 0` branch is unused by Inkling and untested at the batched
  variant (which has no `keep`). It is the one addition to a shared file and is small and
  documented.
- Not trimmable (`ArraysCache`), so `trim_prompt_cache` returns 0 (my `test_trim`), no
  speculative decoding, no `LRUPromptCache` trimming (prefix reuse still works). `--kv-bits`
  is a silent no-op (`maybe_quantize_kv_cache` needs `to_quantized`; `CacheList` has none;
  probe in 4.4). Both declared in the README; the CLI should probably warn.
- Memory hazard (design-inherent, shared with mlx-vlm): prefill materialises a
  `[B, 32, L, S]` bf16 mask per layer; on the 7 global layers at 32K context that is
  32 x 2048 x 34816 x 2 B = 4.6 GB transient per layer (K2 measured 90.1 GiB peak at p_32k),
  ~9 GB at 64K and 18 GB at 128K, on top of the 82 GiB of weights and the 107.5 GiB working set.
  `prefill_step_size` bounds it (M1 used 512 for p_64k). Worth a sentence in the PR.

### 3.7 Validation hazard applied to K2's own tests
K2's tiny tests use `rtol=atol=1e-2` "because fp32 GEMMs may run at TF32 precision" - true, and
with `MLX_ENABLE_TF32=0` the same comparisons hold at 1e-6 (section 4.3). The PR's "~1e-6
relative on a tiny random model" statement comes from `bench/K2/test_vs_oracle.py`, which is
not part of the patch; the repo tests themselves only prove 1e-2.

### 3.8 Numerics changed without being declared
- TF32 router logits at prefill (3.4). Everything else that differs from mlx-vlm is declared:
  fp32 router logits at decode, fp32 tau, transformers-style sconv rounding, routing weights
  rounded to bf16 before the expert sum (same as mlx-vlm), shared experts scaled before the
  down projection (exact order).

### 3.9 Deltas while the review ran
- `cce0834 -> 00cdad1`: `SharedExperts` dense module (3.1, 3.4), `select_experts` returns
  `inds [N, K]` and `weights [N, K+S]` (the kernel no longer writes the shared indices), tests
  adjusted.
- `00cdad1 -> cc6a6fe`: `warnings.warn` in `maybe_quantize_kv_cache` when no cache supports
  `to_quantized` (fixes defect 3; the warning fires once per process thanks to Python's default
  filter, although the function runs every step), `test_cpu_exactness` (decode and chunked
  prefill vs the full forward on the CPU device at 1e-5; a self-consistency test, not a
  reference test) and `test_kv_bits_warns`, the ACKNOWLEDGMENTS line, the short-conv docstring
  ("agree after the rounding to the activation dtype").
- `cc6a6fe -> dc0a8b1`: the fold restored (model file = `cce0834` + 2 docstring lines; the
  `fuse` of shared experts and the `[N, K+S]` kernel indices are back), tests restored.
My suite passes on all four commits (`review/a2-tests-tf32off-cce0834.log`,
`review/a2-tests-tf32off.log` at `00cdad1`, `review/a2-tests-tf32off-dc0a8b1.log` at the head,
15/15, 261 checks each); the real-model hold used `00cdad1`, K2's hold 2 and 3 used the fold.
Any further commit before submission needs the oracle suite re-run (`MLX_ENABLE_TF32=0 python
-m unittest tests.test_inkling_a2`).

## 4. Tests

### 4.1 K2's tests
`python -m unittest tests.test_inkling tests.test_models.TestModels.test_inkling_mm_model`:
11 tests OK at `cce0834`, 13 at the head `dc0a8b1`, with the default environment and with
`MLX_ENABLE_TF32=0` (`review/k2-tests.log`, `review/k2-tests-dc0a8b1.log`).

### 4.2 Whole mlx-lm suite and style
- `uvx pre-commit run --all-files` (black, isort, ruff): passed with no file modified at
  `cce0834` (`review/pre-commit.log`) and at the head (`review/pre-commit-dc0a8b1.log`; the
  only findings were two unused imports in my own test file, fixed).
- `python -m unittest discover tests/` on the branch (worktree, same tree): 409 tests, 19
  failures, 1 skipped (`review/unittest-full.log`). None of the 19 touches Inkling: they are
  `test_cached_matches_forward` / `test_left_padding` of deepseek_v41 and muse_spark, the
  `test_generate` batch tests, `test_ssm`, `test_phixtral`, `test_minicpm3`, `test_flint`,
  `test_afm7_left_padding`, `test_llama4_chunked_kv_cache`, `test_granitemoehybrid_padded_batch`,
  `test_spark2_5_chunked_vs_oneshot_prefill`, `test_stream_generate_input_embeddings*`.
  Classification: section 4.2.1.
- `pytest tests/` from the repo root cannot even collect `tests/models/test_muse_spark.py`
  (`import test_models`) on main; the project's documented runner is `unittest discover`.

#### 4.2.1 Classification of the 19 failures
Exactly those 19 tests re-run on the base commit `1f7057c` with the default environment: all
19 fail (`review/unittest-failures-on-base-1f7057c.log`). The same 19 on K2's branch with
`MLX_ENABLE_TF32=0`: all 19 pass (`review/unittest-failures-on-k2-tf32off.log`). They are
pre-existing fp32-tolerance failures caused by the M5's default TF32 GEMMs, independent of this
PR. (Pre-existing and worth telling the maintainers separately.)

### 4.3 Adversarial oracle tests (`tests/test_inkling_a2.py`, 15 tests, 261 recorded checks)
Run: `MLX_ENABLE_TF32=0 python -m unittest tests.test_inkling_a2 -v` -> 15/15 OK at the head
`dc0a8b1` (`review/a2-tests-tf32off-dc0a8b1.log`), at `00cdad1` (`review/a2-tests-tf32off.log`)
and at `cce0834` (`review/a2-tests-tf32off-cce0834.log`). Tolerance
1e-5 relative (max |d| / max |ref|) unless stated; measured errors are 2e-8 .. 2e-6. GPU cases
skip themselves if TF32 is on. Weights are the R5 tiny model (hidden 64, 2 q / 1 kv heads of
32, d_rel 4, extent 8, window 4, 8 routed experts top-2 + 2 shared, layers [sliding, global,
sliding], dense layer 0, n_floor 4) produced in the HF source layout and the mlx-community
layout and loaded through `sanitize`.

| test | what it covers | worst error |
|---|---|---|
| sanitize_layouts_agree_and_idempotent | HF vs mlx-community layouts identical after sanitize, visual/audio/mtp dropped, idempotent, strict load, bf16 router weight upcast | exact |
| full_forward | B=2 L=12 (both layouts) and L=40 logits + per-layer outputs; L=1 without cache; batch rows == single rows | 1.1e-6 |
| cached_decode_and_chunked_prefill | 40 tokens: decode 7+1x33; chunks of 1, 3, 5, 7; mixed; multi-token steps after rotation (9,1,3,1,2,1,5,1,8,1,7,1); the oracle's own chunking (1.6e-15) | 7.6e-7 |
| cache_contents | after a 9-token prefill: cached K (post-sconv, post-k_norm) and V equal the oracle's last keys; conv states = last 3 inputs (k\|v, attn, mlp); sliding caches bounded at 4 slots through 40 tokens | 5e-7 |
| prompt_cache_save_load | save after rotation (9,1,1,1), load, 28 decode steps identical to the original (0) and vs the oracle | 1e-6 |
| trim | `can_trim_prompt_cache` False, `trim_prompt_cache` returns 0, decoding continues correctly | - |
| batched_decode_left_padding | prompts of 7 and 3 merged (`CacheList.merge` -> Batch*KV caches with left padding), 8 decode steps per row vs per-sequence oracle (rotation under padding, tau from per-row offsets) | 1e-6 |
| batched_right_padded_prefill_chunked | server prompt path: `prepare(lengths, right_padding)`, chunk 4 over 11/5 tokens, `finalize`, 6 decode steps | 1e-6 |
| generate_step | greedy `generate_step` with `prefill_step_size=5`, 8 steps: logprobs vs oracle, tokens = oracle argmax | 2e-7 (tol 1e-4) |
| batch_generator | `BatchGenerator` continuous batching: prompts of 9/4/13 tokens with max_tokens 6/3/5 (sequences leave at different steps: `filter`/`extract`), a 4th prompt of 7 tokens inserted after step 2 while the others' sliding caches are rotated (`extend` of `BatchRotatingKVCache`/`BatchKVCache`/`ArraysCache`), prefill chunk 5 with right padding: every step's logprobs vs the oracle, tokens = oracle argmax | 4e-7 (tol 1e-4) |
| banded_mask_vs_oracle | kernel and fallback vs `rel_bias_ref` + masking at 12 shapes crossing window and extent | 2e-8 |
| router_vs_oracle | kernel and compiled vs `router_ref` at N=1/37/300, R=256, exact tie and 1e-4 near tie | 4e-7 |
| short_conv_kernel_equals_fallback | bf16/fp16/fp32 at L=1,2,5,8,9,20 with residual and state: kernel vs fallback (bit-identical for bf16/fp16; fp32 4.5e-8), fp32 vs oracle, state exact | 4.5e-8 |
| reduced_precision_paths | bf16 / fp16 through the kernels vs oracle with routing flips counted (bf16: 2/32 tokens flip, 0.43 on the rest - see claim 11), kernels vs fallbacks (<= 1e-2), routing identical | - |
| quantized_tiny | `quantize_model` with K2's predicate (attention/embed/unembed 8-bit g64, experts/dense 4-bit g32) vs `dequantize_model` twin, full and decode | 2e-6 |

### 4.4 Probes
- TF32 (`review/` section of this file, probe output): default environment, GPU fp32 matmul vs
  float64: M=1 K=4096 N=258 2.3e-7; M=2 7.5e-4; M=64 9.3e-4; M=2048 7.6e-4.
- `--kv-bits`: `generate_step(..., kv_bits=4, quantized_kv_start=0)` on the tiny model leaves
  every layer's cache as `CacheList(RotatingKVCache | KVCache, ArraysCache)` and the tokens
  unchanged: silent no-op.
- bf16 like-for-like (`review/a2-tiny-bf16.log`): see claim 11.

## 5. Real model (A2 GPU hold, 03:23-03:33 local, 601 s, load 6 at start)

Checkpoint `mlx-community/Inkling-Small-mlx-2bit` (82.4 GiB), lazy load + chunked eval
(2.1 + 6.7 s), wired limit set; the tree under test was `963f755` = the `00cdad1` model code
(dense shared experts; the head at that moment, `cc6a6fe`, differed only by a docstring). K2
committed the fold back (`dc0a8b1`) three minutes into the hold; the fold's real-model numbers
are K2's hold-2 teacher-forced run and hold-2/3 benchmarks, and my follow-up hold (5.6) dumps
it on the outlier step. Reference: the stock mlx-vlm 0.7.11 wheel in `~/.venvs/inkling-base`. Scripts and raw outputs: `review/a2_hold.sh`,
`review/a2_forced.py`, `review/a2-forced-*.{json,log}`, `review/a2-dumps/`,
`review/a2-outlier-compare.md`, `review/hold-branch-state.txt`.

### 5.1 Teacher-forced KL / top-1 (claim 8): CONFIRMED
Same tokens (mlx-vlm's greedy tokens of K2's hold 2), same prefill chunking (2048) and loop as
K2's `equiv.py`, scored against the saved reference logprobs with K2's statistic
(fp16 logprobs, KL(ref || mine)):

| run | prompt tok | noise floor (mlx-vlm vs itself, hold 2) KL mean / max, top-1 | K2 hold 2 (cce0834 fold) | A2 (00cdad1 dense) | argmax == fed |
|---|---|---|---|---|---|
| p_short_raw | 226 | 0.0207 / 0.260, 95.3 % | 0.0315 / 0.538, 93.0 % | 0.0446 / 1.295 (step 111), 94.5 % | 121/128 |
| p_2k_raw | 2072 | 0.0570 / 1.206, 92.2 % | 0.0876 / 5.134 (step 47), 91.4 % | 0.0552 / 1.390 (step 16), 95.3 % | 122/128 |
| p_8k_raw | 8218 | 0.0174 / 0.294, 95.3 % | 0.0178 / 0.322, 96.9 % | 0.0188 / 0.512 (step 8), 98.4 % | 126/128 |

The reference itself re-run on its own tokens gives KL exactly 0.0000 and 128/128 on
p_short and p_2k (`a2-forced-mlx_vlm.log`): the saved logprobs are deterministic, every
difference is the port's. All three runs are at the noise floor's level (means within 2x of the
floor, top-1 agreement equal or better); the 5-nat p_2k maximum of K2's run does not occur
with the dense tree (its maximum is 1.39 at step 16, the step the noise floor also flags).

### 5.2 The p_2k step-47 outlier (claim 9): a near-tie router flip, amplified; not a bug
Per-layer dumps of the decode forward that produces the step-47 distribution (the fed token
" The", id 623, at position 2118) from both implementations on identical cache state
(`a2-outlier-compare.md`, dump "step 46"):

- layers 0-11: the residual stream deviates smoothly, mean |d|/mean |ref| 3e-3 -> 4.8e-2,
  every routed expert set identical;
- layer 12 (local): the first differing routed set, at a 6th/7th selection-score gap of
  0.00094 (mlx-vlm, bf16 logits) / 0.00039 (mlx-lm, fp32 logits), far below the bf16
  resolution of 0.0039 - a near tie that the two routers resolve differently;
- the deviation jumps to 1.3e-1 after layer 12 and further flips cascade (layers 15, 16,
  17, 19, 20, 21, 22, ...), reaching ~0.4 by layer 20; the final distributions differ by
  1.30 nats (" narrator" at logprob -1.41 vs -0.03), top-1 unchanged.
- the neighbouring steps are unaffected (KL 0.097 before, 0.003 after; cache state is shared,
  so nothing persists), exactly the pattern of K2's run (0.022 / 5.134 / 0.004).

So the event is this model's MoE amplification of a sub-bf16 routing tie on one token in a
highly repetitive context (A1's finding), not an implementation error: the same token costs
1.3 nats with the dense tree, 5.1 with the fold tree and 0.14 in the reference's own
chunk-size noise floor; which side of each later near tie a build lands on decides the size.
K2's hold-2 text ("one 5-nat outlier step") is accurate but should say what it is.

Follow-up mini hold (04:03, `review/a2_hold2.sh`, load 5): the fold tree (`cce0834` model file,
i.e. the head's) on the same tokens reproduces K2's hold-2 run to the last digit (KL mean
0.0876 / median 0.0076 / max 5.1338 at step 47, top-1 91.4 %, 117/128,
`a2-forced-mlx_lm-fold.log`), two hours and a different process later: a build is run-to-run
deterministic, and the 5-nat step is a deterministic property of that build on that token. Its
per-layer trace (`a2-outlier-compare-fold.md`, dump step 46) has the same first flip as the
dense tree - layer 12, gap 0.00094 (mlx-vlm) / 0.00106 (mlx-lm) - and the same cascade layers
(15, 16, 17, 19, 20, ...); only the size at the output differs (5.13 nats, " QA" first, versus
1.30 with the dense tree). Neighbouring steps: 0.022 / 0.004 nats.

### 5.3 p_32k (chat template, reasoning effort none)
32807 prompt tokens: TTFT 73.4 s (447 tok/s prefill), answer `The narrator is Ishmael, and the
ship is the Pequod.` then EOS after 17 tokens at 29.2 tok/s; peak 90.1 GiB (K2 measured the
same 90.1). Correct and bounded: the 35 sliding layers hold 512 keys, the global layers 32.8K.

### 5.4 User-facing paths (claims 12 and the PR template transcript)
- `python -m mlx_lm generate --model ... -p "The secret to baking a good cake is" -m 64 --temp 0`
  (`a2-cli-generate.log`): stock one-shot load, exit 0, 30.3 s wall, 64 tokens at 20.4 tok/s,
  peak 88.7 GB, coherent `<|content_thinking|>` output. The one-shot `mx.eval` of 82 GB did not
  trip the Metal watchdog here (page cache warm from the previous load); the 23-token prompt
  reports 2.6 tok/s because the first forward includes the three Metal kernels' JIT and the
  usual first-run cost, the same warm-up every benchmark log shows. The PR's required transcript
  can be taken from this command.
- `python -m mlx_lm.server --model ... --port 8181` (`a2-server-smoke.log`): HTTP up in 5 s,
  model loaded on the first request, `/v1/completions` and `/v1/chat/completions` both 200 with
  coherent text, prompt-cache insertion and the batched prompt path logged. Works.

### 5.5 The 8-bit-attention variant (`~/models/inkling-small/mlx-2bit-attn8`, built by K2's hold 3)
79.3 GiB on disk (86 modules: fused `qkvr_proj`, `wo_ud`, `embed`, `unembed` at 8 bits / g64, the
2-bit experts untouched). p_short teacher forced on the same mlx-vlm tokens, scored against the
bf16-attention mlx-lm logprobs of this hold: KL mean 0.028 / median 0.0033 / max 0.478 (step
12), top-1 95.3 %, 122/128 argmax agreement (`a2-forced-attn8.log`), i.e. the 8-bit attention
costs about as much as the bf16 port's own distance to mlx-vlm (0.045) and a little more than
the reference's chunk-size noise floor (0.021). Speed in the same hold (`a2-bench-attn8.log`,
load 4.2-4.5): p_short decode 27.8 tok/s vs 22.6 for the bf16-attention build and 23.5 for
mlx-vlm (+23 %), prefill 209 vs 193 tok/s, peak 79.9 vs 83.0 GiB. This is the one clear speed
win in the package and matches R3 section 3.1's prediction (~30 % at decode from 24-31 % less
weight traffic); it is a `quant_predicate` and a re-conversion, so it belongs in the PR as the
recommended conversion recipe, with the KL figure.

### 5.6 Determinism and the follow-up mini hold (04:03-04:04, 44 s, load 5)
`review/a2_hold2.sh` ran two things (`a2-hold2.log`, raw lines in section 10.1):
- the fold tree (`cce0834`, worktree `upstream/mlx-lm-a2-fold`) on the same tokens with the
  step 45-48 dumps: it reproduces K2's hold-2 statistics to the last printed digit (5.2), which
  settles determinism - the same build gives the same logprobs in another process two hours
  later - and shows the fold's 5-nat event has the same layer-12 first flip as the dense tree's
  1.3-nat one;
- a second mlx-lm run against the main hold's dense-tree logprobs. Intended as a repeat of the
  dense tree, it actually ran the head's code: my worktree had been rebased onto `dc0a8b1` (the
  fold) between the two holds, and the driver resolves `mlx_lm` through it. So
  `a2-forced-mlx_lm-rerun.log` is the head (fold) versus the dense tree on identical tokens:
  KL mean 0.0505 / median 0.0048 / max 2.34 (step 47), top-1 91.4 %, 117/128 - the two exact
  layouts differ from each other by the same mechanism and about as much as each differs from
  mlx-vlm, which is the expected size of this model's near-tie amplification between any two
  bf16 implementations. (A literal dense-vs-dense repeat was not run; the fold's exact
  reproduction of K2's run makes it redundant.)

## 6. Benchmark skepticism

K2's README (after hold 2) states the port is not faster: decode ~20 % slower, prefill beyond 2K
19-29 % slower, 2.5x faster TTFT on the short prompt, lower peak memory. The underlying logs
(`bench/results/K2-bench-mlxvlm-same-hold.log` 02:05-02:13 and
`K2-bench-mlxlm-2bit-same-hold.log` 02:13-02:15, same hold, same checkpoint, same 239 / 2085 /
8231-token chat prompts, EOS ignored, 128 tokens, one warm-up, 3 timed runs, load1 7-14 on both):

| prompt | mlx-vlm 0.7.11 prefill tok/s, TTFT, decode tok/s (wall, median) | mlx-lm (cce0834) | ratio |
|---|---|---|---|
| p_short | 148, 1.62 s, 45.6 | 373, 0.64 s, 35.3 | TTFT 2.5x better, decode -23 % |
| p_2k | 700, 2.98 s, 40.6 | 568, 3.67 s, 32.0 | prefill -19 %, decode -21 % |
| p_8k | 758, 10.87 s, 38.2 | 537, 14.0 s, 31.9 | prefill -29 %, decode -16 % |
| peak GB (api) | 89.4 / 94.4 / 100.4 | 89.1 / 92.0 / 92.8 | -0.3 / -2.4 / -7.6 GB |

The two scripts (`bench/baseline/bench_mlx.py`, `bench/K2/bench_mlx_lm.py`) define the same
quantities (API `prompt_tps`, wall TTFT, `(n-1)/decode_wall`, `peak_memory`), tokenise to the
same counts and ignore EOS the same way, so the comparison is like-for-like; the earlier
hold-1 table (`K2-hold1-table.md`, load 15-30, mlx-lm faster on p_short decode) is not.

My own back-to-back pair, 03:27-03:32 in A2's hold, same scripts, same prompts, mlx-lm
(dense tree `00cdad1`) first then the mlx-vlm wheel, load1 4-9 (`review/a2-bench-mlxlm.log`,
`review/a2-bench-mlxvlm.log`, table in `review/a2-hold-summary.md`):

| prompt | mlx-vlm 0.7.11 prefill tok/s, TTFT, decode tok/s, peak GiB | mlx-lm 00cdad1 | ratio |
|---|---|---|---|
| p_short | 224, 1.07 s, 23.5, 83.2 | 193, 1.24 s, 22.6, 83.0 | prefill -14 %, TTFT -15 %, decode -4 % |
| p_2k | 404, 5.16 s, 22.4, 88.2 | 408, 5.11 s, 20.6, 85.7 | prefill +1 %, decode -8 %, peak -2.5 GiB |
| p_8k | 410, 20.1 s, 22.4, 93.5 | 430, 19.2 s, 21.8, 86.4 | prefill +5 %, decode -3 %, peak -7.1 GiB |

The machine was in a different state than in hold 2: both engines decode at ~22 tok/s here
against 32-46 tok/s there (K2's hold 3 at 03:14, load 4.6, measured the same ~22 tok/s for
its own builds), so absolute numbers on this shared Mac swing by ~1.6-2x between holds and
only same-hold pairs are comparable. The two same-hold pairs disagree on the size of the gap
(hold 2: decode -16 to -23 %, prefill -19 to -29 % beyond 2K, p_short TTFT 2.5x in mlx-lm's
favour; my hold: decode -3 to -8 %, prefill within +-5 % beyond 2K, p_short TTFT 15 % in
mlx-vlm's favour). What is robust across both: the port is not faster than mlx-vlm 0.7.11 on
decode, it is not slower on long-prompt prefill by more than the machine's variance, and its
peak memory is lower by 2.5-7.6 GiB at 2K-8K (the bounded sliding caches). The 2.5x TTFT claim
is not robust (it came from an abnormally slow mlx-vlm first prefill in hold 2: 148 tok/s with
32 % spread) and should be dropped from the README. "Faster" is not claimed and must not be.

Dense shared experts vs the fold (K2's own hold 3, back to back, load 4.6, same checkpoint,
`bench/results/K2-bench-mlxlm-{fold,dense}-h3.log`, the fold run through a
`inkling_mm_model_fold` copy of the `cce0834` model file): fold p_short / p_8k decode 25.5 /
23.3 tok/s, prefill 209 / 448 tok/s; dense 22.2 / 20.4 tok/s, prefill 196 / 411 tok/s. The
commit `00cdad1` is a 12-13 % decode and 6-8 % prefill regression at real width, outside the
run spreads (25.2-25.8 vs 21.7-23.9). K2's own real-width profile (`K2-profile-decode*.log`)
says why: one 8-expert `gather_qmm` call (0.26 ms) is cheaper than 6 experts plus three dense
quantized GEMVs (0.22 + 0.11 ms), and the dense path adds launches (reshape, scale, add) per
layer in a dispatch-bound step. R3 section 4.3 recommended the dense layout from isolated
microbenchmarks; the whole-model numbers say the opposite. Both layouts are exact. K2 drew the
same conclusion from its hold-3 pair and restored the fold in `dc0a8b1` (03:26), so the head
is the faster layout; the comparison above should still go into the PR as the reason.

## 7. Upstream fit

- CONTRIBUTING: file name == `model_type` (yes); test in `tests/test_models.py` (yes, with the
  runner); `uvx pre-commit run --all` (passes); tests fail on main (yes: `tests/test_inkling.py`
  imports the new module). `ACKNOWLEDGMENTS.md` has no Inkling entry (add one).
- PR template (`new_model.md`): requires the `mlx_lm.generate` transcript (PLACEHOLDER in
  PR.md until K2's hold 3) and the transformers reference output "if you have access to CUDA
  hardware" (honestly marked not run). PR.md states clearly that it is facts for a human-written
  description and carries an AI-use disclosure line; the final description must be written by
  the submitter (CONTRIBUTING prohibits AI-written PR text).
- No dead code found (ruff clean; every branch of `Attention.__call__`, both kernel fallbacks and
  `_select_experts` are reachable and exercised by the tests). `RelativeLogits` is a bare
  parameter holder; fine.
- `cache.py` gains `key_distances()`; small, documented, tested (`test_key_distances` across
  rotations).
- Size: ~830-line model file, in the range of the Kimi K3 / DeepSeek-V4.1 PRs; three
  `mx.fast.metal_kernel`s (banded mask, short conv, routing), each <= 50 lines with a pure-MLX
  fallback and a test - the pattern the maintainers accepted before, but the review record says
  kernels "with not much to gain" get dropped: K2 measured the conv kernel (~3.5 ms/token of
  launches) and the route kernel (~4.5 ms/token) on the tiny-width profile; keep those numbers
  in the PR.
- README of mlx-lm: no per-model list to update.

## 8. Defects and required changes before submission

Defects (none affects correctness; 5 and 6 affect what the submission says about speed):
1. (doc) "bit-identical" for the short-conv kernel vs the compiled taps holds only after
   bf16/fp16 rounding; fp32 differs at 4.5e-8. Repro: `tests/test_inkling_a2.py::
   test_short_conv_kernel_equals_fallback` (REPORT lines "bit-identical: False" for fp32).
2. (doc/numerics) The prefill router runs at TF32 precision by default on the M5 (7.5e-4),
   so "fp32 router logits" is a decode-only statement; the router audit's fp32 side was TF32
   during prefill. Repro: the probe in section 4.4.
3. (UX) `--kv-bits` is silently ignored; `mlx_lm.generate --kv-bits 4` should at least warn
   (the server already refuses batching with kv_bits, but the single-stream path says nothing).
4. (test strength) the repo tests prove the tiny model at 1e-2 only; add `MLX_ENABLE_TF32=0`
   guidance or an oracle-free exactness test that is immune to TF32 (compare CPU vs GPU, or
   decode vs prefill on the CPU device), so the 1e-6 statement in the PR is backed by a test in
   the patch.

5. (performance, `00cdad1` only) The dense shared-expert layout was 12-13 % slower at decode
   and 6-8 % at prefill than the fold it replaced, in K2's own back-to-back hold-3 pair
   (section 6). Repro: `bench/K2/bench_mlx_lm.py` with `--model-type inkling_mm_model_fold`
   then the default, same hold. Both layouts are exact. Reverted by K2 in `dc0a8b1`.
6. (README) "wins on short-prompt TTFT (2.5x)" is not robust: in my same-hold pair mlx-vlm's
   p_short TTFT is 15 % better; hold 2's mlx-vlm p_short prefill was an outlier (148 tok/s,
   32 % spread). Keep "lower peak memory" (robust: -2.5 to -7.6 GiB at 2K-8K) and "not
   faster" (robust).

Already addressed by K2 while the review ran: the kv_bits warning (3) and the "bit-identical"
docstring (1) in `cc6a6fe`, a CPU exactness test (4, partly: it compares the model with itself
on the CPU device at 1e-5, which catches chunking/decode bugs but not reference-semantics
bugs) and the ACKNOWLEDGMENTS entry in `cc6a6fe`, the fold layout (5) in `dc0a8b1`.

Required before submission:
- Remove the 2.5x TTFT sentence (6) and keep "not faster" / "lower peak memory"; quote one
  same-hold pair with its load, and note that absolute tok/s on this machine moved 1.6-2x
  between holds.
- Fill the PR.md placeholders with the real `mlx_lm.generate` transcript
  (`review/a2-cli-generate.log` has one from the stock CLI) and the measured tables; state the
  TF32 caveat (2) and the memory note of 3.6 in the PR text; add the attn8 recipe and its KL.
- Say in the PR what the 5-nat p_2k step is (section 5.2) if the hold-2 numbers are quoted.
- Re-run `tests/test_inkling_a2.py` (or fold the oracle-independent parts of it into
  `tests/test_inkling.py`) after any further commit; the head passes.

## 9. Raw logs (`package/k2-mlx-lm-inkling/review/`)

- `k2-tests.log` K2's tests (default and TF32 off); `pre-commit.log`; `unittest-full.log`
  (409 tests); `unittest-failures-on-base-1f7057c.log`, `unittest-failures-on-k2-tf32off.log`.
- `a2-tests-tf32off-dc0a8b1.log` (head), `a2-tests-tf32off.log` (00cdad1),
  `a2-tests-tf32off-cce0834.log`, `a2-tests-batchgen.log`; `k2-tests-dc0a8b1.log`,
  `pre-commit-dc0a8b1.log`; `test_inkling_a2.py` (copy of the branch file, commit `1966960`
  on `a2-review`). The three `a2-tests-tf32off-after-*.log` files were written by K2 after
  its own commits (not by me); the `after-kvbits-warning` one is an empty shell error.
- `a2-tiny-bf16.log` (`a2_tiny_bf16.py`).
- Hold (03:23-03:33): `a2_hold.sh` (driver log `a2-hold.log`), `a2_forced.py`,
  `a2_outlier_compare.py`, `a2_summarize.py`; outputs `a2-forced-mlx_lm.{json,log}`,
  `a2-forced-mlx_vlm.{json,log}`, `a2-forced-attn8.{json,log}`, `a2-logprobs-mlx_lm/`,
  `a2-dumps/` (8 npz), `a2-outlier-compare.md`, `a2-cli-generate.log`, `a2-server-smoke.log`
  (+ `a2-server-stdout.log`), `a2-bench-{mlxlm,mlxvlm,attn8}.{jsonl,log}`,
  `a2-hold-summary.md`, `hold-branch-state.txt`.
- Follow-up hold: `a2_hold2.sh` (`a2-hold2.log`), outputs `a2-forced-mlx_lm-fold.{json,log}`,
  `a2-dumps-fold/`, `a2-outlier-compare-fold.md`, `a2-forced-mlx_lm-rerun.{json,log}`.

## 10. Follow-up hold results

Ran 04:03-04:04 (44 s, lock released and queue left by the driver). Interpretation in section
5.6; the raw headline lines were appended below by `review/a2_append_followup.sh`.

### 10.1 Results (appended automatically by `review/a2_append_followup.sh` at 2026-10-29T08:04:37Z)

Fold tree (`cce0834` model file = the head `dc0a8b1`'s) teacher forced on p_2k, vs the mlx-vlm reference logprobs (`a2-forced-mlx_lm-fold.log`):

    [04:04:03]   vs reference logprobs (K2-logprobs-mlx_vlm): KL mean 0.0876 median 0.0076 max 5.1338 (step 47); |dlogprob| top-1 mean 0.1234 max 5.2431; top-1 agreement 91.4%
    [04:04:03]   argmax agrees with the fed tokens at 117/128 steps; prefill 5.2s, decode 6.6s (19.2 tok/s), min margin 0.000

Per-layer comparison of the fold tree with mlx-vlm on the dump steps (`a2-outlier-compare-fold.md`; dump step 46 = the step-47 distribution):

    == p_2k_raw step 45: fed token 25 at position 2117: KL(mlx-vlm || mlx-lm) = 0.0223 nats; mlx-vlm top-5 [623, 83078, 730, 357, 220] (lp [-0.18, -2.18, -3.93, -4.43, -5.68]); mlx-lm top-5 [623, 83078, 730, 357, 88228] (lp [-0.12, -2.5, -4.12, -5.5, -6.0])
    first layer whose routed expert set differs: 16
    == p_2k_raw step 46: fed token 623 at position 2118: KL(mlx-vlm || mlx-lm) = 5.1321 nats; mlx-vlm top-5 [101829, 34777, 18225, 6052, 1577] (lp [-0.03, -4.9, -5.65, -6.03, -6.28]); mlx-lm top-5 [103333, 4928, 23963, 220, 30985] (lp [-0.33, -2.58, -3.33, -3.83, 
    first layer whose routed expert set differs: 12
    == p_2k_raw step 47: fed token 101829 at position 2119: KL(mlx-vlm || mlx-lm) = 0.0037 nats; mlx-vlm top-5 [382, 328, 11, 350, 83078] (lp [-0.03, -4.03, -4.9, -7.03, -7.15]); mlx-lm top-5 [382, 11, 328, 83078, 350] (lp [-0.03, -4.28, -4.41, -6.41, -7.41])
    first layer whose routed expert set differs: 18
    == p_2k_raw step 48: fed token 382 at position 2120: KL(mlx-vlm || mlx-lm) = 0.0217 nats; mlx-vlm top-5 [83078, 290, 261, 11484, 4358] (lp [-0.05, -4.67, -4.8, -5.17, -5.17]); mlx-lm top-5 [83078, 290, 11484, 261, 4358] (lp [-0.02, -5.27, -5.89, -6.14, -6.14])
    first layer whose routed expert set differs: 5

Second mlx-lm run against the main hold's dense-tree logprobs (`a2-forced-mlx_lm-rerun.log`). It ran the head's fold code (worktree `mlx-lm-a2` at `1966960` = `dc0a8b1` + tests, see 5.6), so this is the fold versus the dense tree on identical tokens, not a dense repeat; determinism is established by the fold run above matching K2's hold-2 statistics exactly:

    [04:04:24]   vs reference logprobs (a2-logprobs-mlx_lm): KL mean 0.0505 median 0.0048 max 2.3376 (step 47); |dlogprob| top-1 mean 0.0913 max 3.8643; top-1 agreement 91.4%
    [04:04:24]   argmax agrees with the fed tokens at 117/128 steps; prefill 5.3s, decode 6.1s (20.7 tok/s), min margin 0.000

Driver log: `a2-hold2.log`.
