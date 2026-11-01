# A4 adversarial review of K4's `inkling-fast-path` (mlx-vlm, base 138d761)

Reviewer: A4 (independent of the author). Branch reviewed: `inkling-fast-path` at
`eb93183` (9 commits over 138d761 = the 0.7.11 wheel). Experiments live on
`a4-review` (worktree `upstream/mlx-vlm-a4`, same source as eb93183 plus
`mlx_vlm/tests/test_a4_*.py`, `_a4_common.py` and `_a4_language_138d761.py`);
raw logs and results under `review/`. Nothing was pushed.

Method: research/00-method.md section 3. Bit-exact claims are judged by
identical tokens and identical logprobs; everything else by teacher-forced
KL/perplexity against the chunking noise floor of the unmodified code.
The unit-level oracle is `language.py` exactly as at 138d761, imported as a
second module with the same weights (`mlx_vlm/tests/_a4_language_138d761.py`),
so "before" in the unit tests is the released code, not a sibling path of the
branch. Real-model runs use the stock wheel in `~/.venvs/inkling-base` as
"before" and `~/.venvs/inkling-a4` (editable on the eb93183 worktree) as "after".

## 0. Overall

No BROKEN item. Every change K4 declares bit-exact is bit-exact against the
released code: at the unit level (118 adversarial tests against `language.py`
at 138d761, chunks 1/7/64/512/513, windows 8/512, batch rows, rollback,
snapshots, fp32/bf16, with and without TF32) and on the real model (identical
greedy tokens and top-2 margins at 239/8231/32807 tokens, md5-identical fp32
log-softmax matrices teacher-forced at 239 and 8231 tokens; K4's own A/B adds
2085 and 32807 at 128 tokens). Memory and prefill/TTFT gains are CONFIRMED in a
quiet hold (105.1 -> 95.3 GiB at 32K, prefill +17-19 % and TTFT -15-16 % at 8K
and 32K). The decode gain is NOT CONFIRMED (within noise here; K4's A/B drifted
2x mid-hold). kv-bits 8 works and sits at the chunking-noise level in quality,
but saves no memory below ~64K on this model. The code is ready; the PR text
needs four wording fixes (section 8) and I recommend splitting it (section 7).
One robustness gap (deep `trim` on the window cache, guarded today only by a
private dispatch helper) is recommended, not required.

## 1. Verdicts per claim

| # | claim (PR.md / README.md) | verdict | evidence |
|---|---|---|---|
| 1 | chunked loader: identical end state, the 82 GiB checkpoint loads with `lazy=False` on the branch, the wheel trips the GPU watchdog | CONFIRMED | `eval_parameters` review (section 2), `test_a4_loader.py`, K4's `after-eager-load` (13.1 s) and M1's BASELINE 2.5 for the wheel; my hold: eager load 11.7 s, tokens identical (5.4) |
| 2+4 | window caches are bit-exact: same keys in the same order, mask offset unchanged; memory bounded; batch/rollback/snapshot kept | CONFIRMED | section 2 (invariant proof), `test_a4_window_cache.py` + `test_a4_attention_vs_138d761.py` bit-exact vs 138d761 at chunks 1/7/64/512/513, windows 8/512, batch 1/2, fp32/bf16, rollback and snapshots; real model: logprob-identical matrices at 239/8231 tokens, identical tokens at 32807 (5.1) |
| 3 | route kernel indexing is the same arithmetic | CONFIRMED | model-level bit-exactness vs the 138d761 kernel on the tiny MoE (16 experts); real model logprob identity (5.1) |
| 5a | strided sconv kernel bit-exact, no contiguous copies | CONFIRMED | `test_a4_sconv_random.py` (16 random strided shapes, fp32/bf16) bit-exact vs the 138d761 kernel; state exact vs the pure path |
| 5b | tau skip is exact below the floor and unchanged above it | CONFIRMED | proof in section 2; `test_global_layer_tau_skip_bit_exact_across_the_floor`, `test_decode_steps_exactly_at_the_floor` |
| 6 | `logits_to_keep` honored; bit-exact on the single-sequence path, "same math" on multi-row | CONFIRMED (as stated) | `generate_step` processes the last prompt token as a 1-token `_step` (ar.py:528-530) so nothing is sliced on that path; sliced rows match the full head within 1e-6 (`test_a4_logits_to_keep.py`); the batch path reproduces 138d761 single rows |
| 7 | `global_scale` cast: no-op for the shipped checkpoints, fp32 scalar no longer promotes the stream | CONFIRMED | K4's dtype tests; by construction no cast when dtypes match |
| 8 | kv-bits quantizes the 7 global layers only, errors are clear, within the noise floor | PLAUSIBLE (works; quality at the chunking-floor level; no memory benefit below ~64K) | `test_a4_kv_bits.py` (quantized attention == dequantized math to 1e-4 with TF32 off, GQA mask grouping right); real model 5.3: KL 1.3e-2 mean at 8K inside the 2.5e-2 floor, 4.1e-2 at p_short |
| E | end-to-end: identical 128 tokens and top-2 margins on 4 prompts | CONFIRMED (eb93183: identical tokens, margins and fp32 logprob matrices) | `logs/k4-results-reanalysis.txt`; my teacher-forced md5/logprob comparison |
| M | peak memory 104.9 -> 95.3 GiB at 32K, 93.5 -> 91.5 at 8K; 64K fits at 100.4 | CONFIRMED for 32K/8K (105.1 -> 95.3, 93.5 -> 91.7 GiB); 64K "wheel does not fit" is NOT measured | K4 rows reproduce across reps; section 5 |
| P | prefill +5..20 %, TTFT -5..17 % | CONFIRMED at 8K/32K | my hold, best of 3 at load 4-7: +17 % / +19 % prefill, -15 % / -16 % TTFT; short-prompt prefill +3 % is noise; K4's rows carried a load asymmetry (section 6) |
| D | decode +29..45 % | NOT CONFIRMED | my hold: +13 % / +13 % / -4 % within rep spread; K4's A/B drifted 2x mid-hold (5.2) |
| T | 5157 passed / 16 skipped; pre-commit clean | CONFIRMED with the RAM caveat | my run: 5074 passed + 83 APC-budget failures at 5 GiB free, K4's at free RAM: 5157; pre-commit clean (section 3) |
| S | side effect: longer stale per-turn caches are now cold-prefilled | CONFIRMED, and it is a correctness fix | `test_prefix_cache_trim_amount_on_inkling_layer_entries` shows 138d761 reporting a longer stale cache as reusable untrimmed |

## 2. Code review per commit

### 7d50093 loader: `eval_parameters` in 4 GiB chunks

Correct. `tree_flatten(model.parameters())` yields the live parameter arrays, so
chunked `mx.eval` leaves every array evaluated once; `nbytes` on a lazy array
is shape-derived and does not force it. `lazy=True` path unchanged; the
offload-dir branch (`patch_model`) runs before the eval as before. Chunked
evaluation of a graph with inputs shared across chunks gives the same values
as one eval (test `test_a4_loader.py`). The 4 GiB constant is a module
global read at call time (tests monkeypatch it). Nothing to object to;
generic enough to be its own PR.

### f7f79d1 `SlidingWindowKVCache` / `BatchSlidingWindowKVCache`

Semantics checked against the attention contract (R5 section 3c, 6.1): a
local layer's query at absolute position p needs keys p-511..p, i.e. the
`max_size - 1 = 511` keys before the first incoming token plus the incoming
block. `_planned_drop` (inherited from `BufferedRotatingKVCache`) compacts to
`target_start = offset - max_size + 1` only when `_idx + L` would exceed
`max_size + max(buffer_size, L)`, so after every update the returned view has
`n - L >= min(511, offset_before)` old keys in temporal order, and
`InklingAttention`'s existing slice `j0 = S - L - 511` and mask offset `S - L`
see exactly what they saw on the unbounded cache. Invariant
`start_position + _idx == offset` holds through update and trim. Verified at
the cache level under random schedules (window 8 and 512, buffer 0/1/256,
chunks 1..700) and at the attention and model level bit-exactly against
138d761 (section 4).

Boundary: window 512 keeps 511 old keys, matching "visible iff 0 <= i-j < 512"
(512 keys including self). The first 512 tokens: no eviction, identical to
KVCache. Chunks straddling the window (513 after 1000, 7-token chunks across
a compaction): bit-exact.

Memory: decode compacts every `buffer_size` (256) tokens into a fresh buffer of
fixed capacity (768 after a short prompt, `ceil((L+512)/256)*256` after an
L-token chunk); the old buffer is dropped, so the per-layer footprint is
bounded at `capacity x 2 x n_kv x D x 2 B` (1.5 MB at 768 for the real shapes)
and the per-token slice-update touches a 1.5 MB buffer instead of the whole
context (this is the decode-at-depth mechanism, R1 kv_probe). Batch form:
compaction keeps the last 511 columns and shifts `left_padding` (possibly
negative); `extract`/`merge`/`extend`/`filter`/`trim` keep the layout.

Findings:

- F1 (robustness, inherited design): `trim(n)` deeper than the update that last
  compacted silently drops keys the next query needs (the window loses up to
  `n - slack` keys), while `is_trimmable()` is unconditionally True. Repro:
  `test_a4_window_cache.py::test_trim_inside_the_last_update_keeps_coverage_and_beyond_it_is_guarded`
  (window 8, 20-token prefill, 6 decodes, `trim(4)` -> the query at 22 sees 4
  old keys instead of 7). Reachability today: the speculative paths trim at
  most the block they appended (safe: compaction always happens inside the
  block being rolled back); the prompt-cache reuse path is guarded by
  `dispatch._cache_fully_retained` (`start_position == 0`), which my test also
  asserts. So no reachable caller breaks, but the contract is only enforced by
  a private helper in dispatch.py. K4's `test_trim_rolls_back_like_the_reference`
  trims 4 after 1-token updates and compares only the suffix, so it does not
  see the hole. Recommended: `SlidingWindowKVCache.trim` should refuse (or
  clamp and report) a trim past `offset - (start_position + max_size - 1)`.
- F2 (documentation): `BatchSlidingWindowKVCache._planned_drop` returns 0 while
  right padding is pending, so during a batched prompt the sliding layers keep
  the whole prompt (as before) and only bound after `finalize()`. The PR text
  says the caches are bounded; for the batch generator that holds only from
  decode on.
- `BufferedRotatingKVCache.state` setter fix (`getattr(self, "offset", 0)`) is
  right: `from_state` assigns `state` before `meta_state` on a bare instance.

### 0a963ac route kernel indexes `logits` directly

Same arithmetic, only the address computation changes (a `device T*` alias
into a possibly-constant-address-space input replaced by direct indexing).
Needed because MLX binds inputs under 32 bytes in constant memory; the real
model's rows are 516 B (bf16 [258]). Bit-exactness on the real model is
covered by the teacher-forced run (section 5). The 138d761 kernel indeed fails
to build for a 16-byte router row, which is why my oracle configs use 16
routed experts.

### 0abf8bd bounded caches for the 35 sliding layers + speculative snapshot

`make_cache` chooses by `layer.self_attn.sliding > 0`. `_snapshot_cache_state`
now stores `(state, meta_state)` for key-bearing caches and restores
`meta_state` before `state`; for `SlidingWindowKVCache` the state setter then
recomputes `start_position = offset - _idx`, which equals the snapshot's value
by the invariant above. For `KVCache` the empty meta_state round-trips through
`_BaseCache`. Verified: rollback after 600 + 300 tokens of 8-token windows
(several compactions) for accepted = 0, 1, 3 matches a 138d761 replay
bit-exactly; `from_state` and `prefix_cache_snapshot/restore` of compacted
caches continue bit-exactly (section 4).

### 21e690a strided sconv kernel, tau skip

Kernel: `ensure_row_contiguous=False` with `x/state/w/res` read through the
injected strides (`<name>_strides`), outputs contiguous. Checked the index
math for `x` (batch, row, channel strides), the fp32 state `[B, 3, C]`, the
`[C, K, 1]` weight view (strides `(K, 1, K)` for the checkpoint's transposed
layout, so `w[c*K + k]` as before), and the residual. The L <= 8 loop is
unchanged. Random shapes/strides (B 1-3, L 1-8, C 8-1024, column slices at
random offsets, strided state and residual, fp32 and bf16) are bit-exact
against the 138d761 kernel and 1 ulp from the pure-MLX prefill path with the
state exact (section 4).

Tau: the branch now runs iff `offset + L > log_floor`, where `offset = S - L`
on global layers is the absolute position of the first query, so the largest
1-based query position is `offset + L`. For every position <= floor,
`log(max(qpos/floor, 1)) == 0` exactly and `tau == 1` exactly in fp32 and
after the cast, so `q * tau` and `where(mask > -1e29, mask * tau, mask)` are
identities: skipping is bit-exact, including the position exactly at the floor
(`qpos/floor == 1.0`). Positions crossing the floor mid-chunk (chunks 7/64/513
over a floor of 300) and one-token steps at floor-1/floor/floor+1 match
138d761 bit-exactly, and `mx.log` is called exactly when a position passes the
floor (section 4). Left padding: tau is still computed from column positions
(pre-existing behaviour, unchanged).

### beb844a `logits_to_keep`

Slices the pre-norm hidden state to the trailing `keep` rows before the final
norm (per-token) and the head; hidden states for drafters stay whole; `None`,
0, >= T and negative values pass through. The cache is untouched (the head is
downstream of the stream). Reaches Inkling from `generate_step` (it calls
`model.language_model(...)` directly with `logits_to_keep=1` on both the
chunked and the one-shot prompt step) and from `PromptProcessingBatch`
(`1 + max right pad of unfinished rows`). Note: the VLM wrapper
`Model.__call__` (inkling.py) forwards only `return_hidden/return_shared_kv/
skip_logits`, so a caller going through the top-level model does not get the
reduction (pre-existing, not introduced here). Rows are bit-exact with the
full head (section 4).

### c160787 dense-MLP `global_scale` cast

Bit-exact for the mlx-community checkpoints (bf16 scale: no cast). For an
fp32-scale checkpoint it changes numerics by design: the output is now
rounded to bf16 as every other layer output instead of promoting the residual
stream to fp32. Declared in the PR. The MTP drafter's dense blocks share the
class.

### 9a56b74 kv-bits for the 7 global layers

`InklingCacheList` exposes `offset` and `to_quantized` (only a plain
`KVCache` is converted; `SlidingWindowKVCache` returns `self`, idempotent).
Attention accepts `(packed, scales, biases)` tuples/lists, takes `S` from the
packed keys, builds the mask from the packed shape, groups the mask as
`[B, n_kv, n_rep, L, S]` and routes through the base wrapper with `cache=kv`
so `quantized_scaled_dot_product_attention` runs. The grouping is necessary:
`align_attention_mask_to_scores` would turn a 4-D `[B, H, L, S]` mask into
`[B, 1, H, L, S]` and alias H against `n_rep`. Verified against a dequantized
reference computed from the captured `(q, packed k/v, mask)` at 1e-4 (fp32),
including steps that cross the log floor on a quantized cache (section 4).
Side effect (declared): the layer entry now has `offset`, so
`dispatch._prefix_cache_trim_amount` sees Inkling's real cached length. The
per-turn `PromptCacheState` stores prompt + generated ids; when the next turn's
rendering is an exact extension, `n_drop == 0` and the cache is reused as
before; when the cache is longer than the shared prefix, the entry is not
trimmable (`ArraysCache`) and the turn is cold-prefilled, where 138d761 reused
the stale cache untrimmed (`cached_len` read as 0): a correctness fix that
costs a prefill in that case (`test_prefix_cache_trim_amount_on_inkling_layer_entries`).
Nit: `InklingCacheList.extract` is unreachable: single caches have no
`extract`, and `_make_cache` rebuilds layer entries as plain `CacheList`.

### eb93183 tests

Contract-test entry uses `max(16, 2 * length)` windows, so the APC contract
never exercises eviction (my model-level snapshot test does). The
continuous-batching test is sound; I extended it to three rows and chunk sizes
64/513 against 138d761 single rows.

## 3. Tests run

- K4's new tests on K4's branch (`~/.venvs/inkling`, clone at eb93183, clean
  tree): `test_load_eval_chunks.py` + `test_sliding_window_cache.py` +
  `test_inkling_fast_path.py`: **50 passed** (5 + 18 + 27) in 53 s, default
  TF32 setting (`review/logs/k4-tests.log`; the first scripted attempt logged
  "no tests ran" because of zsh word splitting, the foreground re-run is the
  result).
- Full suite on K4's branch, `pytest -q mlx_vlm/tests -n 2` at eb93183 with
  5 GiB of free host RAM (M1's llama hold resident): **83 failed, 5074 passed,
  16 skipped** in 18 min under load average 20-45
  (`review/logs/pytest-full-eb93183.log`). The 83 are the APC tests in
  `test_cache.py` / `test_generate.py` (`test_model_cache_contract[*]`,
  `test_blocks_and_statistics`, disk/exact-cache tests...) failing with
  `memory_skips=1`: `APCManager._memory_headroom` budgets from
  `_free_ram_bytes()` and no env override bypasses it (`APC_MEMORY_MAX_GB`
  only caps the budget). This reproduces K4's observation; K4's own run with
  free RAM shows 5157 passed / 16 skipped. A re-run of the two files waits
  for >= 40 GiB free RAM (`review/rerun_apc_tests_a4.sh`,
  `review/logs/rerun_apc_tests_a4.log`): **956 passed** in 27 s at 02:03 once
  free RAM exceeded 40 GiB, i.e. the 83 are environmental and the branch's
  suite is 5157 passed / 16 skipped as K4 reports.
  The 16 skips are the same count as K4's run; the suite's gated skips are
  Metal-availability guards (none on this host) and model-download tests.
- `pre-commit` (black, isort, autoflake) on the seven changed files: clean.

## 4. Adversarial tests (branch `a4-review`, `~/.venvs/inkling-a4`)

118 tests, **all pass** with the default TF32 setting and with
`MLX_ENABLE_TF32=0` (`review/logs/pytest-a4-{default,tf32off}.log`, 01:42-01:45,
load average 13-36). Every comparison is against `language.py` at 138d761
running the same weights, unless stated.

| file | what | result |
|---|---|---|
| `test_a4_window_cache.py` (46) | tiny 3-layer model (sliding/global/sliding, 16 routed experts), 600-token prefill at chunks 1/7/64/512/513 + 200 greedy decodes, windows 8 and 512: logits **bit-exact** vs 138d761 at every step; batch of three left-padded rows (600/591/300 tokens) at chunks 7/64/513 + 200-300 decodes vs single rows: within 1e-5 (pad columns change the SDPA reduction); cache-level coverage invariant under 4 random schedules x windows 8/512 x buffer 0/1/256 (`n - L >= min(window-1, offset)` after every update, suffix equality with an unbounded cache, capacity bound); trim inside/beyond the compaction slack (F1); `_prefix_cache_trim_amount` on Inkling entries vs the 138d761 layout; speculative verify(4)/rollback for accepted 0/1/3 after 600 + 300 tokens (several compactions): **bit-exact** vs 138d761 fed the same stream with the same chunking; `from_state` and `prefix_cache_snapshot/restore` of compacted caches: **bit-exact** continuation | pass |
| `test_a4_attention_vs_138d761.py` (49) | `InklingAttention` alone, random weights: sliding layer on `SlidingWindowKVCache` vs `KVCache`, windows 8/512, chunks 1/7/64/512/513 over 1200 tokens (last 150 as single tokens), batch 1/2, fp32/bf16: **bit-exact**; global layer across a log floor of 300 (chunks 1/7/64/513, fp32/bf16): **bit-exact**, and `mx.log` runs exactly when a 1-based position passes the floor; one-token steps at floor-1/floor/floor+1 | pass |
| `test_a4_sconv_random.py` (16) | strided kernel at 16 random (B 1-3, L 1-8, C 8-1024, column offset, residual kind, state present/strided) shapes, fp32/bf16: **bit-exact** vs the 138d761 kernel (output and fp32 state), state exact vs the pure-MLX CPU path and output within 1e-6 (fp32) / 1 bf16 ulp | pass |
| `test_a4_logits_to_keep.py` (5) | kept rows equal the full head within 1e-6 (not bit-equal: a GEMM over fewer rows), pass-through for None/0/T/>T/negative, cache untouched, 1-token step with the kwarg **bit-exact**; `PromptProcessingBatch` with three right-padded rows at prefill 64/513, windows 8/512, 30 greedy tokens per row == 138d761 single-row greedy tokens | pass |
| `test_a4_kv_bits.py` (1) | global layer on `QuantizedKVCache(32, 8)` through 37/1/1/5/1/15-token steps crossing a log floor of 40: the captured quantized SDPA output equals softmax(q k_deq^T/32 + mask) v_deq within 1e-4 under `MLX_ENABLE_TF32=0` (2e-2 under TF32 for multi-row steps, exact to 4e-7 for single rows either way); the 8-bit path is within 5 % (norm) of the float path | pass |
| `test_a4_loader.py` (1) | 1-byte chunks over a parameter tree with inputs shared across chunks equal a one-shot `mx.eval` | pass |

Things the tests taught me (not defects in the branch):

- Three first-draft failures were mine: a capacity assertion that ignored
  that the buffer keeps the capacity of the largest earlier chunk; a rollback
  reference that replayed the stream in one call (the branch and the
  reference must be chunked identically to be bit-exact, since prefill uses
  the unfused conv and decode the fused kernel, 1 ulp apart); and a
  reference that read `q` after upstream's
  `quantized_scaled_dot_product_attention` had scaled it in place
  (`queries *= scale` mutates the shared MLX object).
- fp32 GEMMs with M > 1 run at TF32 by default on this machine (R5 6.4), so
  "exact" comparisons of multi-row quantized attention need
  `MLX_ENABLE_TF32=0`; the branch-vs-138d761 comparisons are bit-exact under
  both settings because both sides execute the same kernels.

## 5. Real model (A4 GPU hold, 02:50-03:13 on 2026-10-29, 23.4 min)

Setup: `mlx-community/Inkling-Small-mlx-2bit` (82.3 GiB), MLX 0.32.4,
`review/a4_hold.sh` + a frozen copy of K4's driver (`review/a4_equiv_bench.py`,
same md5 as `bench/equiv_bench.py`). "before" = the 0.7.11 wheel in
`~/.venvs/inkling-base`; "after" = eb93183 through the `a4-review` worktree
(`git diff eb93183 -- mlx_vlm ':!mlx_vlm/tests'` empty). Both sides: lazy load
+ 4 GiB chunked eval, chat template with reasoning "none", greedy, EOS
ignored, warm-up on p_short, then timed reps; load1 3-7 throughout (a quiet
machine, unlike K4's 13-26). Logs `review/logs/a4-*.log`, rows
`review/results/*.jsonl`, summary `review/logs/a4-results-summary.txt`.

### 5.1 Equivalence of eb93183 with the wheel

- Greedy tokens and top-2 margins: identical on p_short, p_8k (64 tokens, 3
  reps each, self-identical) and p_32k (64 tokens, 3 reps), and for the first
  16 tokens of the eager-load run and the first 64 of the 2048-token run.
- Teacher-forced (the wheel's own 64 tokens fed to both, 2048-token prefill
  chunks, full fp32 log-softmax over the 200058-wide rows saved and hashed):
  **identical matrices** (same md5, KL 0, |dlogprob| 0, top-1 64/64) on
  p_short and p_8k. K4's own A/B hold (02:16-02:46, eb93183) shows the same
  identity on p_short, p_2k, p_8k and p_32k with 128 tokens
  (`bench/results/tf-after.jsonl`).
- Noise floor of the unmodified code under a 512-token chunking: p_8k KL
  mean 2.5e-2, max 0.73 nats (step 18), |dlogprob| max 1.47, top-1 62/64
  (p_short is a single chunk either way: identical). K4's floor rows at 128
  tokens: 1.3e-2..3.3e-2 mean, max up to 1.62. So the branch's zero gap is
  far inside the floor, as a bit-exact change should be.

### 5.2 Speed and memory, same hold, best / median of 3 (load1 in brackets)

| prompt | side | prefill tok/s | TTFT s | decode tok/s | peak GiB |
|---|---|---|---|---|---|
| p_short 239 | before | 164 / 159 | 1.46 / 1.51 | 21.2 / 21.1 | 83.2 |
| p_short 239 | after | 169 / 160 | 1.42 / 1.50 | 24.0 / 24.0 | 83.3 |
| p_8k 8231 | before | 336 / 323 | 24.5 / 25.5 | 18.5 / 17.6 | 93.5 |
| p_8k 8231 | after | 393 / 375 | 20.9 / 22.0 | 20.9 / 19.6 | 91.7 |
| p_32k 32807 | before | 333 / 332 | 98.5 / 98.9 | 20.8 / 19.3 | 105.1 |
| p_32k 32807 | after | 395 / 374 | 83.1 / 87.9 | 19.9 / 19.2 | 95.3 |

(load1 4-7 for every row.) Ratios after/before on the best-of-3: prefill
1.03x / 1.17x / 1.19x, TTFT 0.97x / 0.85x / 0.84x, decode 1.13x / 1.13x /
0.96x, peak +0.1 / -1.8 / -9.8 GiB at short / 8K / 32K.

- Memory: CONFIRMED. 105.1 -> 95.3 GiB at 32K and 93.5 -> 91.7 at 8K
  reproduce K4's 104.9 -> 95.3 and 93.5 -> 91.5 to within 0.2 GiB; the
  2048-token generation on the branch stays at 83.3 GiB.
- Prefill / TTFT: CONFIRMED at 8K and 32K (+17-19 % prefill, -15-16 % TTFT;
  K4: +19 % / +5 % prefill, -16 % / -5 % TTFT). The 32K prefill gain is larger
  here than in K4's run (their before 32K reps ran at load 14-32).
- Decode: NOT CONFIRMED. +13 % at short and 8K, -4 % at 32K, each within
  the rep spread (7-16 %); K4's +29-45 % is not reproduced on a quiet
  machine. K4's interleaved A/B (`bench/results/ab-*.jsonl`) does not settle
  it either: its second pair ran at roughly half the speed of the first on
  both sides (p_32k prefill 450-462 -> 219-272 tok/s before, 468-528 ->
  213-271 after; decode 15.9-18.9 -> 9.3-11.5 before, 24.7-25.2 -> 10.4-17.1
  after), i.e. the machine drifts by 2x within a hold. Absolute numbers
  from different holds are not comparable (my wheel rows are 20-45 % below
  K4's for the same code); only same-hold, adjacent runs are.
- Correctness at 32K: "The narrator is Ishmael, and the ship is the Pequod."
  on both sides, all reps.

### 5.3 kv-bits 8 (branch, global layers quantized, `quantized_kv_start 0`)

Runs end to end on p_short and p_8k with the correct answer; greedy tokens
diverge from the float path at step 17 on both prompts (expected for a lossy
cache). Teacher-forced vs the wheel: p_8k KL mean 1.28e-2, max 0.24 (step
20), |dlogprob| max 0.80, top-1 63/64: inside the chunking noise floor (2.5e-2
/ 0.73 / 62). p_short KL mean 4.05e-2, max 1.89 (step 46), top-1 63/64: here
the floor is degenerate (0), and the gap is at the level of K4's 2K-chunking
floor (3.3e-2 mean, 1.62 max). K4's 128-token rows: 6.6e-3..2.7e-2 mean. So
kv8 is "as good as a different chunking" at 8K and 32K and slightly above
that at a short prompt: acceptable for an opt-in lossy cache, PLAUSIBLE as a
quality claim. Memory: peak 92.1 GiB at 8K vs 91.7 without kv8 and 83.3 =
83.3 at short: the 7 global layers' caches are too small below 32K for 8-bit
to show (K4's 128-token row at 32K: 98.5 vs 95.1 GiB, also higher); the
option costs a little memory and ~5 % prefill at these contexts, and would
only pay off well beyond 64K. Decode 22.3 / 19.5 tok/s (short / 8K).

### 5.4 Loader and stability

- Eager load (`lazy=False`) of the 82.3 GiB checkpoint on the branch: 11.7 s
  (K4: 13.1 s), generation after it identical to the wheel's tokens. The
  wheel's failure is documented in M1's BASELINE.md 2.5 (`GPU Timeout` after
  ~20 s); not re-run here (it would have cost a watchdog abort inside the hold).
- 2048-token greedy generation on the branch with EOS suppressed: finishes
  at 22.7 tok/s, peak 83.3 GiB (flat), coherent English throughout
  (`review/logs/a4-gen2k-decoded.txt`; the low distinct-4-gram ratio is the
  model re-answering after each suppressed `<|end_message|>`, not garbage).
  K4's 64K run (100.4 GiB, correct answer) stands as their evidence for the
  48K+ case; not repeated within my budget.

## 6. Benchmark skepticism (K4's own numbers, re-read from `bench/results/*.jsonl`)

What the raw rows say (my re-analysis, `review/logs/k4-results-reanalysis.txt`):

- Both sides ran the same driver, `lazy=True` + the driver's own 4 GiB chunked
  eval, the same chat template (reasoning "none"), EOS ignored, 128 new tokens,
  one warm-up on p_short, then 3 timed reps per prompt (2 for the ablations and
  64K). `before` imported the wheel from `~/.venvs/inkling-base` (meta row), the
  `after*` rows imported the clone at 355ea6f / 0abf8bd / 21e690a. The baseline
  was not handicapped by the setup.
- Equivalence: I re-compared the raw `token_ids` and `margins` arrays: for
  p_short/p_2k/p_8k/p_32k all 129 entries (128 tokens + the stream's closing
  duplicate of the last token) and all 129 top-2 margins are identical between
  `before` and `after`, `after-window-only`, `after-kernels`. The eager-load row
  "differs" only at its 33rd entry, which is the closing duplicate of a
  32-token run. Within-process reps are self-identical everywhere.
- Load: `before` ran at load1 14-32 (p_32k reps: 32.5 / 20.5 / 14.2), `after`
  at 6-13 (p_32k: 9.0 / 7.8 / 6.2). The decode column is therefore confounded
  exactly as K4 says; the before p_32k decode spread (10.7-20.0 tok/s) is
  wider than the claimed effect. K4 marks it unresolved; the PR body still
  prints the decode column with "load-dominated" only on the first row.
- Prefill/TTFT at 2K/8K (+20 %) ran under the same load asymmetry (before 14-18
  vs after 11-12), so part of that gain may be load too. The ablation rows
  (window-only 443 vs kernels 510 tok/s at 32K, same hold, similar load
  15-18) are the cleaner evidence that the kernel commit carries the prefill
  gain. My hold re-measures p_short/p_8k/p_32k before/after with load recorded.
- Memory is the robust signal: peak `mx.get_peak_memory` is identical across
  reps (104.7-104.9 vs 95.3 GiB at 32K; 93.5 vs 91.5 at 8K) and reproduces
  the arithmetic (4.7 GB of sliding keys + the tau temporaries).
- The 64K row: the wheel was not run ("not attempted" in README; "wheel: does
  not fit" in the PR table). That cell is a projection and must be labelled so.
- kv-bits: `after-kv8` at 355ea6f crashed with the `NotImplementedError` the
  branch itself raises for list-typed quantized keys (fixed in eb93183 by
  widening the `isinstance`); the real-model kv8 result in K4's package is
  therefore absent, and the teacher-forced runs (`run_ab.sh`) had not executed
  when this review started. Covered by my hold.
- Token counts and warm-up are right; `prefill tok/s` is the API's
  `prompt_tokens / prompt_time` and TTFT is wall time to the first yielded
  token, both consistent across sides.

## 7. Upstream fit

- Base: `origin/main` is still 138d761 at review time (fetched 2026-10-29 01:20),
  so the branch applies cleanly. No open PR or issue in Blaizzy/mlx-vlm
  touches `models/inkling/*` or adds a window cache; the only open Inkling
  issue is #2526 (`convert --mtp` drops `q_mode`, unrelated). Recently merged
  and adjacent: #2462 (Inkling Metal gating, kept), #2548 (rotating-cache
  absolute positions, already in base), #2510 (strided K/V reads in ragged
  decode, same idea as the strided sconv kernel). #2206 (speculative decoding
  rebuilt around request caches, open since 2026-09-18, by the maintainer)
  would replace the Inkling adapter's `speculative_verify_hidden`/
  `rollback_speculative_cache` path where commit 4's `meta_state` snapshot
  lives; if it lands first, the window cache's `trim`/`state`/`meta_state`
  contract is what the new transaction model needs anyway (see F1). The
  maintainer's own open #2505 (LFM2.5 MoE + fused short-convolution decode)
  and #2614 (LFM3 native sliding-window attention) show active interest in
  exactly this area, which helps the pitch and raises the chance of a
  conflicting kernel helper landing first; worth checking at submission time.
- Style: `pre-commit run` (black 26.3.1, isort 5.13.2, autoflake 2.2.1) passes
  on all seven changed files (`review/logs` / my run in the a4 worktree).
  No PR template in the repo; CONTRIBUTING.md asks for `pytest -q mlx_vlm/tests`
  and sanitize idempotency (not touched here).
- Size and split: +1212/-32 over 7 files, nine commits mixing a generic loader
  change, two generic cache classes, Inkling-only changes and a new feature
  (kv-bits). The maintainer's recent approvals in this area are 2-file,
  ~200-line PRs (#2510 approved without comments, #2462 one comment).
  Recommended split: (A) commit 1 alone (generic, trivially reviewable, fixes
  a real failure to load); (B) commits 2 + 3 + 4 + 9 (window caches and the
  Inkling wiring, with the contract-test entry and the batching test);
  (C) commits 5 + 6 + 7 (kernel-count and head reductions, all bit-exact);
  (D) commit 8 (kv-bits, a behaviour change with its own validation). (C)
  could ride with (B) if the maintainer prefers fewer PRs.
- PR.md accuracy: the per-item exactness table matches what I verified; two
  wording fixes are needed (64K "wheel: does not fit" is unmeasured; the
  decode column needs the A/B result or a "not resolved" label on every row),
  and the validation paragraph should cite the final commit, since the
  equivalence runs were taken at 355ea6f (same generation code, the fixup
  widens an `isinstance`). AI assistance is disclosed plainly in a
  "Disclosure" section and every commit carries a `Co-Authored-By` trailer.

## 8. Defects, required and recommended changes

No BROKEN item on the default single-sequence path: the branch reproduces
138d761 bit for bit on the tiny model (prefill at chunks 1/7/64/512/513,
decode across window compactions, batch rows, rollback) and on the real
model (section 5). Findings, by weight:

Required before submission (wording and evidence, not code):

0. PR.md: drop the "decode tok/s" gains from the claims (keep the column
   only with a "not resolved" note): neither K4's interleaved A/B (2x drift
   mid-hold) nor my quiet hold (+13 / +13 / -4 %, inside the spread) supports
   +29-45 %. Prefill, TTFT and memory carry the PR on their own.

1. PR.md: label the 64K "wheel: does not fit" cell as not measured
   (projected from the 32K peak), or run it.
2. PR.md: the decode column is confounded by load (before at load 14-32,
   after at 6-13); either report the interleaved A/B (`run_ab.sh`) or label
   every decode cell "not resolved" and keep prefill/TTFT/memory as the
   claims. My hold's numbers (section 5) give the same load caveat.
3. PR.md: the validation paragraph should name the commit the equivalence
   runs were taken at (355ea6f) or be re-run at the final commit; my hold
   re-validates eb93183.
3b. PR.md item 8: say that kv-bits 8 saves no memory below ~64K on this
   model (7 global layers) and costs ~5 % prefill; its value is at very long
   contexts, and its quality gap is at the level of a different prefill
   chunking (5.3).

Recommended (code, small):

4. `SlidingWindowKVCache.trim` / `BatchSlidingWindowKVCache.trim`: refuse or
   clamp-and-report a trim deeper than `offset - (start_position + max_size - 1)`
   (F1). Today only `dispatch._cache_fully_retained` protects callers; a
   future trim site (e.g. #2206's transaction commit, or a server
   compaction) would silently narrow the window. Also strengthen
   `test_trim_rolls_back_like_the_reference` to assert window coverage after
   the trim, not only the suffix.
5. Document that `BatchSlidingWindowKVCache` bounds memory only after
   `finalize()` (F2); the batched prompt phase keeps the whole prompt.
6. Remove the unreachable `InklingCacheList.extract` or make `_make_cache`
   rebuild layer entries as `InklingCacheList` so the batch path keeps
   `offset`/`to_quantized` (then kv-bits with continuous batching could be
   supported later instead of refused).
7. Note in the kv-bits commit that `quantized_scaled_dot_product_attention`
   scales its `queries` argument in place (MLX augmented assignment mutates
   the shared object); harmless in `InklingAttention` today because `q` is
   not reused, but worth a copy or a comment since the Inkling call site is
   new.

Nits: `_snapshot_cache_state` now evaluates `meta_state` strings into the
snapshot tuple (fine); the window constructor table entry in
`test_cache.py` never evicts (window = 2 x length), so the APC contract
test does not cover a compacted window (my model-level
`test_snapshot_round_trips_mid_window_at_model_level` does).

## 9. Raw logs and files (all under `package/k4-mlx-vlm-fast-path/review/`)

- `logs/k4-tests.log`, `logs/pytest-full-eb93183.log`: K4's branch, full suite.
- `logs/a4-tests.log`, `logs/pytest-a4-default.log`, `logs/pytest-a4-tf32off.log`: the A4 tests.
- `logs/rerun_apc_tests_a4.log`, `logs/pytest-cache-generate-eb93183-a4.log`: the APC-gated files with free RAM.
- `logs/k4-results-reanalysis.txt`: re-read of K4's raw benchmark rows.
- `a4_hold.sh`, `a4_equiv_bench.py` (frozen copy of K4's driver, md5 4604f6fe...), `logs/a4_hold.log`, `logs/a4-*.log`, `results/*.jsonl`, `results/*-tf-*.npy`: the GPU hold.
- `analyze_a4.py`: the summary printer for `results/`.
- Branch `a4-review`: `mlx_vlm/tests/_a4_common.py`, `_a4_language_138d761.py` (the oracle), `test_a4_*.py`.
