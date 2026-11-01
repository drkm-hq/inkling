# Pull-request text (combined view; target Blaizzy/mlx-vlm main at 138d761)

**Status: draft, not submitted; nothing pushed.** The series is `inkling-fast-path`
at `ee1be25` (12 commits), split into four PRs for review (A4's recommendation):
`patches/A-loader`, `patches/B-window-caches`, `patches/C-decode-kernels`,
`patches/D-kv-bits`, each with its own fact sheet. This file is the combined narrative.
Equivalence runs were taken at `355ea6f` (K4) and re-validated by the independent
reviewer at `eb93183`; `ee1be25` has the same generation code plus two review
commits (a trim guard on the window cache with tests, and comments/dead-code removal
in the kv-bits commit) and passes the same tests.

## Title (combined)

Inkling: bounded window caches, fewer decode kernels, kv-bits for global layers, and a chunked loader

## Body

This series makes Inkling-Small (276B-A12B) on mlx-vlm leaner at long context and
fixes four latent problems found while profiling it on an M5 Max (128 GB). Every
change except kv-bits is bit-exact for the shipped checkpoints: greedy 128-token
outputs on 239 / 2085 / 8231 / 32807-token prompts are identical before and after,
and so are the full teacher-forced fp32 log-softmax matrices (see "Validation").

Commits, in series order:

1. **Evaluate eagerly loaded parameters in bounded chunks** (`utils.py`, PR A).
   `load_model_weights` ran one `mx.eval` over every parameter; for an 82 GiB
   checkpoint that single Metal command buffer trips the GPU watchdog on a 128 GB
   M5 Max (`GPU Timeout`), so `mlx-community/Inkling-Small-mlx-2bit` could not be
   loaded with `lazy=False`. `eval_parameters` evaluates at most `EVAL_CHUNK_BYTES`
   (4 GiB) per `mx.eval`; smaller models still take one eval.
2. **Add bounded, temporally ordered sliding-window KV caches** (`cache.py`, PR B):
   `SlidingWindowKVCache` and `BatchSlidingWindowKVCache`, for layers that build
   their own attention bias from query/key distance (the circular `RotatingKVCache`
   returns its ring out of order once it wraps). One-line fix in
   `BufferedRotatingKVCache`'s state setter so `from_state` works for it.
3. **Inkling: index router logits directly in the route kernel** (PR B). MLX binds
   inputs under 32 bytes in constant memory; the kernel took a `device` pointer into
   them, so tiny MoE test configs failed to compile. Production shapes unaffected.
4. **Inkling: bounded window caches for the sliding-window layers** (PR B). The 35
   sliding layers only read their last 511 keys but had an unbounded `KVCache`:
   4.7 GB of unreadable keys at 32K, 18.8 GB at 128K, plus 256-step reallocations.
   The speculative snapshot now carries `meta_state`.
5. **Window caches in the cache contract test and under continuous batching** (PR B,
   tests): contract-table entry; `PromptProcessingBatch`/`GenerationBatch` with
   right-padded prompts reproduce single-row greedy tokens.
6. **Refuse window-cache trims that would rewind past the compaction slack** (PR B):
   `trim` raises when the cache has evicted and `n` exceeds the retained slack, instead
   of silently narrowing the window while `is_trimmable()` is True; the batch form too.
   During a right-padded prompt the batch cache keeps the whole prompt and bounds from
   the first update after `finalize()` (documented).
7. **Inkling: strided short-conv kernel and no log-scaling passes below the floor**
   (PR C). Two contiguous copies per layer per token removed; tau (exactly 1 below
   `log_scaling_n_floor` = 128000) no longer computed and applied, which also removes
   three mask-sized passes per global layer per prefill chunk.
8. **Inkling: honor `logits_to_keep`** (PR C).
9. **Inkling: keep the dense MLP output scale in the activation dtype** (PR C; H1).
10. **Inkling: quantized KV cache for the global layers** (PR D; H2). `--kv-bits`
    silently did nothing (uniform) or crashed (TurboQuant); now the 7 global layers'
    caches are quantized through the base SDPA wrapper, split policies work, pure
    TurboQuant fails clearly, kv-bits with continuous batching keeps the generic
    refusal. Side effect: with `offset` exposed on the layer entry, a per-turn prompt
    cache longer than the shared prefix is cold-prefilled instead of reused untrimmed.
11. **Inkling: test the split key/value kv quantization policy** (PR D, test).
12. **Inkling: note the in-place query scaling of the quantized SDPA, drop dead
    extract** (PR D).

Not done, on purpose: folding the top-k weighted sum / `yr + ys` / the muP `h / 16`
into neighbouring ops (not bit-exact, and the norm-weight fold is not
idempotency-safe in `sanitize`); a fused prefill short-conv kernel (cannot keep the
prompt-phase outputs bit-identical against `nn.Conv1d`'s accumulation order).

### Exactness, per item

| item | designed bit-exact? | why / how verified |
|---|---|---|
| 1 chunked loader | yes | same arrays evaluated, only the eval batching changes; identical tokens and logprobs |
| 2+4+6 window caches | yes | the attention reads exactly the same keys in the same order (the slice to `L + 511` already existed); identical tokens and logprobs; A4: bit-exact vs 138d761 at chunks 1/7/64/512/513 |
| 3 route kernel indexing | yes | same arithmetic, pointer replaced by an index |
| 7 strided sconv kernel | yes | same loop and casts, only the address computation changes; bit-equal to the old kernel on strided inputs (K4: production shapes; A4: 16 random shapes) |
| 7 tau skip | yes | the skipped branch multiplies by exactly 1.0 and applies `where` with an all-true predicate; A4: bit-exact across the floor |
| 8 `logits_to_keep` | yes on the single-sequence path (`L == 1`, nothing is sliced); multi-row prompts run the head GEMM over fewer rows (same math, within 1e-6) |
| 9 `global_scale` cast | yes for the shipped checkpoints (bf16 scalar: no-op); an fp32 scalar is now rounded to bf16 before the multiply instead of promoting the stream, by design |
| 10 kv-bits | no (8-bit keys/values): judged by teacher-forced KL against the stock implementation's own chunk-size noise floor |

### Validation

Machine: MacBook Pro M5 Max, 128 GB, macOS 26.5.1, MLX 0.32.4; checkpoint
`mlx-community/Inkling-Small-mlx-2bit` (82.3 GiB). "Before" is the released 0.7.11
wheel (byte-identical to 138d761 for every Inkling file), "after" the branch. Greedy,
temperature 0, 128 new tokens, EOS ignored, chat template with reasoning effort
"none"; warm-up once, then the median of 3 timed runs.

Equivalence (bit-exact items): on p_short (239 tokens), p_2k (2085), p_8k (8231) and
p_32k (32807), all 128 greedy token ids and the top-2 logit margin at every step are
identical before and after (K4 at 355ea6f; A4 at eb93183 on 239/8231/32807 tokens).
Teacher-forced (the wheel's own 128 tokens fed to both, 2048-token prefill chunks):
the full fp32 log-softmax matrices (128 x 201024) are bit-identical on all four
prompts (KL 0; A4 confirms on 239/8231). The wheel's own noise floor between 512- and
2048-token chunks is KL mean 1.3-3.3e-2 nats, max 0.26-1.62, 3-6 top-1 flips per 128
steps (p_2k/p_8k/p_32k). Loader: the 82.3 GiB checkpoint loads with `lazy=False` in
13.1 s (A4: 11.7 s); the wheel trips the GPU watchdog.

Performance (median of 3; shared machine, load average 13-26):

| prompt | prefill tok/s before -> after | TTFT s | decode tok/s | peak GiB |
|---|---|---|---|---|
| 239 | 284 -> 306 | 0.85 -> 0.78 | not resolved | 83.2 -> 83.3 |
| 2085 | 470 -> 566 | 4.45 -> 3.70 | not resolved | 87.4 -> 87.8 |
| 8231 | 509 -> 606 | 16.2 -> 13.6 | not resolved | 93.5 -> 91.5 |
| 32807 | 459 -> 484 | 71.5 -> 67.9 | not resolved | 104.9 -> 95.3 |
| 65574 | wheel not measured | 139.5 (after) | not resolved | 100.4 (after) |

The 64K row is the branch only; the wheel was not run at 64K (projected not to fit:
its 32K peak is already 104.9 GiB and the unbounded caches plus the tau temporaries
grow with context). A4's quiet hold (load 4-7) reproduces memory (105.1 -> 95.3 GiB at
32K, 93.5 -> 91.7 at 8K) and prefill/TTFT (+17-19 % prefill, -15-16 % TTFT at 8K/32K;
short-prompt prefill +3 % is noise). Decode: not resolved. K4's interleaved
wheel/branch/wheel/branch A/B swung 2x between consecutive processes of the same code
(pooled medians 21.9 vs 20.2 tok/s at short context, 13.7 vs 20.9 at 32K) and A4's
quiet hold measured +13 % / +13 % / -4 % at short / 8K / 32K, inside the rep spread;
no regression is evidenced and the expected gain is small by construction (two copy
kernels per layer and a dozen tau ops per global layer per token, plus no 256-step
cache reallocation at long context). Prefill, TTFT and memory carry the series.

Per-commit attribution at 32K (same hold): window caches alone take the peak from
104.9 to 101.1 GiB; the strided sconv + tau skip commit takes it to 95.3 GiB and
carries the prefill gain (459 -> 510 tok/s in the ablation, 484 in the full run).

kv-bits 8 (global layers quantized): teacher-forced KL against the wheel mean
0.7-2.7e-2 nats, max 0.15-0.73, 0-5 top-1 flips, i.e. at the stock chunking floor
(A4: 1.3e-2 mean at 8K inside the 2.5e-2 floor); greedy outputs diverge at token 17
on both tested prompts, as a lossy cache does. It saves no memory below roughly 64K on
this model (7 global layers; 32K peak 98.5 GiB with kv8 vs 95.3 without, because the
generic quantized attention materializes its scores) and costs about 5 % prefill; its
value is at very long contexts. It is a correctness and "no silent no-op" fix.

Unit tests: `pytest -q mlx_vlm/tests` at eb93183: 5157 passed, 16 skipped (the APC
tests budget from free host RAM and fail while another 100 GB model is resident; run
with free RAM). At ee1be25 the Inkling, cache, loader, speculative and kernel-gate
files pass at every PR-group boundary (A, A+B, A+B+C, all). New tests:
`test_load_eval_chunks.py` (5), `test_sliding_window_cache.py` (19),
`test_inkling_fast_path.py` (28); `pre-commit` (black, isort, autoflake) clean.

### Disclosure

This series was written with AI assistance (Claude, driven by a human reviewer who
owns the design choices, the measurements and the final review), and independently
reviewed (A4, `REVIEW.md`). All tests and benchmarks were run on the machine above.
