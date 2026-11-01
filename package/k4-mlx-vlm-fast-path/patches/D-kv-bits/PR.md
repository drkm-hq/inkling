# PR D: Make `--kv-bits` reach Inkling's global layers (or fail clearly)

Target: `Blaizzy/mlx-vlm` main after PRs B and C (patches `0010`-`0012`, 3 commits,
`models/inkling/language.py` + `tests/test_inkling_fast_path.py`, +146/-8). Depends on
B (window caches stay in floats) and on C's attention hunks. Draft, not submitted.

**Problem.** `--kv-bits` was a silent no-op for Inkling under the default uniform scheme
(`maybe_quantize_kv_cache` only converts entries exposing `offset`/`to_quantized`; the
per-layer `CacheList` has neither) and crashed under TurboQuant (the attention read
`.shape` on a state proxy).

**Change.** Layers use `InklingCacheList`, a `CacheList` exposing the kv cache's `offset`
and a `to_quantized` that converts a global layer's `KVCache` to `QuantizedKVCache` and
leaves a sliding layer's window cache alone. The attention accepts the (packed, scales,
biases) tuples or lists, slices them like float keys, groups the per-head mask as
`[B, n_kv, n_rep, L, S]` (the generic aligner would alias heads against kv heads) and
routes through the base SDPA wrapper so the quantized matmuls run. Split key/value
policies work (HybridQuantKVCache returns floats); pure TurboQuant raises a clear
`NotImplementedError`; kv-bits with continuous batching keeps the generic refusal.
Side effect: with `offset` exposed, a per-turn prompt cache longer than the shared prefix
is cold-prefilled instead of being reused untrimmed (a correctness fix).

**What to expect.** Quality: teacher-forced against the wheel's own 128 tokens, KL mean
0.7-2.7e-2 nats (max 0.15-0.73), 0-5 top-1 flips per 128, which is the level of the
stock implementation's own difference between 512- and 2048-token prefill chunks
(mean 1.3-3.3e-2, max up to 1.62, 3-6 flips). Greedy outputs do diverge from the float
path (at token 17 on both tested prompts), as any lossy cache does. Memory: no saving
below roughly 64K context on this model (7 global layers of 32K keys are 0.9 GB; the
generic quantized attention materializes its scores, so the 32K peak was 98.5 GiB with
kv8 vs 95.3 without) and about 5 % slower prefill; the value is at very long contexts.
This PR is a correctness and "no silent no-op" fix, not a speed or memory feature.

**Disclosure.** Written with AI assistance (Claude), reviewed, tested and measured by a
human-directed workflow; every commit carries a `Co-Authored-By` trailer.
