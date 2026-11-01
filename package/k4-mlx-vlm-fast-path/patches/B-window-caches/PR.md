# PR B: Bounded window caches for Inkling's sliding-window layers

Target: `Blaizzy/mlx-vlm` main (138d761). Patches `0002`-`0006` (5 commits: `cache.py`,
`models/inkling/language.py`, `tests/test_sliding_window_cache.py`,
`tests/test_inkling_fast_path.py`, `tests/test_cache.py`; +801/-9). Applies on main alone
(independent of PR A). Draft, not submitted.

**Problem.** Inkling-Small's 35 sliding layers only ever read their last 511 keys (the
slice in `InklingAttention`), but `make_cache` gave them an unbounded `KVCache`: 4.7 GB of
unreadable keys at 32K context, 18.8 GB at 128K, plus a 256-step reallocation of the whole
buffer per layer. `RotatingKVCache` cannot serve these layers: it returns the ring out of
temporal order once it wraps, and the distance-indexed bias needs order.

**Change.**
- `SlidingWindowKVCache` (on `BufferedRotatingKVCache`): temporal order, in-place
  compaction every `buffer_size` tokens, one contiguous view the model slices to its
  window; trimmable (refuses a trim that would rewind past the compaction slack),
  snapshot/`from_state`, APC merge; `BatchSlidingWindowKVCache` for continuous batching
  (left/right padding, extend/filter/extract/merge; during a right-padded prompt the whole
  prompt is kept and the bound applies after `finalize()`). One-line fix in
  `BufferedRotatingKVCache`'s state setter so `from_state` works for it.
- Inkling sliding layers use the window cache, global layers keep `KVCache`; the
  speculative snapshot carries `meta_state`; the route kernel indexes its logits directly
  (tiny MoE configs failed to compile the Metal kernel; production shapes unaffected).
- Attention code unchanged: same keys, same order, same mask offset.

**Exactness.** Designed bit-exact; verified: identical greedy tokens, top-2 margins and
full fp32 log-softmax matrices (teacher-forced) on 239/2085/8231/32807-token prompts
versus the 0.7.11 wheel (K4 runs at 355ea6f, A4's re-validation at eb93183, same code
as this series). A4: 46 + 49 adversarial tests against `language.py` at 138d761
(chunks 1/7/64/512/513, windows 8/512, batch rows, rollback, snapshots, fp32/bf16).

**Measured (M5 Max 128 GB, Inkling-Small-mlx-2bit).** Peak memory at 32K context
104.9 -> 101.1 GiB from this PR alone (95.3 with PR C); 64K context fits at 100.4 GiB
on the full series (the wheel was not run at 64K: projected not to fit from its 32K
peak of 104.9 GiB). Decode speed: not resolved (shared machine, see the index README).

**Disclosure.** Written with AI assistance (Claude), reviewed, tested and measured by a
human-directed workflow; every commit carries a `Co-Authored-By` trailer.
