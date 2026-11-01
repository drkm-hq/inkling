# PR C: Fewer kernels per Inkling step (strided sconv, tau skip, logits_to_keep, scale dtype)

Target: `Blaizzy/mlx-vlm` main after PR B (patches `0007`-`0009`, 3 commits,
`models/inkling/language.py` + `tests/test_inkling_fast_path.py`, +217/-16). The code
hunks are independent of PR B; the test file is shared, so apply after B (or let the
maintainer take B+C together). Draft, not submitted.

**Changes, all bit-exact by design.**
1. Short-conv decode kernel declared `ensure_row_contiguous=False` and reads `x`, state,
   weight and residual through their strides: the k/v streams are column slices of the
   fused qkvr projection and were copied to contiguous buffers before every call (two
   kernels per layer per token). Same loop and casts; bit-equal to the previous kernel on
   strided inputs at the production shapes.
2. The log-scaling factor tau is exactly 1 for every query position up to
   `log_scaling_n_floor` (128000 for the shipped checkpoints) but was computed and
   applied anyway: a dozen small kernels per global layer per token and three
   mask-sized passes per global layer per prefill chunk (12.9 GB of temporaries for an
   8192-token chunk). Skipped while `offset + L <= floor`; unchanged past it.
3. `logits_to_keep` honored: multi-token steps no longer run the 201024-wide head over
   the whole input before the caller slices the last row. Bit-exact on the
   single-sequence path (nothing is sliced there); multi-row prompts run the head GEMM
   over fewer rows (same math, within 1e-6).
4. The dense MLP's `global_scale` is cast to the activation dtype: an fp32 scalar (as in
   the reference checkpoint) promoted the residual stream to fp32 for every later layer
   (about 5x slower per token); no-op for the mlx-community bf16 scalars.

**Evidence.** Identical tokens, margins and teacher-forced fp32 logprob matrices versus
the wheel on four prompts (355ea6f / eb93183). Peak memory at 32K 101.1 -> 95.3 GiB on
top of PR B; prefill +17-19 % and TTFT -15-16 % at 8K/32K in the reviewer's quiet hold
(K4: +19 % / +5 %). A4: 16 random strided-shape kernel tests bit-exact vs the 138d761
kernel; tau crossing the floor mid-chunk bit-exact.

**Disclosure.** Written with AI assistance (Claude), reviewed, tested and measured by a
human-directed workflow; every commit carries a `Co-Authored-By` trailer.
