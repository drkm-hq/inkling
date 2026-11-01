# PR A: Evaluate eagerly loaded parameters in bounded chunks

Target: `Blaizzy/mlx-vlm` main (138d761). Patch: `0001-*.patch` (1 commit, `utils.py`
+ `tests/test_load_eval_chunks.py`, +113/-1). Applies on main alone. Draft, not submitted.

**Problem.** `load_model_weights` runs one `mx.eval` over every parameter of an eager
load. For an 80+ GB checkpoint that is a single Metal command buffer, which exceeds the
GPU watchdog budget on a 128 GB M5 Max: `mlx-community/Inkling-Small-mlx-2bit`
(82.3 GiB) aborts with `GPU Timeout` from inside `load_model_weights` (M1's
BASELINE.md 2.5).

**Change.** `eval_parameters(parameters, chunk_bytes=None)` walks the parameter tree and
evaluates at most `EVAL_CHUNK_BYTES` (4 GiB, a module constant read at call time) per
`mx.eval`. The end state is identical (every array evaluated once); models smaller
than the chunk still take exactly one eval; `lazy=True` is unchanged.

**Evidence.** The 82.3 GiB checkpoint loads with `lazy=False` in 13.1 s (reviewer: 11.7 s)
and generates the same tokens as the lazy+chunked load. Tests spy on `mx.eval`: every
parameter evaluated exactly once under small chunk bounds, eager `load_model_weights`
covers all loaded parameters, lazy loads evaluate nothing. Reviewer A4: CONFIRMED
(`test_a4_loader.py`: 1-byte chunks over a tree with inputs shared across chunks equal a
one-shot eval).

**Disclosure.** Written with AI assistance (Claude), reviewed, tested and measured by a
human-directed workflow on the machine above; the commit carries a `Co-Authored-By` trailer.
