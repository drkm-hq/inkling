# K4: mlx-vlm Inkling fast path (branch `inkling-fast-path`)

Target: `Blaizzy/mlx-vlm` at 138d761 (clone: `~/code/inkling-contrib/upstream/mlx-vlm`,
branch `inkling-fast-path` at **`ee1be25`**, 12 commits, nothing pushed). The series is
split into four PRs for the maintainer's review style (A4, REVIEW.md section 7); each
group has its patches and a short fact sheet under `patches/`:

| PR | directory | commits (series order) | applies on | what |
|---|---|---|---|---|
| A | `patches/A-loader` | `d6b2552` | main | `load_model_weights` evaluates parameters in 4 GiB chunks (82 GiB loads with `lazy=False`; wheel: GPU Timeout) |
| B | `patches/B-window-caches` | `71eab0a` `4a951b5` `f5f4543` `d13d4c4` `a261464` | main | `SlidingWindowKVCache`/`BatchSlidingWindowKVCache`, Inkling sliding layers bounded, route-kernel indexing, contract/batching tests, trim guard |
| C | `patches/C-decode-kernels` | `982cc5a` `d0ea67c` `fc5504f` | after B (shared test file) | strided sconv kernel, tau skip, `logits_to_keep`, `global_scale` cast |
| D | `patches/D-kv-bits` | `aca5359` `968125a` `ee1be25` | after B and C | `--kv-bits` reaches the global layers or fails clearly; split-policy test; notes |

The equivalence runs below were taken at `355ea6f` (K4) and re-validated at `eb93183`
(A4's review); `ee1be25` is the same generation code reordered into the four groups plus
two review commits (the trim guard with tests, the kv-bits comment and dead-code
removal). The pre-split linear history is kept as `inkling-fast-path-linear`.
Combined PR text: `PR.md`. Raw benchmark rows: `bench/results/*.jsonl`, logs:
`bench/logs/`, tables: `bench/results/COMPARE.md`. Independent review: `REVIEW.md`
(A4: no BROKEN item; bit-exactness, memory and prefill/TTFT confirmed; decode not
confirmed; kv-bits plausible, no memory saving below ~64K, about -5 % prefill).

Not done (see PR.md): MoE epilogue folds and the muP fold (not bit-exact), the
prefill sconv kernel (cannot be bit-exact against `nn.Conv1d`'s accumulation order).

## Exactness by design

Items 1-7 are designed bit-exact (same arrays, same math order; see the per-item
table in PR.md) and are judged by identical tokens and identical per-token
logprobs. Item 8 (kv-bits) is not, and is judged teacher-forced: the same 128
tokens are fed to every run and the per-step KL against the stock implementation
is compared with the noise floor of the stock implementation at a different prefill
chunk size (512 vs 2048). `bench/run_ab.sh` runs that (`--teacher-forced`), and
`bench/compare.py` prints the section.

## Equivalence (bit-exact)

`before` = the released 0.7.11 wheel (`~/.venvs/inkling-base`), `after` = the branch at
355ea6f (K4's runs; same generation code as eb93183, which A4 re-validated, and as the
final ee1be25). Greedy, temperature 0, 128 new tokens, EOS ignored, chat
template with reasoning effort "none". On p_short (239 tokens), p_2k (2085), p_8k
(8231) and p_32k (32807): **all 128 generated token ids identical, and the top-2
logit margin identical at every step** (including 1/1/5/1 exact bf16 ties on the
paths, which fall on the same steps). Runs within a process are self-consistent.
Teacher-forced (second hold, 02:16): feeding the wheel's 128 tokens to both
implementations with 2048-token prefill chunks, the full fp32 log-softmax
matrices (128 x 201024 per prompt) of the branch are **bit-identical** to the
wheel's on all four prompts (equal md5, KL 0, max |dlogprob| 0;
`bench/results/tf-after.jsonl`). The wheel's own teacher-forced argmax matches
its greedy tokens on only 123/125/124/128 of 128 steps (short/2K/8K/32K): the
128-token prefill chunk (GEMM) and token-by-token decode (GEMV) accumulate in
different orders, so near-tie steps flip; that is the stock implementation's
own path noise, not a property of this branch.

Noise floor of the stock implementation (wheel, 512- vs 2048-token prefill
chunks, teacher-forced on the same tokens; `bench/results/tf-before-512.jsonl`):

| prompt | KL(2048 || 512) mean | max (step) | mean abs dlogprob of the teacher token | max | top-1 agree |
|---|---|---|---|---|---|
| p_short | 0 (one chunk either way) | 0 | 0 | 0 | 128/128 |
| p_2k | 3.3e-2 | 1.62 (120) | 8.1e-2 | 4.83 | 124/128 |
| p_8k | 1.8e-2 | 0.26 (18) | 4.5e-2 | 0.77 | 123/128 |
| p_32k | 1.3e-2 | 0.30 (17) | 4.5e-2 | 0.95 | 122/128 |

So a non-bit-exact change is a defect only if its KL exceeds roughly 1e-2 nats
mean / 1 nat max; token-level divergence alone proves nothing on this model.

kv-bits 8 on the branch (item 8: 8-bit keys and values on the 7 global layers,
window caches untouched), teacher-forced against the same wheel reference
(`bench/results/tf-after-kv8.jsonl`):

| prompt | KL(wheel || kv8) mean | max (step) | mean abs dlogprob | max | top-1 agree | argmax = wheel's greedy token |
|---|---|---|---|---|---|---|
| p_short | 1.6e-2 | 0.35 (56) | 4.3e-2 | 0.47 | 126/128 | 125/128 (wheel itself: 123) |
| p_2k | 1.5e-2 | 0.36 (50) | 4.7e-2 | 0.52 | 126/128 | 125/128 (wheel: 125) |
| p_8k | 2.7e-2 | 0.73 (17) | 6.7e-2 | 1.66 | 123/128 | 123/128 (wheel: 124) |
| p_32k | 6.6e-3 | 0.15 (17) | 2.3e-2 | 0.45 | 128/128 | 128/128 (wheel: 128) |

Every row sits inside the stock implementation's chunk-size floor above, so the
quantized global-layer cache is as faithful as the wheel is to itself. It is not
a memory win on this model, though: the 7 global layers' keys are only 0.9 GB at
32K, and the generic quantized attention materializes its scores (no fused
kernel), so the teacher-forced peak at 32K was 98.5 GiB with kv8 vs 95.3 GiB for
the branch's float caches (wheel: 104.9). kv-bits on Inkling is a correctness
and "no silent no-op" fix, as the brief asked, not a speed or memory feature.

## Measured (one GPU hold, 00:00-00:28 on 2026-10-29, median of 3, min-max in parentheses)

The machine was not quiet: load average 13-26 from other jobs' CPU work during
every run. Decode on this model is CPU-dispatch bound (one Python-driven kernel
launch chain per token), so decode tok/s tracks the load of the minute it ran in;
prefill, TTFT and memory are GPU/allocation bound and much steadier. The decode
column is not resolved: the interleaved A/B below drifted 2x between consecutive
processes of the same code, and A4's quiet hold saw +13 / +13 / -4 % inside the rep
spread. Prefill, TTFT and memory carry the series.

| run | prompt | prefill tok/s | TTFT s | decode tok/s | peak GiB (`mx.get_peak_memory`) |
|---|---|---|---|---|---|
| before | p_short | 284 (215-409) | 0.85 | 18.4 (16.6-24.3), not resolved | 83.2 |
| after | p_short | 306 (232-339) | 0.78 | 26.8 (26.0-28.1), not resolved | 83.3 |
| before | p_2k | 470 (441-477) | 4.45 | 17.7 (16.2-18.1), not resolved | 87.4 |
| after | p_2k | 566 (473-568) | 3.70 | 23.9 (23.9-25.8), not resolved | 87.8 |
| before | p_8k | 509 (454-525) | 16.21 | 14.9 (13.3-19.4), not resolved | 93.5 |
| after | p_8k | 606 (601-614) | 13.61 | 21.0 (18.9-23.3), not resolved | 91.5 |
| before | p_32k | 459 (414-489) | 71.5 | 17.8 (10.7-20.0), not resolved | 104.9 |
| after | p_32k | 484 (434-497) | 67.9 | 23.0 (21.7-24.9), not resolved | 95.3 |
| before | p_64k | not measured (projected not to fit from the 32K peak) | - | - | - |
| after | p_64k | 472 (449-495) | 139.5 | 13.6 (13.5-13.7) | 100.4 |

Ratios after/before (medians): prefill 1.08x / 1.20x / 1.19x / 1.05x, TTFT 0.93x /
0.83x / 0.84x / 0.95x, peak memory +0.1 / +0.4 / -2.1 / -9.6 GiB at short / 2K / 8K /
32K; the decode ratios are not a claim (see the A/B paragraph below). 64K context
fits on the branch (100.4 GiB peak); the wheel was **not measured** at 64K: it already
peaks at 104.9 GiB at 32K and would add 5.5 GB of unreadable keys plus four mask-sized
tau temporaries (8.6 GB each for a 2048-token chunk against 65K keys), so "does not
fit" is a projection. A4's quiet hold (load 4-7) reproduces memory (105.1 -> 95.3 GiB
at 32K) and prefill/TTFT (+17-19 % / -15-16 % at 8K and 32K; short-prompt prefill
+3 % is noise) and measured decode at +13 / +13 / -4 % within the rep spread.

Per-item attribution (p_short + p_32k, 2 runs each, same hold):

| run (commits) | p_32k prefill tok/s | p_32k TTFT s | p_32k decode tok/s | p_32k peak GiB |
|---|---|---|---|---|
| before (wheel) | 459 | 71.5 | 17.8 (10.7-20.0) | 104.9 |
| window caches only (`0abf8bd`) | 443 | 74.2 | 18.4 (15.7-21.1) | 101.1 |
| + strided sconv, tau skip (`21e690a`) | 510 | 64.6 | 18.5 (17.8-19.2) | 95.3 |
| whole branch (`355ea6f`) | 484 | 67.9 | 23.0 (21.7-24.9) | 95.3 |

Memory is the clean signal: -3.8 GiB from bounding the 35 sliding-layer caches
(4.7 GB of keys at 32K; the rest of the delta is buffer rounding), -5.8 GiB more
from not materializing the three tau passes over the [1, 32, 2048, 32K] mask of
each global layer per prefill chunk. Prefill: the kernels commit carries the gain
(the window cache alone changes nothing in the prefill math). Decode: the two
ablation rows sit at the baseline's level and the whole-branch row 30 % above it,
with load averages 15-18 vs 13; that spread is the machine, not the code.

Interleaved decode A/B (second hold, 02:29-02:50, wheel/branch/wheel/branch on
p_short + p_32k, 2 timed runs each, `bench/results/ab-*.jsonl`): consecutive
processes under nominally the same conditions swing by 2x (branch at short
context: 27.3, 25.9 then 14.4, 13.1 tok/s; wheel: 21.9, 19.6, 21.9, 22.3; at 32K
branch 25.2, 24.7, 10.4, 17.1 vs wheel 15.9, 18.9, 11.5, 9.3), and prefill in
this hold ran 25 % below the first hold's on the same code, so the machine was
the variable. Pooled medians: short context wheel 21.9 vs branch 20.2 tok/s, 32K
wheel 13.7 vs branch 20.9. **Decode is not resolved on this shared machine; no
regression is evidenced**, and the expected gain is small by construction
(R1: 84 copy kernels + ~80 tau ops per token out of ~2000, 3.5 ms/token of
cache reallocation at 32K, i.e. 5-8 % at 32K, ~2 % at short context). A quiet
machine, or a synthetic exact-shape layer benchmark (R1's `layer_bench.py`
with the two kernel changes toggled), would pin it down.

kv-bits 8 generation on the branch (`after-kv8`, p_short + p_8k, 2 runs): the
greedy outputs leave the wheel's at token 17 on both prompts (the first token
after the answer; both continuations stay coherent, margins 0.4-0.8 nats), which
a non-bit-exact item is allowed to do while its teacher-forced KL is at the floor
(table above). Decode 25.2 / 24.3 tok/s, peak 83.3 / 92.1 GiB (float branch:
83.3 / 91.5).

Loader (item 1): `after-eager-load` loaded the checkpoint with `lazy=False`
(`load_model_weights` -> `eval_parameters`) in 13.1 s; the wheel's one-shot
`mx.eval` aborts with `GPU Timeout` on this machine (M1's BASELINE.md section 5).

## Tests

- `pytest -q mlx_vlm/tests` at the final commit eb93183: **5157 passed, 16 skipped**
  (`bench/logs/pytest-full-eb93183.log`, run by `bench/rerun_full_suite.sh` once
  40+ GiB of host RAM was free). Earlier at 970c6ed: 5155 passed, 16 skipped. The
  run at 355ea6f showed 83 failures in `test_cache.py`/`test_generate.py` APC tests
  while K3's 98 GB model was resident: the APC manager budgets from free host RAM
  (6.4 GiB at the time); the same files re-run with 102 GiB free: 956 passed
  (`bench/logs/pytest-cache-generate-355ea6f.log`).
- At the final series (`ee1be25`) the loader, window-cache, Inkling fast-path,
  kernel-gate, cache and speculative test files pass at every PR-group boundary
  (A, A+B, A+B+C, all; `bench/logs/pytest-boundaries.log`); pre-commit clean.
- New tests: `test_load_eval_chunks.py` (5), `test_sliding_window_cache.py` (19),
  `test_inkling_fast_path.py` (28): window vs full cache on a 600-token chunked
  prefill plus hundreds of decodes (8- and 512-token windows), mask equality and the
  closed-form bias, speculative verify/rollback vs replay, left-padded batch vs rows,
  `PromptProcessingBatch`/`GenerationBatch` with right-padded prompts, strided sconv
  kernel vs contiguous (exact) and unfused (1 ulp), tau skip, `logits_to_keep`,
  dense-MLP scale dtype, uniform/split/TurboQuant kv policies. All pass with
  `MLX_ENABLE_TF32=0` too.
- `pre-commit` (black 26.3.1, isort 5.13.2, autoflake 2.2.1): clean on every changed file.

## How to run

```
source ~/.venvs/inkling/bin/activate            # branch (editable install of the clone)
cd ~/code/inkling-contrib/upstream/mlx-vlm && git checkout inkling-fast-path
python -m pytest -q mlx_vlm/tests/test_load_eval_chunks.py mlx_vlm/tests/test_sliding_window_cache.py mlx_vlm/tests/test_inkling_fast_path.py
pre-commit run --files $(git diff --name-only 138d761 HEAD)

# real weights, one GPU hold each (queue protocol in bench/run_all.sh):
nohup zsh bench/run_all.sh > bench/logs/run_all.out 2>&1 &      # before/after, eager load, 64K, ablations
nohup zsh bench/run_ab.sh  > bench/logs/run_ab.out  2>&1 &      # interleaved A/B + kv8
python bench/compare.py bench/results/before.jsonl bench/results/after.jsonl ...
```

`~/.venvs/inkling-base` holds the pristine 0.7.11 wheel package (plus a `.pth`
onto the shared venv for every other dependency) and is the "before" interpreter;
`~/.venvs/inkling` resolves `mlx_vlm` to the clone's working tree.

## Shared-environment caveat

`~/.venvs/inkling` has `mlx-vlm` installed **editable from the clone** (as the
brief asked), so from 22:25 on 2026-10-28 every process started from that venv
imports the clone's working tree, i.e. this branch (and, during the ablation
runs, briefly an earlier commit of it). Any "baseline" measurement of mlx-vlm
must use `~/.venvs/inkling-base/bin/python`. M1's `bench/results/mlx-kv8-rerun.*`
(started 22:36) ran against the working tree at that time, i.e. with the chunked
loader only; its kv-bits were still a no-op there.

## Review status

Not yet reviewed (REVIEW.md pending).
