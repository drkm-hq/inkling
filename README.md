# Inkling contributions from Darkmatter Labs

Patches, tests, benchmarks and review notes for running Thinking Machines' Inkling-Small
(276B total, 12B active, hybrid sliding-window and global attention with a relative bias,
short convolutions, 256-expert MoE) faster on Apple silicon through llama.cpp, mlx-lm and
mlx-vlm. The work was done by Darkmatter Labs, Inc. on a MacBook Pro M5 Max with 128 GB of
unified memory (macOS 26.5.1, MLX 0.32.4, mlx-vlm 0.7.11, mlx-lm 0.32.2) that was shared with
other jobs while it measured, plus one CUDA verification pass on rented H100 and H200 GPUs.

## Status

Nothing in this repository has been submitted upstream yet. Every change set is complete and
has an adversarial review in `package/*/REVIEW.md`, written by a second AI agent that received
the branch and the numeric oracle rather than the author's conclusions. Before a pull request
can be opened, each `PR.md` fact sheet lists what is still owed: a human line-by-line review of
the diff, a quiet-machine decode A/B for change sets 1a and 3, and human-written PR text and
commit messages where the upstream project requires them. The code, tests and measurement
scripts were written with an AI coding agent (Claude) under human direction; the upstream
projects' AI-use disclosure rules are quoted in each fact sheet. Change set 1b ports a fix
first published by F3zz1k in a llama.cpp fork (commit 371c74a) and credits that author.

## The four change sets

| # | Change | Target | Where |
|---|---|---|---|
| 1a | Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED` (Inkling's relative-bias attention), plus a separate `q_offset` op-parameter proposal for the op's own PR | ggml-org/llama.cpp, on top of draft PR #25731 | `package/k1-metal-banded-fa/` |
| 1b | llama-server prompt-cache reuse for hybrid/recurrent memories with a sliding window (port of the F3zz1k fork fix to the current gate) | ggml-org/llama.cpp master | `package/k3-llamacpp-prompt-cache/` |
| 2 | First-class Inkling text model for mlx-lm (`inkling_mm_model.py`, caches, tests, kv-bits warning) | ml-explore/mlx-lm | `package/k2-mlx-lm-inkling/` |
| 3 | mlx-vlm Inkling fast path in four PR groups: chunked loader, bounded window caches, fewer decode kernels, kv-bits for the global layers | Blaizzy/mlx-vlm | `package/k4-mlx-vlm-fast-path/` |

Measured effects, each taken from the file named (raw logs sit next to every number):

- 1a: with flash attention on, the seven global layers no longer fall back to the CPU; prefill
  stays flat with context, 32K and 64K prompts answer correctly, and the multi-GiB dense scratch
  that made a 70K context unusable is gone (`package/README.md`). Op-level tests: 85/85 banded
  cases on Metal (`package/k1-metal-banded-fa/PR.md`, section 3) and 80/80 on CUDA
  (`package/k1-metal-banded-fa/bench/cuda/CUDA-VERIFY.md`). Perplexity against the CPU reference:
  kernel +1.1%, dense chain +4.0% (`REVIEW.md`, section 4b). The decode delta is unresolved on a
  loaded machine; `bench/ab_quiet.sh` is the A/B to run first.
- 1b: a 200-token extension of an 8K prompt evaluates 204 tokens (1.43 s) instead of 716 (3.60 s)
  at `-ub 512`, and instead of 8418 at `-ub 256`; chat turns evaluate 16 to 23 tokens instead of
  550 to 592 (`package/k3-llamacpp-prompt-cache/bench/RESULTS.md`). The same prompt_n and cache_n
  counts were reproduced on an H200 (`package/k1-metal-banded-fa/bench/cuda/CUDA-VERIFY.md`, step 3).
- 2: loads the mlx-community checkpoints in mlx-lm with teacher-forced KL at mlx-vlm's own noise
  floor (91 to 97% top-1 agreement) and 2.5 to 7.6 GiB lower peak memory at 2K to 8K; not faster
  than mlx-vlm on decode (minus 3 to 23% across holds). Its 8-bit attention recipe gives +23% decode
  at KL 0.028 (`package/k2-mlx-lm-inkling/README.md`, "Measured numbers").
- 3: identical greedy tokens to the stock wheel on four prompts
  (`package/k4-mlx-vlm-fast-path/bench/results/COMPARE.md`); peak memory at 32K 104.9 to 95.3 GiB
  and a 64K prompt fits at 100.4 GiB (`REVIEW.md`, section 5); prefill +5 to +20% and TTFT minus 5
  to 17% as median ratios (`package/k4-mlx-vlm-fast-path/README.md`, "Ratios after/before");
  decode not resolved.

Two later patch sets sit beside the package. `review-fixes-2026-10-31/` holds four correctness
patches from a second review (mlx-lm training-mode dispatch, loader mapping and retained-prefix
distances; mlx-vlm window-cache disk restore, batched state and uniform KV quantization after cache
restore; see its `APPLY.txt`). `performance-fixes-2026-10-31/` holds five further Metal and MLX
kernel patches with paired whole-model measurements: mlx-lm decode at 8K +5.78% median over five
pairs, mlx-vlm +1.54% and +1.58% (preliminary), llama.cpp inconclusive
(`performance-fixes-2026-10-31/RESULTS.txt`).

## How to read it

1. `package/REVIEW-GUIDE.md`: the suggested reading order (1b, 1a, 3, 2) and what to verify.
2. `package/README.md`: the four change sets, the facts that shaped them, the upstream rules.
3. `package/<set>/README.md`, `PR.md` (fact sheet), `REVIEW.md` (adversarial review), `patches/`,
   `bench/` and `review/` (raw evidence).
4. `research/RANKING.md` and `research/00-method.md`: how the candidates were chosen and the
   acceptance criterion (teacher-forced KL against a measured noise floor, since this MoE model
   amplifies f16-level differences into greedy-token divergence). `research/R1..R5` are the
   engine studies and the reference-semantics audit; `research/oracle/` is the float64 numpy
   oracle validated against transformers.
5. `branches/BRANCHES.md`: every branch as a git bundle, a format-patch series and a log.
6. `bench/PROTOCOL.md` and `bench/results/BASELINE.md`: the measurement protocol and baselines.

## Applying the patches

Clone the upstream repository, check out the base commit named in `branches/BRANCHES.md`, then
either `git fetch <bundle> <branch>:<branch>` or `git am patches/*.patch`. The bases are
llama.cpp PR #25731 head `3582c5ee0` (1a, 1b; 1b also applies to master `2e0d86beb`), mlx-lm
`1f7057c` and mlx-vlm `138d761`. The PR #25731 commits themselves are not redistributed here;
fetch them from the upstream pull request. The 2026-10-31 fixes are plain unified diffs for
`git apply`; their `APPLY.txt` files give the order and the targeted test commands.

## Running the tests and benchmarks

- 1a: `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT_BANDED` and `test-flash-attn-bias`
  (`package/k1-metal-banded-fa/README.md`, "Verify"); the CUDA harness in `bench/modal/` runs
  under your own Modal account.
- 1b: `pytest -v unit/test_completion.py -k test_completion_cache_hybrid_memory` in
  `tools/server/tests`; benchmark `package/k3-llamacpp-prompt-cache/bench/bench_prompt_cache.py`.
- 2: `python -m unittest tests.test_inkling tests.test_models.TestModels.test_inkling_mm_model`;
  scripts in `package/k2-mlx-lm-inkling/bench/scripts/`.
- 3: `pytest -q mlx_vlm/tests`; `bench/run_all.sh` and `bench/run_ab.sh` under
  `package/k4-mlx-vlm-fast-path/`.

Models are not included: mlx-community/Inkling-Small-mlx-2bit and unsloth/Inkling-Small-GGUF
UD-IQ3_XXS from Hugging Face. Benchmark prompts are built from the public-domain text of
Moby-Dick (`bench/prompts/`). Absolute tok/s swung 1.6 to 2x between holds on the shared machine
(`package/README.md`), so only same-hold, back-to-back pairs are quoted as comparisons.

## License

The patches are offered under the license of the project they target: MIT for llama.cpp and ggml
and MIT for mlx-lm; mlx-vlm is expected to be MIT as well, and each upstream LICENSE file must be
checked at submission time. Our own scripts, reports and reviews in this repository are released
under the MIT License. Copyright 2026 Darkmatter Labs, Inc. (Darkmatter is the company's product,
drkm its command and package, drkm.ai its site.)