# Inkling contribution package (final, 2026-10-29)

Three opportunities, one per engine Mac users actually run, delivered as four pull-request-ready
change sets. Everything is local: nothing has been pushed, opened or uploaded. Start with
REVIEW-GUIDE.md in this directory.

| # | Contribution | Target repo | Local branch (final commit) | Patches | Independent review |
|---|---|---|---|---|---|
| 1a | Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED` (Inkling's relative-bias attention), plus a separately packaged `q_offset` op-parameter proposal for the op's own PR | ggml-org/llama.cpp (follow-up to PR #25731 / the #29491 split) | upstream/llama.cpp-inkling `metal-banded-fa` (fa06bb282, caad009a4, 7c25a0a9b) | k1-metal-banded-fa/patches/0001-0003 | A1: kernel CONFIRMED; package reworked; CUDA verified on Modal (H100): q_offset correct, tolerance folded; 85/85 Metal, 80/80 CUDA |
| 1b | llama-server prompt-cache reuse for hybrid/recurrent memories (Inkling, LFM2-style) | ggml-org/llama.cpp (applies to master) | upstream/llama.cpp-k3 `server-hybrid-prompt-cache` (9b8b6bd93) | k3-llamacpp-prompt-cache/patches/0001 | A3: CONFIRMED, no code defect; 22 adversarial server cases; CUDA (H200) identical token counts |
| 2 | First-class Inkling model for mlx-lm (`inkling_mm_model.py`, caches, tests, kv-bits warning, ACKNOWLEDGMENTS) | ml-explore/mlx-lm | upstream/mlx-lm `inkling` (dc0a8b1; clean series `inkling-pr` f98688c) | k2-mlx-lm-inkling/k2-mlx-lm-inkling.patch | A2: no BROKEN claim; 261 oracle checks at <= 2e-6; teacher-forced KL at mlx-vlm's own noise floor |
| 3 | mlx-vlm Inkling fast path and robustness, as four PR groups: A loader, B window caches, C decode kernels, D kv-bits | Blaizzy/mlx-vlm | upstream/mlx-vlm `inkling-fast-path` (ee1be25, 12 commits) | k4-mlx-vlm-fast-path/patches/{A-loader,B-window-caches,C-decode-kernels,D-kv-bits} | A4: no BROKEN item; bit-exact (KL 0 on four prompts); memory and prefill/TTFT CONFIRMED |

## What each one does for a Mac Studio user (measured on an M5 Max 128 GB; details and raw logs in each package)

- **1a** With flash attention on, the seven global layers no longer run on the CPU: prefill stays flat with context (pp512 320/267/340 tok/s at 4K/16K/32K vs 72/34 at 0/4K on today's path), 32K and 64K prompts answer correctly, and the multi-GiB dense scratch is gone, which is what made a 70K context unusable (the dense path wired 112 GB and decoded at 3-4 tok/s). Perplexity vs the CPU reference: kernel +1.1%, dense chain +4.0%. Decode: prefill gain confirmed, decode delta unresolved on this loaded machine (bench/ab_quiet.sh is the quiet-machine A/B to run before submission).
- **1b** A 200-token extension of an 8K prompt evaluates 200-204 tokens (1.4 s) instead of 716-8418 (3.6-42 s); chat turns evaluate 16-23 tokens (0.8 s) instead of 550-2217 (3-9 s); an edit near the end restores the nearest checkpoint (253 tokens, 1.9 s vs 30 s). Same counts on CUDA.
- **2** Loads the mlx-community checkpoints (and the HF layout) in mlx-lm, so mlx_lm.generate/server and everything built on mlx-lm work; numerically equivalent to mlx-vlm (teacher-forced KL 1.0-1.5x the noise floor, 91-97% top-1); peak memory 2.5-7.6 GiB lower (15 GiB at 32K). Not faster than mlx-vlm on this machine (decode -3 to -23% across holds; the gap is loop overlap, documented). The 8-bit-attention recipe its quant_predicate enables: +23% decode, +22% prefill, -3 GiB, KL 0.028.
- **3** Bit-identical outputs to the stock wheel; peak memory at 32K 104.9 -> 95.3 GiB and a 64K prompt now fits (the wheel already peaks at 105 GiB at 32K); prefill +8-20%, TTFT -8-17%; the loader no longer trips the GPU watchdog on 80+ GB models; kv-bits now quantizes the global layers instead of being silently ignored. Decode delta unresolved.

## Facts that shaped the campaign
- On this MoE model, per-layer differences of ~1e-4 are amplified ~10x every 8 layers through expert routing; two faithful f16 paths diverge in greedy tokens with 0.5-0.7-nat disagreements. Acceptance therefore uses teacher-forced KL/perplexity against a measured noise floor (research/00-method.md section 3), token equality only for bit-exact changes.
- MLX 0.32.4 on the M5 runs fp32 GEMMs at TF32 precision by default; every fp32 exactness check sets MLX_ENABLE_TF32=0.
- Absolute tok/s swung 1.6-2x between holds on this shared machine (other sessions' load); only same-hold back-to-back pairs are quoted as comparisons. Baselines: bench/results/BASELINE.md (loaded-machine curves for both engines, plus a quiet 16K-context llama.cpp server set); the stock mlx-vlm quiet-machine numbers for the fixed prompts come from K2's and A4's same-hold pairs (45.6/40.6/38.2 tok/s decode and 148/700/758 tok/s prefill at short/2K/8K, load ~7); a stock quiet 32K MLX row is still unmeasured.
- GGUF downloads of Inkling outnumber MLX downloads ~500:1, so 1a/1b reach the most users; mlx-vlm is vendored by oMLX and pinned by LM Studio's MLX engine, so 3 propagates beyond its repo.

## Project rules that affect submission
- llama.cpp: AI-written PR descriptions and commit messages are prohibited (AGENTS.md / CONTRIBUTING.md); the assistant trailer is `Assisted-by:`; the submitter writes the PR bodies and final commit messages after his own review. 1a can only land after the op exists on master (PR #25731 or the #29491 CPU-first split); the q_offset patch is a proposal for that PR's author, not a Metal follow-up.
- mlx-lm: new-model PR template needs an mlx_lm.generate transcript (included); AI-written PR text prohibited, AI use disclosed; the submitter writes the PR text.
- mlx-vlm: AI assistance disclosed in each fact sheet; four PRs to match the maintainer's review size.

## Before submission
1. The submitter reviews each diff (REVIEW-GUIDE.md lists the reading order).
2. Quiet-machine A/B for the decode numbers of 1a and 3 (scripts ship in the packages).
3. The submitter writes the PR bodies and commit messages where the project requires it.
4. Push only on the submitter's explicit instruction, one item at a time.
