# Review guide

Suggested order: 1b (smallest, cleanest), 1a, 3, 2.

## 1b llama.cpp prompt cache (k3-llamacpp-prompt-cache/)
Read: README.md (mechanism and numbers), patches/0001-*.patch (11 lines of server code + a test), REVIEW.md (A3), bench/RESULTS.md. Verify: `git am` onto llama.cpp master, build llama-server, run the new server test. You write: the commit message and PR body (PR.md is the fact sheet; it carries the F3zz1k credit and the `Assisted-by:` trailer rule).

## 1a llama.cpp Metal kernel (k1-metal-banded-fa/)
Read: patches/README.md (which patch is which), PR.md (fact sheet with Metal and CUDA evidence), REVIEW.md (A1), bench/cuda/CUDA-VERIFY.md (M2). Verify on this Mac: `git am` 0001+0002 onto pr-25731, build, `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT_BANDED`. Run bench/ab_quiet.sh on an idle machine for the decode number. Decide: submit 0001+0002 as the Metal follow-up once the op is on master; send 0003 (q_offset) to the op's author as a proposal. You write: commit messages and the PR body.

## 3 mlx-vlm fast path (k4-mlx-vlm-fast-path/)
Read: README.md (index), patches/<group>/PR.md (four fact sheets), REVIEW.md (A4), bench/results/COMPARE.md. Verify: `git am` group A, then B, B+C, B+C+D on mlx-vlm main; `pytest -q mlx_vlm/tests` with RAM free. Four PRs, in the order A, B, C, D.

## 2 mlx-lm Inkling model (k2-mlx-lm-inkling/)
Read: README.md (design, deviations from the research plan, honest performance), PR.md (fact sheet + transcript), REVIEW.md (A2), tests/. Verify: `git am` k2-mlx-lm-inkling.patch onto mlx-lm 1f7057c, `uvx pre-commit run --all-files`, `pytest tests/test_inkling.py`, then `mlx_lm.generate --model <mlx-community/Inkling-Small-mlx-2bit snapshot>`. Decide whether to submit now (correct, memory-lighter, enables the 8-bit-attention recipe, but not faster than mlx-vlm on decode) or after the loop-overlap gap is closed; README lists the candidates.

## Where everything lives
- Branches: branches/BRANCHES.md lists every branch with its base commit, as a git bundle, a format-patch series and a log.
- Research: research/R1..R5, RANKING.md, 00-method.md, oracle/.
- Baselines: bench/results/BASELINE.md; prompts in bench/prompts/.
- Models (not included): mlx-community/Inkling-Small-mlx-2bit, unsloth/Inkling-Small-GGUF UD-IQ3_XXS and
  thinkingmachines/Inkling-Small (mtp.safetensors) from Hugging Face.
