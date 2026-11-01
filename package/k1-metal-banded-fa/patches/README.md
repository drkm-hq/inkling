# Patch series (`git format-patch pr-25731..metal-banded-fa`)

Base: ggml-org/llama.cpp#25731 at `3582c5ee0` (branch `pr-25731` in `upstream/llama.cpp-inkling`).
Apply with `git am` in order. The commit messages are placeholders (the submitter rewrites them; `Assisted-by:` trailers
per `AGENTS.md`).

| # | file | target | depends on |
|---|---|---|---|
| 0001 | `0001-metal-add-GGML_OP_FLASH_ATTN_EXT_BANDED.patch` | the Metal follow-up PR, once the op is on master | nothing (applies to `pr-25731`) |
| 0002 | `0002-tests-banded-flash-attention-cases.patch` | the same Metal follow-up PR | 0001 |
| 0003 | `0003-ggml-q_offset-for-the-banded-flash-attention-proposa.patch` | **the op's own PR** (#25731 or the CPU-first split #29491), as a suggestion to its author | 0002 (shares the test helper and one line of the Metal dispatcher) |
| 0004 | `0004-tests-banded-flash-attention-tolerance-for-the-CUDA-.patch` | **folded into 0002** (M2's CUDA tolerance hunk, measured on an H100 against the previous 0002); kept for the record of `../bench/cuda/`, do not apply | - |

- 0001 + 0002 are self-contained: the kernel computes the op exactly as the PR defines it
  (`rel_dist = q_idx + (n_kv - n_batch) - kv_idx`) and the tests pass on `pr-25731` + 0001 + 0002 alone
  (`../bench/test-v2-B-banded.log`). If 0003 is rejected the kernel still runs the model's exact `pos_max + 1`
  views: unaligned KV goes through the half8x8 kernel plus the pad pass (covered by the tests) at ~1.5x the prefill
  attention time of the aligned case.
- 0003 changes the signature of `ggml_flash_attn_ext_banded` (a function that only exists in #25731), the CPU op,
  both CUDA kernels, the model's call site and `tests/test-flash-attn-bias.cpp`; its only Metal hunk replaces
  `q_off = ne11 - ne01` by the `op_params[5]` read. Its CUDA hunks are verified on an H100 (`../bench/cuda/CUDA-VERIFY.md`:
  80/80 banded cases with the series, `q_offset` correct on the scalar and the MMA path). If the op's author takes 0003
  first, 0001 is rebased by changing that one line.
