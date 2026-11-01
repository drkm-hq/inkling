# Fact sheet: Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED` (Inkling on Apple Silicon)

This is not a PR description. llama.cpp prohibits AI-written PR descriptions, commit messages and reviewer
responses (`AGENTS.md` "Prohibited AI Usage", `CONTRIBUTING.md` "AI Usage Policy" item 5). The submitter writes the PR body,
the final commit messages and every reply to reviewers himself; this file lists the facts he needs, with the logs
behind each one. The commit messages on the branch are placeholders (short, `Assisted-by: Claude Fable 5.1` per
`AGENTS.md` line 96, never `Co-authored-by`) for him to rewrite with `git rebase -i`.

Independent review: `REVIEW.md` (reviewer A1, read-only, own builds, 165 extra op-level cases, reference-based
end-to-end evidence). Its verdict: the kernel is correct and fast; the first version of this package was not
submittable on process and evidence. This version follows its section 8.

## 1. What the branch is

`metal-banded-fa` on `pr-25731` (ggml-org/llama.cpp#25731 "Add TML Inkling architecture", head `3582c5ee0`),
three commits, exported as `patches/`:

| patch | commit | files | what |
|---|---|---|---|
| `patches/0001-*.patch` | `fa06bb282` | `ggml/src/ggml-metal/*` (+253/-34) | the Metal kernel for the op exactly as #25731 defines it (`rel_dist = q_idx + (n_kv - n_batch) - kv_idx`) |
| `patches/0002-*.patch` | `caad009a4` | `tests/test-backend-ops.cpp` | the extended test helper, the Inkling-geometry cases, the tightened tolerance, perf cases |
| `patches/0003-*.patch` | `7c25a0a9b` | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml-cpu/ops.cpp`, `ggml-cuda/{fattn-banded.cu,fattn-common.cuh,fattn-mma-f16.cuh}`, `ggml-metal-ops.cpp` (one line), `src/models/inkling.cpp`, `tests/test-backend-ops.cpp`, `tests/test-flash-attn-bias.cpp` | **a proposal for the op's own PR** (#25731 or the CPU-first split #29491): an explicit `q_offset` op parameter and the model passing its padded cache views |

0001 + 0002 are the Metal follow-up: they apply to `pr-25731` alone and were verified alone (section 3). 0003 is
written on top of them because it touches the same test helper and one line of the Metal dispatcher; its
`ggml.h`/`ggml.c`/CPU/CUDA/model hunks are independent of Metal. Submission order per A1 section 8.3: 0003 goes to
the op's author as a suggestion first (its CUDA part is verified on an H100, section 3b); 0001 + 0002 go as the Metal
follow-up once the op is on master, with or without 0003. the CUDA tolerance patch (`patches/0004-*`, written against the
previous `0002`) is folded into `0002`; the file stays for the record of the H100 runs. Without 0003 the kernel runs the model's exact `pos_max + 1` views (unaligned KV, half8x8 kernel
plus pad pass for prefill, covered by the tests); with it, every prefill ubatch takes the tensor-API kernel.

## 2. Facts about the change (for the PR body)

Why: Metal has no kernel for the op, so on a Mac `-fa on` runs the seven global layers' attention on the CPU scalar
path (220 / 1404 / 7123 ms per global layer per 512-token ubatch at 2K / 8K / 32K on an M5 Max,
`research/R2-llamacpp-metal.md`; the reported prefill collapse from 336 to 15.6 tok/s at 28K), and `-fa auto`/`off`
runs the dense chain with O(n_kv x n_ubatch x n_head) scratch (6.3 GiB per global-layer chain at 32K x 512).

How (0001): the bias is a second additive score term next to the KQ mask in the three existing kernel families
(half8x8 simdgroup `fa_common.metal`, Metal-4 tensor API `fa_tensor.metal`, vec `fa_vec_common.metal`): one more
buffer (the table, `src[5]`), function constants `has_rel` / `rel_type` (F32, F16, BF16 read as raw bits; `fa_rel_load`
in `kernels/common.h`), kargs `ne_rel0` (= rel_extent), `ne_rel3`, `q_off`, `nb_rel1..3`. For query row `iq` and key
`ik` the term is `rel[iq + q_off - ik, head, iq]` when `0 <= dist < rel_extent`. A uniform per-block test skips the
gather outside the band; the kv-pad path keeps the absolute key index (`ick`); the sparse vec path uses the gathered
key index; the `blk == 2` mask shortcut still adds the bias. Host: `GGML_OP_FLASH_ATTN_EXT_BANDED` dispatches into
`ggml_metal_op_flash_attn_ext`, the pipeline getters carry `rel_type` (-1 = none) in the name, `supports_op` mirrors
the plain op plus the table checks, `ggml_backend_metal_buffer_type_get_alloc_size` reserves the same scratch. The
dequant-to-F16 pre-pass, the pad/blk/reduce auxiliaries and the vec tuning table are reused unchanged, so f16, bf16,
q8_0/q4_0/q5_x KV work (Inkling itself still gates quantized KV off in `inkling.cpp` and `llama-context.cpp`).

Why 0003: the op's implicit offset (`n_kv - n_q`) forces the model to pass K/V/mask views of exactly `pos_max + 1`
rows; those are rarely 64-aligned, so prefill ubatches fall off the tensor kernel (1.5x the attention time at 8K
in `bench/perf-mtl0-banded.log`: 19.6 ms unaligned vs 12.7 ms aligned) and the model copies the mask view per layer.
`ggml_flash_attn_ext_banded(..., rel_extent, q_offset)` stores `q_offset` in `op_params[5]`; CPU reads it, the CUDA
scalar kernel takes it as an argument, the CUDA MMA path carries it in its banded ABI's unused `m0` slope slot
(exact below 2^24), Metal reads it. The model passes the cache's 256-padded `n_kv` views and the raw mask with
`q_offset = pos_max + 1 - n_tokens`. A1 confirmed the contiguity contract that makes this safe (REVIEW.md section 1).

Caveats to state: the CUDA MMA path of 0003 is tested on an H100 and the scalar path is exercised through
test-backend-ops (section 3b), but the MMA ABI smuggling (sinks pointer, negative `max_bias`, now `m0`) is the PR's
existing design that the CUDA maintainer may want replaced by real kernel arguments. The test tolerance of 0002 is
set by the CUDA MMA kernel's f16 VKQ accumulation (1e-4 up to 8192 keys, 5e-4 above, 1e-4 for bf16 KV); the Metal
kernels and the CPU reference agree 100-1000x below it.

## 3. Test facts

Op-level, all against the CPU reference, Apple M5 Max, Metal 4:

| what | result | log |
|---|---|---|
| 0001 + 0002 alone (the follow-up without `q_offset`): `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT_BANDED` (the PR's 13 cases, 10 at the Inkling geometry, 49 from A1's categories: block boundaries, batch sizes, band edges, window, no mask, heads, KV/table types, batch, strided, sparse hint, realistic magnitudes, 1M keys) | 72/72 OK | `bench/test-v2-B-banded.log` |
| 0001 + 0002 alone: plain `FLASH_ATTN_EXT` regression | 5020 OK, 359 not supported (pre-existing), 0 FAIL | `bench/test-v2-B-plain-fa.log` |
| full series (0003): the 72 above plus 13 with the `q_offset` parameter (padded KV with the band anchored at `n_kv - n_q`, incl. the model's exact p_32k and SWA shapes; offsets 0 / 4096 / 7780 / 13192 / -300 / 2048 / 1000, with and without padding) | 85/85 OK | `bench/test-v2-C-banded.log` |
| full series: plain `FLASH_ATTN_EXT` | 5020 OK, 359 not supported (pre-existing), 0 FAIL | `bench/test-v2-C-plain-fa.log` |
| full series: the PR's standalone `test-flash-attn-bias` (CPU) | 11/11 PASS | `bench/test-v2-C-flash-attn-bias.log` |
| full series: build of `test-backend-ops`, `test-flash-attn-bias`, `llama-cli` (the model-side hunk) | 0 warnings | `bench/build-v2-C.log` |
| A1's 165 adversarial cases on v1 head `90290efad` (n_q 1..1024, n_kv 1..1M incl. every block boundary, E 512/1024, windows, no mask, GQA 32/8, 16/1, 8/8, 1/1, head 64, bf16/q8_0/q4_0/f32 KV, F16/BF16 tables, batch 2-3, strided rows, sparse hint, non-tail `q_offset`, the model's exact padded shapes, realistic input magnitudes) | 165/165 OK, NMSE 1e-13..7e-8 (f16/f32/quantized KV), 2e-5 (bf16 KV), 4e-5 (1M keys) | `review/test-banded-a1cases*.log`, `review/groupQ-summary.txt` |
| A1: the PR's `test-flash-attn-bias` on MTL0 | 16/16 PASS | `review/test-flash-attn-bias-mtl0.log` |
| A1: plain `FLASH_ATTN_EXT` on its build | 5020 OK, 359 not supported (pre-existing), 0 FAIL | `review/test-plain-fa-a1build.log` |
| v1 (kernel `90290efad`): the first 26 cases | 26/26 | `bench/test-4.log` |

The kernel code of 0001 is byte-identical to v1 head `90290efad` except the one host line that reads the offset
(`q_off = ne11 - ne01` in 0001, `op_params[5]` in 0003), so A1's 165 cases apply to it. The 72/72 and 85/85 runs above
were made under the earlier 1e-6 bound of `0002`, stricter than the folded one, so they hold under it; the re-run at the
folded head is `bench/test-v3-*.log` (section 3c).

### 3b. CUDA (on Modal, one H100 80GB, nvcc 12.8, `bench/cuda/CUDA-VERIFY.md`, ~44 GPU-minutes)

Stages on a fresh clone of the PR head `3582c5ee0`, patches applied with plain `git am`:

| stage | `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT_BANDED` | `test-flash-attn-bias` cuda small / medium / hard / overflow, cpu small | plain `FLASH_ATTN_EXT` |
|---|---|---|---|
| (a) PR head | 13/13 | 10/10, 5/5, 1/1, PASS, 10/10 | 4040/4040 |
| (b) + 0001 + 0002 (previous `0002`, 1e-6 bound) | 48/67: 19 FAIL, 5 not supported | all PASS | - |
| (c) + 0003 (previous `0002`) | 55/80: 25 FAIL, 5 not supported | all PASS | 4040/4040 (untouched) |
| final: 0001 + 0002 + 0003 + the tolerance hunk now folded into 0002 | **80/80**, 5 not supported | all PASS | - |

The 5 "not supported" cases are q8_0/q4_0 K/V, which the PR's CUDA banded op rejects (unchanged). Every (b)/(c) failure
was the tolerance, not a kernel: all are f16-K/V prefill cases (n_q 512/1024, n_kv >= 1025, the MMA path) at NMSE
1.1e-6..4.2e-5 against the 1e-6 bound set from Metal's f32 accumulators; the CUDA MMA kernel accumulates VKQ in f16,
exactly what the PR's original tolerance comment said. Every decode / small-batch / f32 / bf16 / scalar-path case
passes at 1e-6. `q_offset` is correct on CUDA: every `kv_pad` and non-tail `q_offset` case passes at the f16 noise level
(a wrong offset at these shapes would be NMSE ~1e-1); the `m0`-slot plumbing of the MMA path and the scalar-kernel
argument both work, and the 0003 hunks compile without a warning. On Linux the Metal sources are not compiled, so (b)
is 0002's cases against the PR's own CUDA kernels.

CUDA perf (H100, the PR's kernels, Inkling geometry, f16 K/V; context for the Metal numbers, not a claim of this
package): banded 512 x 8192 475 us vs plain FA 294 us (1.6x), 512 x 32768 1.60 ms vs 1.15 ms (1.4x), decode 0.6-0.9x;
the unaligned 512 x 8209 costs 2.7x the aligned 512 x 8192 (1.29 ms vs 475 us), so the 256-padded views of 0003 are a
CUDA prefill speed-up as well.

### 3c. Re-run at the folded head (this branch, `7c25a0a9b`, MTL0)

Build of `test-backend-ops`, `test-flash-attn-bias`, `llama-cli`: 0 warnings (`bench/build-v3.log`). Banded cases
85/85 OK (`bench/test-v3-banded.log`), plain `FLASH_ATTN_EXT` 5020 OK / 359 not supported / 0 FAIL
(`bench/test-v3-plain-fa.log`), `test-flash-attn-bias` 11/11 PASS (`bench/test-v3-flash-attn-bias.log`). The tree is
the one proved on the H100 (`git diff 2dee8b495` is empty).

## 4. Performance facts (op level)

A1's interleaved min-of-3 (`test-backend-ops perf`, v1 head, other jobs running, load average 13-30;
`review/e2e/perf-{banded,plain}-{1,2,3}.log`), Inkling geometry, f16 KV, f32 table, causal mask, E = 1024:

| shape (n_q x n_kv) | banded min | plain `FLASH_ATTN_EXT` min | banded / plain |
|---|---:|---:|---:|
| 512 x 8192 | 6.02 ms | 7.31 ms | 0.82x |
| 512 x 32768 | 33.0 ms | 26.5 ms | 1.24x |
| 1 x 8192 | 113 us | 96 us | 1.18x |
| 1 x 32768 | 535 us | 495 us | 1.08x |

K1's own two runs (`bench/perf-mtl0-*.log`, `bench/e2e/perf-quiet-*.log`, single shot each, GPU shared): the same
shapes at 0.85x-1.57x, the two runs disagreeing by up to 1.75x on the plain op between themselves. Against the CPU
fallback (`bench/perf-cpu-banded.log`, 18 threads): 83-310x on prefill, 46-69x on decode. The plain-op minima
(7.3 ms at 512 x 8K) are still 2x above R2's idle-GPU ceiling (3.47 ms), so none of these runs had a quiet GPU; the
ratios are the usable part. The single-row-pointer form of the gather (v1 commit `90290efad`, folded into 0001) measured
6-23% faster than the per-query-pointer form at 2K-8K in one back-to-back window (`bench/e2e/perf-quiet-banded*.log`).

## 5. End-to-end facts (Inkling-Small UD-IQ3_XXS, 91.2 GiB, M5 Max 128 GB)

Which build each number comes from matters (A1 section 5): K1's llama-bench and greedy runs (`bench/e2e/`) are from
v1 commit `4ddc6ed6d` (build 11985, before the `q_offset` model change and the final kernel form); A1's runs
(`review/e2e/`) are from v1 head `90290efad` (build 11987, the same kernel code as 0001 and the same model-side views
as 0003). No end-to-end run exists yet for the restructured series; its code is identical to `90290efad`.

Reference-based evidence (A1 section 4; reference = pristine `3582c5ee0` with `-fa on`, i.e. the CPU banded op):

- **Perplexity**, p_8k (8218 tokens, `-c 4096 -b 2048`): reference 8.2437 +- 0.4155; kernel (`-fa on`) 8.3361 +- 0.4238
  (+1.1%); dense chain (`-fa off`) 8.5764 +- 0.4387 (+4.0%). `review/e2e/ppl-*.err`.
- **KL divergence vs the reference**, p_8k (`-c 2048`, ~8K scored tokens): kernel mean KLD 0.137 (median 0.0055,
  99% 2.77), same top-1 token 90.5%; dense chain 0.151 (median 0.0059, 99% 2.75), top-1 90.3%. Both GPU paths are far
  from the CPU op by the same amount (f16 weight-quantisation noise in this tool is ~0.002-0.005); the kernel is the
  closer one on every statistic. `review/e2e/kld-*.err`.
- **Per-layer dump** (`llama-eval-callback`, p_short, 226 tokens): at layer 0, with bit-identical inputs, the attention
  output of the kernel is 1.7e-4 (relative, max over the printed values) from the CPU op, the dense chain 4.2e-4, the
  two GPU paths 3.4e-4 from each other. From there all three runs drift apart at the same rate, x10 every ~8 layers
  (5e-3 at layer 8, 1.5e-2 at 17, 1e-1 at 35; logits 1.4e-2 kernel / 5.9e-2 dense), which is the MoE routing
  amplifying f16-level attention differences. Token-exact greedy agreement is therefore unattainable for any f16
  attention implementation of this model; K1's earlier "near-tie" reading of the `-fa on` vs `-fa off` divergences
  was unmeasured and is withdrawn (against the reference the divergences are 0.5-0.7-nat disagreements at the step,
  0.03-0.06 nats mean on the common prefix; `review/e2e/greedy-kernel-vs-ref.txt`). `review/e2e/evalcb-*.log`,
  `review/e2e/evalcb-growth.txt`.
- **Stability**: p_32k (32,794 tokens) and p_64k (65,561 tokens) through llama-server `-c 70000 -fa on`: correct
  answer, the 32K continuation reproduces the source text, no garbage or NaN at either depth; prefill 270 tok/s at
  32.8K and 235 tok/s at 65.5K on a loaded machine (vs 15.6 tok/s reported for the CPU fallback). Peak RSS 93.8 GiB
  (`-c 70000`), 90.7 GiB (`-c 8192`), 92 GiB llama-bench, 94.9 GiB llama-perplexity, the same for both paths.
- **Prefill vs the dense chain**: +3% to +10% at d0/d4096 in A1's interleaved llama-bench rounds (`review/e2e/abab-*.md`),
  +6% in K1's run (`bench/e2e/llama-bench-fa-*.md`). Confirmed; small because the MoE dominates a 512-token ubatch.
- **Decode vs the dense chain: UNRESOLVED.** K1's single run (v1 `4ddc6ed6d`, `-r 2`, load 15-19) showed `-fa on`
  +28-30% (19.5 vs 15.1 tok/s at d0, 16.3 vs 12.7 at d16384); A1's three interleaved pairs on the final build (load
  18-32, falling during the rounds) showed the opposite sign, `-fa 1` 10-30% slower at d4096 (12.2 vs 17.5, 15.2 vs
  21.7, 17.0 vs 22.0 tok/s). The attention op cannot produce a 5 ms per-token gap at 4K (7 global layers x ~0.1 ms), so
  whatever the sign is, it is CPU-side or scheduling and it is confounded by load. The number goes into the PR only
  after `bench/ab_quiet.sh` (three interleaved rounds of `-fa 0` / `-fa 1`, pp512 and tg128 at d0 / 4096 / 16384 /
  32768, min and median, load1 recorded) has run on a quiet machine. Until then the README says "pending".

## 6. For the PR template and the commits

- Requirements section: "AI usage disclosure: YES" with a sentence such as: the Metal kernel changes, the test cases
  and the `q_offset` plumbing were written with an AI coding agent (Claude) from a design the author specified and
  reviewed line by line; the agent also produced the measurement scripts; all numbers were re-measured by the author
  or by an independent agent-driven review; the PR text and commit messages are the author's own.
- Commit trailers: `Assisted-by: Claude Fable 5.1` on each commit the agent produced (`AGENTS.md:96`), never
  `Co-authored-by`.
- Before submitting (`CONTRIBUTING.md`): a manual review of the +253/-34 kernel diff and the tests (about an hour per
  200-400 lines), `ab_quiet.sh` on a quiet machine, and the discussion opened on #25731 / #29491 before a PR (one open
  PR per new contributor; features start with an issue). The CUDA side of 0003 is covered by section 3b.

## 7. Open questions a maintainer will raise

1. A separate op versus a sixth source of `ggml_flash_attn_ext`: the Metal host treats both ops in one dispatcher
   either way; the decision belongs to the op's PR.
2. `q_offset` as a required constructor argument (no hidden `n_kv - n_q` contract) versus a setter with a sentinel.
3. The CUDA MMA ABI (table through the sinks pointer, E through `max_bias`, now `q_offset` through `m0`); verified
   on an H100 but a design the CUDA maintainer may want replaced by real kernel arguments.
4. The vec-kernel tuning table was built for the plain op; the banded decode cases take the same configs.
5. The decode A/B (section 5).
