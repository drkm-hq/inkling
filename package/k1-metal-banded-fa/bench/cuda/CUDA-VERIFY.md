# CUDA verification of the K1 patch series (Modal, NVIDIA H100)

2026-10-29. The parts of the package this Mac cannot test: the CUDA side of 0003 (`q_offset` in the scalar
kernel and in the MMA ABI) and 0002's test cases against the PR's CUDA kernels. The three op-level stages ran in one
Modal container (`bench/modal/cuda_verify.py`, app `inkling-cuda-verify`, id `<modal-app>`); the
final run (`final/`) and the K3 step (`k3/`, `bench/modal/cuda_k3_verify.py`) in their own, see the end. The logs in
this directory are the functions' own captures (`events.log` is the timeline, `meta.json` the parsed counts).

## Setup

| item | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0, 81071 MiB (one GPU) |
| host driver | NVIDIA-SMI 610.57.04 (CUDA UMD 13.3) |
| image | `nvidia/cuda:12.8.1-devel-ubuntu22.04` + git, cmake 3.22.1, ninja 1.10.1, gcc 11.4.0; nvcc 12.8 (V12.8.93) |
| container | 32 vCPU, 64 GiB |
| source | `https://github.com/ggml-org/llama.cpp`, `git fetch origin pull/25731/head:pr-25731`, head verified `3582c5ee05115892e4e96520be0026c3ceb9f829` ("Fix merge conflicts") |
| patches | `patches/0001` sha256 `1f75ed8d…7cc53`, `0002` `5e0126e6…bf934`, `0003` `ab776606…c2075`; applied with plain `git am` (no `--3way`), all three applied cleanly |
| configure | `cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=90` |
| build | `cmake --build build --target test-backend-ops test-flash-attn-bias llama-cli -j 32` (llama-cli = compile check of the model-side change in 0003) |

On Linux the Metal sources of 0001 are not compiled (no `GGML_METAL`), so stage (b) checks 0002's test cases against
the PR's own CUDA kernels and stage (c) checks the CUDA hunks of 0003 (scalar kernel argument, MMA `m0` slot).

## Commands per stage

```
# (a) PR head unpatched, (b) after `git am 0001 0002`, (c) after `git am 0003`; rebuild between stages
./build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT_BANDED
./build/bin/test-flash-attn-bias cuda small | cuda medium | cuda hard | cuda overflow | cpu small
./build/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT_BANDED        # (b), (c)
./build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT               # (a) baseline, (c): plain FA untouched
```

`-o FLASH_ATTN_EXT` is an exact op-name match in this `test-backend-ops` (plain names are not regexes), so the plain
run contains no banded case. The CUDA banded op takes the MMA path (`fattn-mma-f16.cuh`, `q_offset` through `m0`)
for F32 contiguous tables with F16/BF16/quantized K/V at head 64/128 and `rel->ne[3] == q->ne[3]`; every other
case (F16/BF16 table, F32 K/V, a table shared across the batch) goes through the scalar kernel
(`fattn-banded.cu`, `q_offset` as a kernel argument). Both paths are in the case list.

## Results

| stage | tree | build (32 vCPU) | `test-backend-ops -o FLASH_ATTN_EXT_BANDED` (CUDA0 vs CPU) | `test-flash-attn-bias` cuda small / medium / hard / overflow, cpu small | plain `FLASH_ATTN_EXT` | perf |
|---|---|---|---|---|---|---|
| (a) | PR head `3582c5ee` | full, 102 s, 0 warnings | **13/13** | 10/10, 5/5, 1/1, PASS, 10/10 | **4040/4040** (1339 not supported) | - |
| (b) | + 0001 + 0002 | 24 s (tests only), 0 warnings | **48/67** passed: 19 FAIL, 5 not supported | 10/10, 5/5, 1/1, PASS, 10/10 | - | `perf-b-banded.log` |
| (c) | + 0003 (`q_offset`) | full (ggml.h), 100 s, 0 warnings | **55/80** passed: 25 FAIL, 5 not supported | 10/10, 5/5, 1/1, PASS, 10/10 | **4040/4040** (1339 not supported; untouched) | `perf-c-banded.log` |
| final | 0001 + 0002 + 0003 + 0004 (tolerance) | full, 123 s, 0 warnings | **80/80** passed, 5 not supported | 10/10, 5/5, 1/1, PASS, 10/10 | - | `final/perf-final-*.log` |

"not supported" = the 5 cases with q8_0 / q4_0 K/V: the PR's CUDA banded op accepts F32/F16/BF16 K/V only
(`fattn_banded_type_supported`), unchanged by the patches. Every `test-flash-attn-bias` suite passes at every stage
(`bias-<stage>-*.log`; the overflow kernel test runs at a 2 GiB table stride on the GPU).

### The failures: 0002's tolerance, not the kernels

All 19 (b) and 25 (c) failures are the same thing. They are the f16-K/V prefill cases (`n_q` 512 or 1024 with
`n_kv >= 1025`, F32 table, the MMA path) failing 0002's bound of NMSE 1e-6 with errors between 1.1e-6 and 4.2e-5;
the complete list with each `ERR` is in `test-b-banded.log` / `test-c-banded.log`. 0002 lowered the PR's bound
(`n_kv > 8192 ? 1e-3 : 5e-4`) to 1e-6 from the Metal kernels, which keep the VKQ accumulator in f32 (measured ~1e-8);
the CUDA MMA kernel accumulates VKQ in f16 (the PR's original comment, "fp16 VKQ accumulation error grows with the KV
length", described exactly this), so its error grows with the number of visible keys:

| shape (f16 K/V, F32 table, qk 0.25) | ERR (b) | ERR (c) |
|---|---:|---:|
| 8 heads, 512 x 8192 | 2.9e-6 | 2.5e-6 |
| 8 heads, 512 x 16384 / 16403 / 16896 / 17024 | 6.0e-6 / 5.1e-6 / 5.7e-6 / 5.3e-6 | 5.5e-6 / 4.7e-6 / 5.2e-6 / 5.4e-6 |
| 8 heads, 512 x 32768 | 9.8e-6 | 9.5e-6 |
| 32/8 heads, 512 x 1025 / 1536 / 2048 (no mask) / 4097 / 8193 / 8209 | 1.1e-6 / 1.7e-6 / 2.6e-6 / 4.5e-6 / 9.4e-6 / 9.4e-6 | 1.1e-6 / 1.7e-6 / 2.7e-6 / 4.6e-6 / 9.4e-6 / 1.0e-5 |
| 32/8 heads, 512 x 8192 E = 512; 1024 x 4100; 512 x 8209 no mask | 9.9e-6; 3.6e-6; 9.7e-6 | 1.0e-5; 3.8e-6; 1.0e-5 |
| 16/1 heads 512 x 2048; d = 64 512 x 4096; batch 2 512 x 2049 | 1.1e-6; 4.7e-6; 1.9e-6 | 1.4e-6; 4.5e-6; 1.9e-6 |
| 32/8 heads, 512 x 8192, qk 1.7 / rel 4.0 (peaked softmax) | 1.8e-6 | 1.8e-6 |
| (c) only: 512 x 8209 `kv_pad` 239; 512 x 32794 `kv_pad` 230 | - | 1.0e-5; **4.2e-5** |
| (c) only: 512 x 8192 at `q_offset` 4096 / 7780 / 13192; 512 x 4096 no mask at -300 | - | 5.4e-6 / 9.6e-6 / 1.1e-5; 5.8e-6 |

Everything with a decode or small batch (`n_q` 1, 2, 7, 17, 20, 64), everything with `n_kv <= 1024`, the f32 and bf16
K/V cases, the F16/BF16 tables (scalar path), the strided and batch-2-table cases and the 1M-key case pass at 1e-6
(bf16 and 1M at 1e-4). The same shapes pass in (a) under the PR's bound. Run-to-run variation of the same shape between
(b) and (c) is about 10 %; the 512 x 32794 padded case is the largest at 4.2e-5 (4x its unpadded neighbour, still
f16-accumulation territory: a wrong block or a wrong offset would be at 1e-1 or more, see below).

### 0003: `q_offset` on CUDA is correct

The CUDA hunks of 0003 compile without a warning (`build-c.log`) and no case fails for a reason other than the
tolerance above. The cases that only pass if the offset is carried correctly all pass at 1e-6 or fail only by the f16
noise: `kv_pad` 239 at 1 x 8209 (OK), `kv_pad` 247 at 1 x 777 window (OK), `kv_pad` 60 at 7 x 4100 (OK), `kv_pad` 31
at 17 x 33 strided F16 table (OK, scalar kernel), `q_offset` 2048 at 1 x 4096 (OK), 1000 at 20 x 4100 (OK), 0 at
512 x 8192 (OK), and the `n_q` 512 cases at offsets 4096 / 7780 / 13192 / -300 / 7697 / 32282 at 5e-6 .. 4e-5. The
MMA path decodes the offset from the `m0` slope slot (`fattn-mma-f16.cuh`, `int(m0)`) and the scalar path takes it as
a kernel argument; a wrong offset at these shapes moves the band by thousands of keys and would show as NMSE ~1e-1.
`test-flash-attn-bias` (tail-anchored offsets only) is 16/16 + overflow on the GPU and 10/10 on the CPU after 0003.

### 0004: the tolerance patch (new, for the submitter to fold into 0002)

`patches/0004-tests-banded-flash-attention-tolerance-for-the-CUDA-.patch` (branch `m2-cuda-tolerance` in the
worktree `upstream/llama.cpp-m2`, one commit on `metal-banded-fa`) changes only `max_nmse_err()` of
`test_flash_attn_ext_banded`: bf16 K/V 1e-4 (unchanged), otherwise 1e-4 up to 8192 keys and 5e-4 above (8x .. 12x
above the largest measured H100 error, 5x tighter than the PR's bound below 8K, equal above). It belongs in 0002 (same
hunk, the Metal-only rationale replaced); it is a separate file here so the measured series is reproducible. Proven
by the final run (below).

Final run (`final/`, app `<modal-app>`, a fresh container: clone, `git am` 0001 0002 0003 0004, full
build 123 s, 0 warnings): `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT_BANDED` **80/80 passed** (the same 5
quantized-KV cases not supported), every `test-flash-attn-bias` suite PASS, `am-final.log` shows the four patches
applying cleanly in order on `3582c5ee`.

### Perf (`test-backend-ops perf -b CUDA0`, Inkling-Small geometry 32/8 heads x 128, E 1024, f16 K/V)

| shape (n_q x n_kv) | banded (b) | banded (c) | banded (final) | plain FA, same geometry (final) | banded / plain |
|---|---:|---:|---:|---:|---:|
| 512 x 2048 | 188.9 us (91.0 TFLOPS) | 185.8 us (92.5 TFLOPS) | 185.9 us | 79.3 us | 2.34x |
| 512 x 8192 | 480.0 us (143.2 TFLOPS) | 472.1 us (145.6 TFLOPS) | 474.8 us | 294.3 us | 1.61x |
| 512 x 8209 (unaligned KV) | 1316.8 us (52.3 TFLOPS) | 1317.2 us (52.3 TFLOPS) | 1290.1 us | 1129.2 us | 1.14x |
| 512 x 32768 | 1622.3 us (169.4 TFLOPS) | 1597.9 us (172.0 TFLOPS) | 1595.2 us | 1148.0 us | 1.39x |
| 1 x 2048 | 8.7 us | 8.5 us | 8.7 us | 9.5 us | 0.91x |
| 1 x 8192 | 13.6 us | 13.4 us | 13.7 us | 21.5 us | 0.64x |
| 1 x 32768 | 54.4 us | 54.4 us | 54.5 us | 68.3 us | 0.80x |

(`perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\]'` for the plain column; the PR's CUDA kernels,
not K1's, so this is context for the Metal numbers in `PR.md`, not a claim of the package.) 0003 changes nothing in
speed (same kernels, one more scalar argument). The unaligned 512 x 8209 shape costs 2.7x the aligned 512 x 8192 for
the banded op and 3.8x for the plain op (the non-multiple-of-`FATTN_KQ_STRIDE` path of the MMA kernel): llama.cpp's
plain-attention views are 256-padded so the plain op never hits it, the PR's banded model code passes the exact
`n_kv = pos_max + 1` and hits it on every ubatch, the 256-padded views of 0003 restore the aligned path, so the
`q_offset` proposal is also a CUDA prefill speed-up.

## Step 3: K3's prompt-cache fix on CUDA (H200, `cuda_k3_verify.py`)

Backend-independence check of `package/k3-llamacpp-prompt-cache/patches/0001-server-do-not-apply-the-SWA-back-off-to-a-recurrent-.patch`
on the same PR head (`3582c5ee`, no K1 patches), `llama-server` built with `GGML_CUDA` (sm_90), one H200 (141 GB):
`unsloth/Inkling-Small-GGUF` UD-IQ3_XXS (3 shards, 12.99 MB + 49.41 GB + 48.51 GB, downloaded once into the Modal
Volume `inkling-small-gguf` by the CPU-only `download` function, 446 s), copied to the container disk (89 s), served
with `-fa 1 -ngl 99 -np 1 --no-kv-unified -c 16384 -ub 512 --temp 0` (`-lv 4`): 92.7 GiB CUDA0 model buffer, load 21-30 s,
`n_swa = 512`, "the context does not support partial sequence removal" (seq_rm type FULL) at start-up, exactly as on
the Mac. K3's case (a)/(a2) with the same prompts (`bench/prompts/p_8k.txt` = 8218 tokens, the 200-token continuation
from `src/moby-dick.txt`), `/completion` with `n_predict 64`, read `timings.prompt_n` / `cache_n` / `prompt_ms`
(`k3/before.jsonl`, `k3/after.jsonl`; the run was done twice, app `<modal-app>` streamed the numbers only,
`<modal-app>` wrote the files; the numbers are identical):

| request | before (PR head) prompt_n / cache_n / prompt_ms | **after** (+ K3 patch) prompt_n / cache_n / prompt_ms | Mac, Metal `-fa 0` (K3's `bench/RESULTS.md`) before -> after |
|---|---|---|---|
| p_8k, no cache (reference) | 8218 / 0 / 7.24 s | 8218 / 0 / 7.18 s | 8218 / 29.9 s -> 34.5 s |
| p_8k, cache on, after the uncached requests | 8218 / 0 / 7.02 s (forced full re-processing) | 316 / 7902 / 0.41 s | 8218 -> 316 / 7902 |
| (a) p_8k + 200 tokens, cache on | 716 / 7702 / 0.77 s (checkpoint 516 back) | **204 / 8214 / 0.35 s** | 716 / 7702 -> 204 / 8214 |
| p_8k again, cache on (prime for a2) | 516 / 7702 / 0.51 s | 4 / 8214 / 0.07 s | - |
| (a2) p_8k + its generated tokens + 200, cache on (strict extension) | 779 / 7702 / 0.81 s | **200 / 8281 / 0.33 s** (live state, no restore) | 779 / 7702 -> 200 / 8281 |
| (a2) same, no cache | 8481 / 0 / 7.22 s | 8481 / 0 / 7.24 s | 8481 / 32.8 s |

`k3/k3-server-*-reuse-lines.txt`: before, the identical p_8k hits "forcing full prompt re-processing due to lack of
cache data" and the +200 / strict-extension requests "restored context checkpoint (... n_tokens = 7702)"; after, the
identical p_8k restores the 7902 checkpoint, the +200 request restores the 8214 checkpoint (4 tokens before the end of
the previous prompt), and the strict extension has no restore line (the live state is reused, a new checkpoint is
created at 8281). Same `prompt_n` / `cache_n` as the Mac in every row: the fix is in the server's reuse gate and is
backend-independent. Greedy tokens of the cached requests vs the uncached reference differ on both binaries (before:
B1 at token 5, B2 at 10; after: 9 and 5; the uncached p_8k vs the checkpoint-restored p_8k: 31 / 24), i.e. the
ubatch-split numerics of the CUDA path, not a K3 effect (the Mac showed the same: divergences on both binaries).

Side result: the e2e run used the PR's CUDA banded kernel for the real model under `-fa 1` (8218-token prefill in
7.0-7.2 s on the H200, 64 greedy tokens generated per request) without an error; it is not a check of 0001-0003.

## Cost, apps, what is left on Modal

| app (all `modal run`, attached, profile `<profile>`; all stopped on their own, 0 tasks at the end) | GPU | in-function time |
|---|---|---|
| `<modal-app>` `inkling-cuda-verify`, stages (a) (b) (c) | H100 | 1376 s (22.9 min) |
| `<modal-app>` `inkling-cuda-verify-k3::download` | none (8 vCPU) | 447 s |
| `<modal-app>` `inkling-cuda-verify-k3` K3 run (function only, numbers streamed) | H200 | ~5 min |
| `<modal-app>` `inkling-cuda-verify-k3` K3 rerun through the entrypoint (the files) | H200 | 361 s (6.0 min) |
| `<modal-app>` `inkling-cuda-verify --mode final`, 0001..0004 | H100 | 456 s (7.6 min) |

About 44 GPU-minutes in all (H100 ~32 incl. container start-up, H200 ~12), under the 90-minute budget; the launch registry is kept privately. Left behind on purpose: the Modal Volume `inkling-small-gguf`
(91 GiB, the three UD-IQ3_XXS shards under `unsloth/Inkling-Small-GGUF/UD-IQ3_XXS/`) for the next CUDA model run;
`MODAL_PROFILE=<profile> modal volume delete inkling-small-gguf` removes it. Nothing was pushed anywhere; the only local
git change is the worktree `upstream/llama.cpp-m2` (branch `m2-cuda-tolerance`, one commit `2dee8b495` on
`metal-banded-fa`) that produced patch 0004. `patches/README.md`, `PR.md` and `REVIEW.md` were not updated: 0004 and
the CUDA rows above are for the submitter to fold in (the 0002 commit message's "1e-6 tolerance" sentence and the README's
"tightened tolerance" need the CUDA caveat).
