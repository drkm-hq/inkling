# R2: llama.cpp PR #25731 (TML Inkling) on Apple Silicon / Metal

Agent R2, 2026-10-28/29. Target: `ggml-org/llama.cpp#25731` "Add TML Inkling architecture" (branch `add-inkling`,
head `3582c5ee05115892e4e96520be0026c3ceb9f829`, 2026-10-27 16:36 -0700, 69 files, +4118/-125, draft, still
merging master daily). Local clone: `~/code/inkling-contrib/upstream/llama.cpp-inkling` (branch `pr-25731`, blobless
partial clone; nothing pushed). Scripts and raw logs: `~/code/inkling-contrib/bench/R2/` and `bench/R2/results/`.
Machine: MacBook Pro M5 Max (18 CPU cores: 6 "Super" + 12 "Performance"; 40-core GPU, Metal 4, tensor API on),
128 GB unified memory, macOS 26.5.1, Command Line Tools only (no `xcrun metal`), cmake 4.4.3.

Every code claim below cites `file:line` in the PR-head checkout; every number says how it was measured.
The real GGUF weights were NOT available (the `hf download` of `unsloth/Inkling-Small-GGUF` UD-IQ3_XXS was
restarted by another process at 21:44 and is at ~11 GB of 91 GiB as of 22:07), so no `llama-bench` or
`llama-server` run was possible; all measurements are op-level, at Inkling-Small shapes, with the PR's own kernels.

## 1. Summary

1. **Metal has no kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED`.** Zero mentions of the op in `ggml/src/ggml-metal/*`
   (grep), `ggml_metal_device_supports_op` falls to `default: return false` (`ggml-metal-device.m:1539` ff.), and
   `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT_BANDED` reports all 13 of the PR's cases `not supported [MTL0]`
   (`bench/R2/results/test-mtl0-fa-banded.txt`). Every other op the Inkling graph uses is supported on Metal
   (section 4).
2. **The reported prefill collapse (336 -> 15.6 tok/s from 2K to 28K context, 2.39 us per context token per prompt
   token) is the CPU fallback of that one op, on the 7 global layers only.** With `-fa on` the scheduler runs the
   banded op on the CPU backend's scalar per-row path (`ggml-cpu/ops.cpp:9561` disables the tiled kernel when
   `rel != nullptr`; `:8989-8996` adds the bias per key). Measured here: 220 / 1404 / 7123 ms per 512-token ubatch
   per global layer at n_kv 2048 / 8192 / 32768 (38-78 GFLOP/s, 18 threads), i.e. a slope of 2.6-3.2 us per
   context token per token for the 7 global layers, matching the field slope of 2.39 us. The 35 SWA layers do not
   use the fallback once the prompt exceeds the SWA cache (`llama-kv-cache.cpp:1291-1293` returns 0 and the layer
   takes the dense Metal path), which is why decode stays at 25-31 tok/s instead of collapsing too.
3. **Two zero-code facts for users today (Metal):** (a) `-fa auto` (the default) resolves to *disabled* for Inkling
   (`llama-context.cpp:631-668`), which routes all 42 layers through the dense Metal path: measured 7.1 / 25.3 /
   117 ms per global layer per 512-token ubatch at 2K / 8K / 32K, i.e. 31x / 55x / 61x faster than the `-fa on`
   fallback, at the price of O(n_kv x n_ubatch x 32 heads) scratch (6.3 GiB per global layer chain at 32K x 512,
   proportionally more at larger ubatch / context; 103 GiB reported at 262K x 1024 on SYCL). The dense chain is
   numerically identical to the PR's banded reference (NMSE 2e-7 vs the CPU banded op on identical inputs,
   `bench/R2/results/verify_dense.log`), so the ROCm "`-fa 0` degrades quality" report does not reproduce at the
   kernel level. (b) Do not pass `-fa on` on Metal with this PR.
4. **A Metal banded flash-attention kernel is the highest-impact contribution.** Its ceiling is the existing Metal
   FA at the same shapes: 0.91 / 3.47 / 14.4 ms per global layer per 512-token ubatch at 2K / 8K / 32K (tensor-API
   kernel, 18-20 TFLOP/s effective) and 0.12 / 0.17 / 0.43 ms per decode token (vec kernel). Versus the current
   `-fa on` fallback that is 240x / 400x / 495x per global layer on prefill; versus the dense path it is 7.8x / 7.3x
   / 8.1x and removes the multi-GiB scratch (154 MiB at 32K x 512). Whole-model attention time per 512-token ubatch
   at 32K context: 50.0 s (`-fa on` today) -> 0.97 s (dense) -> 0.13-0.26 s (banded Metal kernel; the lower number
   needs a 64-aligned KV view, see section 7). Engineering: ~400-600 lines across three Metal kernel families plus
   host glue; the PR already ships the 13 `test-backend-ops` cases that would gate it. Design spec in section 7.
5. **Other findings.** The prompt-cache miss is real at the PR head and F3zz1k's one-line fix still applies
   (`tools/server/server-context.cpp:4098-4198`, section 8a). The "garbage after ~42K" report was retracted by its
   author (it was the q8_0 V path; a 63,809-token prompt was clean) and the PR head now rejects a quantized V cache
   for Inkling at init (`llama-context.cpp:4083-4087`). The ROCm decode slope is the scalar CUDA fallback kernel
   that walks all keys serially per warp because HIP has no Turing MMA (`fattn-banded.cu:93`, `fattn.cu:569`). The
   MTP drafter is dropped at conversion (`conversion/inkling.py:16` skips `model.mtp.*`); only n-gram
   self-speculation is available today. Upstream: the Metal kernel must be a follow-up PR after the op lands via
   the CPU-first text-model PR (#25731 or the #29491 split); it cannot be a standalone `ggml-metal` PR because the
   op does not exist on master.

## 2. Build notes

```
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp ~/code/inkling-contrib/upstream/llama.cpp-inkling
cd ~/code/inkling-contrib/upstream/llama.cpp-inkling
git fetch origin pull/25731/head:pr-25731 && git checkout pr-25731      # HEAD 3582c5ee05115892e4e96520be0026c3ceb9f829
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j 12 --target llama-cli llama-server llama-bench test-backend-ops llama-gguf-split llama-mtmd-cli
```

- Head SHA verified against `gh pr view 25731 --json headRefOid` (identical). Merge base with master:
  `19cce6919` ("tests: fix CI flakes in server cancel and sleep tests (#31330)"). The PR is 45 commits, the last 11
  are "Fix merge conflicts" merges (2026-10-11 .. 10-27), so the graph code is what matters, not the history.
- Build: 0 warnings, 0 errors in `bench/R2/results/build.log` (configure log: `build-configure.log`). Nothing in
  the PR triggers a compiler warning on Apple clang 17 (the `__int128` MSVC failure reported in the PR thread is in
  `tests/test-flash-attn-bias.cpp:322`, not built here because it is not in the target list).
- `GGML_METAL_EMBED_LIBRARY` defaults to ON (`ggml/CMakeLists.txt:242`), so the Metal source is embedded and
  compiled at process start; no `xcrun metal` needed (it is absent on this CLT-only machine). Cost: ~25 s of
  shader compilation on every process start (`compiled 'fa_vec_f16' library in 3.2 sec`, etc.), i.e. every
  `llama-bench`/`llama-server` start pays it.
- The Metal device reports `has tensor = true` (M5 gate at `ggml-metal-device.m:1146-1163`), `has bfloat = true`,
  simdgroup mm/reduction true (`bench/R2/results/fa_pipelines.log`).
- `test-backend-ops` sees 3 devices: MTL0 (109 GiB reported), BLAS, CPU.

## 3. Graph map (as `src/models/inkling.cpp` builds it, Inkling-Small shapes)

Shapes: n_embd 4096, 42 layers, 32 q heads / 8 kv heads x 128, SWA 512 on 35 layers, global layers 5/11/17/23/29/35/41
(from the GGUF `sliding_window_pattern`, per the PR thread), d_rel 16, rel extent 1024 (global) / 512 (SWA)
(`conversion/inkling.py:149-152`), sconv kernel K = 4, MoE 256 experts top-6 + 2 shared (`inkling.cpp:104-117`),
n_ff_exp ~ 2048 (inferred from 276B total; confirm from the GGUF). T = ubatch tokens (512 prefill, 1 decode); N = n_kv
of the global cache = `GGML_PAD(used cells, 256)` (`llama-kv-cache.cpp:1255`); the SWA cache has
`size_swa = GGML_PAD(min(n_ctx, 512 + n_ubatch), 256)` cells (`llama-kv-cache-iswa.cpp:73`) so its n_kv <= 1024 at
ubatch 512 (1536 at ubatch 1024).

Per layer, in order (`inkling.cpp:616-640`):

| # | op (ggml) | shapes, prefill T=512 | decode T=1 | Metal? |
|---|-----------|----------------------|-----------|--------|
| 1 | `build_rs`: GET_ROWS of the packed conv state `[30720, n_cells]` by `rs_s_copy` (+ SCALE zero view, CPY extra) (`llama-graph.cpp:3900-3940`, `inkling.cpp:617`) | `[30720, 1]` | same | yes |
| 2 | RMS_NORM + MUL (attn_norm) | `[4096, T]` | | yes (fused) |
| 3 | MUL_MAT wq/wk/wv/wr (quantized weights) | `[4096,T]`,`[1024,T]`x2,`[512,T]` | | yes |
| 4 | `build_sconv` on k and on v (`inkling.cpp:308-338`): TRANSPOSE view, CONCAT(state `[3,1024,1]`, x^T) -> `[T+3,1024,1]`, CPY of the last 3 columns into the recurrent cache, SSM_CONV with `[4,1024]` kernel -> `[1024,T]`, ADD residual | | | yes (CONCAT, CPY, SSM_CONV, ADD all supported) |
| 5 | RESHAPE q `[128,32,T]`, k/v `[128,8,T]`; RMS_NORM+MUL q_norm/k_norm | | | yes |
| 6 | global layers only: MUL q by tau `[1,1,T]` (log-N scaling input, `inkling.cpp:372-374`) | | | yes |
| 7 | rel bias logits: CONT(TRANSPOSE(attn_rel_proj `[E,16]`)) every graph (`inkling.cpp:380`), MUL_MAT -> `[E, 32*T]` f32 with `GGML_PREC_F32_PEDANTIC` (`:382-383`), RESHAPE `[E,32,T]`, global: MUL tau | 64 MiB (E=1024) | 128 KiB | yes |
| 8a | **FA branch** (`use_banded_flash(il)`, `inkling.cpp:235-247,395-478`): SET_ROWS k/v into the cache (`:416-417`), K/V views of `n_kv_flash = pos_max+1` rows (`:428-443`), CONT of the mask view `[n_kv_flash, T]` f16 (`:462-464`, a copy of up to 32 MiB per global layer at 32K), **FLASH_ATTN_EXT_BANDED**(q `[128,T,32]`, k/v `[128,n_kv_flash,8]` f16, mask, rel `[E,32,T]`, scale 1/128, E) -> `[128,32,T]` (`:466-468`) | | | **NO: CPU fallback** (everything else yes) |
| 8b | **dense branch** (`:479-511`): SCALE q by 1/128 (`:481`), PAD rel to `[E+1,32,T]`, CONT(PERMUTE) -> `[32,E+1,T]`, RESHAPE `[32,(E+1)T]`, GET_ROWS by `rel_idx` `[N*T]` i32 (filled by `set_input_pos_rel_flat`, `llama-kv-cache.cpp:1848-1879`) -> `[32,N*T]`, CONT(PERMUTE) -> `[N,T,32]` (`:484-496`); then `build_attn_mha` (`llama-graph.cpp:2970-3131`): MUL_MAT(k,q) -> `[N,T,32]` f32, ADD kq_b (`:3071`), SOFT_MAX_EXT with f32 mask `[N,T]` (`:3075`), MUL_MAT(v^T, kq) -> `[128,T,32]` (`:3086`), PERMUTE+CONT -> `[4096,T]` | kq, kq_b, get_rows out: 3 x 2 GiB at N=32768 | 3 x 4 MiB | yes |
| 9 | MUL_MAT wo | `[4096,T]` | | yes |
| 10 | `build_sconv` (shortconv_attn `[4,4096]`), ADD residual | | | yes |
| 11 | RMS_NORM+MUL ffn_norm | | | yes |
| 12 | MoE (`inkling.cpp:529-602`): MUL_MAT gate_inp -> `[258,T]` (PEDANTIC), CONT view `[256,T]`, SIGMOID, ADD bias, TOP_K (6), GET_ROWS of the raw logits, CONCAT with the 2 shared logits -> `[8,T]`, NEG/SOFTPLUS/NEG (logsigmoid), SOFT_MAX, SCALE, MUL gscale, CONT view, 3x MUL_MAT_ID routed (`[n_ff_exp,6,T]`, SWIGLU), MUL weights, 5 ADDs over expert views, 3x MUL_MAT_ID shared with constant ids `[2,T]`, MUL gammas, 2 ADDs | | | yes (all) |
| 13 | `build_sconv` (shortconv_mlp), ADD residual | | | yes |

Tail: GET_ROWS(out_ids) after the full stack (`:643-646`, conv states need all tokens), RMS_NORM, SCALE by
`1/logit_scale_denom`, MUL_MAT output `[4096, 201024]`, ADD vocab_mask (`:652-663`).

Roughly 140-150 nodes per layer (the field report on Metal shows 6151 graph nodes with `-fa on`). Which of 8a/8b runs
is decided per layer at graph build: `use_banded_flash` needs `cparams.flash_attn`, head dim 64/128, an f32/f16/bf16
K and V cache, and `get_n_kv_pos_contiguous() > 0` (`inkling.cpp:235-247`): a single-sequence ubatch that is the
monotonic tail of a cache in which position p sits in cell p (`llama-kv-cache.cpp:1268-1314`). Consequences:

- Global layers (cache size n_ctx): contiguous while one sequence is used -> 8a. With `-fa on` the op is the CPU
  fallback; with `-fa auto` the probe marks FA unsupported and `cparams.flash_attn` becomes false -> 8b for all layers.
- SWA layers: `pos_max >= cells.size()` returns 0 as soon as the prompt exceeds `size_swa` (1024-1536 cells)
  (`llama-kv-cache.cpp:1291-1293`) -> 8b (dense, on Metal, with n_kv <= 1536) for any prompt longer than ~1K tokens.
- Multi-sequence ubatches, unified KV with several slots, or any non-identity cell layout -> 8b everywhere
  (`:1269-1271` and `:1307-1311`). The reserve graph pretends the whole cache is contiguous so the worst-case
  allocation is the banded one (`llama-kv-cache.cpp:2943-2956`), which is why a 262K-context server can start on
  `-fa on` and then fail to allocate the dense scratch later on SYCL.

## 4. Metal support matrix

Probe: `test-backend-ops support -b MTL0 -o <ops>` over the PR's test cases (`bench/R2/results/support-mtl0-inkling-ops.txt`,
3968 cases) plus `test -b MTL0 -o FLASH_ATTN_EXT_BANDED` (`test-mtl0-fa-banded.txt`).

| op | Metal | note |
|----|-------|------|
| FLASH_ATTN_EXT_BANDED | **not supported** (13/13 cases) | no `case` in `ggml-metal-device.m:1539` switch, no kernel, no dispatch in `ggml-metal-ops.cpp:485-488` |
| FLASH_ATTN_EXT | supported | head 128 f16/bf16/q8_0/q4_0/... (`ggml-metal-device.m:1728-1775`); three kernel families (section 7) |
| SSM_CONV | supported (60/60) | `kernels/ssm.metal` |
| CONCAT, CPY/CONT, PAD, GET_ROWS (f32, i32 idx), SET_ROWS (i64 idx), ADD, SCALE, RMS_NORM, SOFT_MAX(_EXT, mask f16/f32), MUL_MAT, MUL_MAT_ID, TOP_K, ARGSORT, SIGMOID/SOFTPLUS/NEG (f32) | supported | the only "not supported" rows in the probe are bf16 unary ops and nvfp4/tq1_0 GET_ROWS, which Inkling does not use |
| `GGML_PREC_F32_PEDANTIC` on MUL_MAT | accepted | Metal ignores the hint (it only affects the CUDA cuBLAS path, `ggml-cuda.cu:1557-1600`) |

So: with `-fa off` the whole Inkling graph runs on the GPU; with `-fa on` exactly the global-layer attention runs on
the CPU; with `-fa auto` (default) FA is turned off for the whole context by the probe in `llama-context.cpp:631-668`
(the warning `layer %d is assigned to device MTL0 but Flash Attention is assigned to device CPU` then
`Flash Attention not supported, set to disabled`, `:646-656`). A quantized V cache is rejected for Inkling at init
regardless of backend (`llama-context.cpp:4083-4087`); a quantized K cache with `-fa on` makes `use_banded_flash`
false (`inkling.cpp:231-233,245`) and the layer dense.

## 5. Measurements

Tools (all in `bench/R2/`, compiled against the PR build's `libggml*`):

- `fa_bench.cpp`: one attention variant on one backend at Inkling shapes, with the KV-cache strides of
  `llama_kv_cache::get_k/get_v` (`llama-kv-cache.cpp:1316-1364`), the f16/f32 masks and the exact dense bias chain
  of `inkling.cpp:481-510`. Variants: `fa` (existing `ggml_flash_attn_ext`, f16 KV, f16 mask), `banded`
  (`ggml_flash_attn_ext_banded`), `dense`, `dense-nobias`, `bias` (only the pad/permute/get_rows/permute chain), `qk`.
  Each graph is computed twice to warm up, then N times with `ggml_backend_synchronize`; min and mean are reported.
  `run_matrix.sh` -> `results/fa_bench_matrix.log`; `run_extra.sh` -> `results/fa_bench_extra.log`;
  `run_cpu_decode.sh` -> `results/fa_bench_cpu_decode.log`.
- `sched_bench.cpp`: a 42-layer toy graph (Metal op, attention, Metal op per layer; 7 global, 35 SWA) on a
  `{MTL0, CPU}` `ggml_backend_sched`, `fa` (all Metal) vs `banded` (CPU fallback with the real split structure).
  `run_sched.sh` -> `results/sched_bench.log`.
- `verify_dense.cpp`: identical logical inputs through the CPU banded op, the Metal dense chain and the CPU dense
  chain; reports NMSE and max abs diff. `run_verify.sh` -> `results/verify_dense.log`.
- `results/fa_pipelines.log`: which Metal pipeline each shape compiles (`compiling pipeline` lines).

Conditions: GPU numbers are stable (min of N). CPU numbers were taken while unrelated processes loaded the machine
(load average 11-77 from other unrelated jobs on the machine, including the `hf download`); the first matrix run at
21:49 (load ~12) is quoted for CPU, the re-runs at 21:54-21:58 (load 30-77) were 1.5-3x slower and are kept as
upper bounds. All CPU banded figures should be read as "tens of GFLOP/s, not thread-scalable".

### 5.1 Prefill, one global layer, n_q = 512, 32 heads / 8 KV heads, d = 128, f16 KV, causal mask (ms, min of N)

| n_kv | Metal FA `fa` (ceiling) | Metal dense (8b) | of which bias chain | dense w/o bias | QK matmul only | CPU banded (`-fa on` fallback, 18 thr) |
|-----:|----:|----:|----:|----:|----:|----:|
| 2048 | 0.91 (18.9 TF/s) | 7.14 | 4.79 | 1.43 | 0.65 | 220 (78 GF/s) |
| 8192 | 3.47 (19.8 TF/s) | 25.3 | 15.9 | 5.84 | 1.98 | 1404 (49 GF/s) |
| 32768 | 14.4 (19.1 TF/s) | 117 | 65.0 | 37.4 | 6.32 | 7123 (39 GF/s) |
| SWA layer: n_kv 1024, window 512 | 0.72 | 4.23 | | | | 55.4 |

Scratch of the dense chain (`compute_MiB` in the log): 392 / 1672 / 6280 MiB at 2K / 8K / 32K for one layer's chain
(the gallocr reuses it across layers; it scales with n_kv x n_ubatch x 32 x 4 B x ~3). The FA op needs 34 / 58 /
154 MiB (pad/blk/tmp scratch, `ggml-metal.cpp:230-236`).

Extras (`fa_bench_extra.log`): an unaligned K view, which is what the banded path passes (`n_kv_flash = pos_max+1`),
falls off the tensor-API kernel onto the half8x8 kernel plus the pad pass: 6.39 ms at n_kv 8209 (vs 3.47 at 8192)
and 34.0 ms at 32785 (vs 14.4 at 32768), i.e. 2.3x (`fa_pipelines.log`: `kernel_flash_attn_ext_pad` +
`kernel_flash_attn_ext_f16_dk128_dv128` instead of `kernel_flash_attn_ext_tensor_f16_dk128_dv128`; the gate is
`ne11 % 64 == 0` at `ggml-metal-ops.cpp:3260`). The dense path also slows with the *allocated* cache size because
the transposed V cache is strided by `kv_size` (`llama-kv-cache.cpp:1357-1363`): 204 ms at n_kv 32768 with a
65536-cell cache vs 117 ms with a 32768-cell cache.

CPU thread scaling of the banded op at 512 x 8192: 6 threads 2766 ms, 12 threads 1609 ms, 18 threads 1404 ms
(1.97x for 3x the threads): the per-row scalar path (`ggml-cpu/ops.cpp:8846-9060`, one `vec_dot` per key, f32 VKQ
accumulation forced by `:8925`) is latency bound, consistent with the field observation of ~576% CPU on 18 threads.
The tiled CPU kernel that `rel` disables runs the same shape in 134 ms (`CPU fa 512 8192`: 513 GF/s), so the
fallback is also ~10x slower than the CPU could do with a banded-aware tiled kernel.

### 5.2 Decode, one layer, n_q = 1 (ms, min of N)

| n_kv | Metal FA (vec kernel) | Metal dense | CPU banded, 18 thr, first run | CPU banded, re-runs (loaded) |
|-----:|----:|----:|----:|----:|
| 2048 | 0.123 | 0.226 | 4.8 | 8.6-29 |
| 8192 | 0.171 | 0.345 | 10.8 | 14-100 |
| 32768 | 0.429 | 1.03 | 15.3 | 27-150 |
| SWA: 768, window 512 | 0.138 | 0.141 | 5.6 | 1.8-12 |

Single-op Metal graphs carry ~0.1 ms of command-buffer commit/wait, so the per-layer Metal decode numbers are
overestimates; the 42-layer `sched_bench` all-Metal graph decodes in 3.5 ms (8K) / 4.3 ms (32K) per token for all
attention layers together (min of 30, loaded machine; medians 14 / 7.7 ms).

### 5.3 The scheduler fallback itself (`sched_bench.log`, load average 50-77 during the run)

| graph (42 layers, 7 global) | splits | decode 8K: median / min | decode 32K: median / min | prefill 512 @ 8K: median / min |
|---|---:|---|---|---|
| all Metal (`ggml_flash_attn_ext`) | 1 | 14.3 / 3.5 ms | 7.7 / 4.3 ms | 55 / 48 ms |
| banded on CPU (`-fa on` structure) | 85 | 98 / 42 ms | 149 / 94 ms | 12.3 / 11.1 s |

85 splits for 42 fallback ops is the structure behind the "graph splits = 86" in the field report. At the PR head only
the 7 global layers fall back once the prompt exceeds the SWA cache, so the production decode penalty is ~1/6 of the
toy-graph numbers: ~15-25 ms per token at 8-32K (7 CPU ops of 2-15 ms each plus 14 split synchronisations), which
is most of the 32-40 ms per token behind the observed 25-31 tok/s.

### 5.4 Numerical equivalence of the dense Metal path (`verify_dense.log`)

| shape | Metal dense vs CPU banded | CPU dense vs CPU banded |
|---|---|---|
| 512 x 2048 | NMSE 2.1e-7, max 2.9e-5 | 1.4e-6 |
| 512 x 8192 | 2.5e-7, max 1.5e-5 | 5.5e-6 |
| 512 x 1024, window 512 | 1.9e-7 | 5.1e-7 |
| 1 x 8192 | 2.2e-10 | 5.7e-6 |
| 17 x 4100 | 1.9e-7 | 2.9e-6 |

The dense chain on Metal reproduces the PR's banded semantics (bias inside the band, zero outside, causal/window
mask) to f16-level tolerance (the PR's own test tolerance for these shapes is 5e-4 to 1e-3, `test-backend-ops.cpp:8768-8775`).

## 6. Root cause of the prefill collapse (quantitative)

Whole-model attention time per 512-token ubatch, 7 global + 35 SWA layers, from section 5.1 (SWA layers dense on
Metal at 4.23 ms each, as the PR head routes them for prompts longer than the SWA cache):

| context N | `-fa on` today (7 x CPU banded + 35 x dense) | `-fa auto`/`off` (all dense on Metal) | banded Metal kernel, aligned KV (7 x FA + 35 x FA-SWA) | same, unaligned KV |
|---:|---:|---:|---:|---:|
| 2048 | 1.69 s (303 tok/s ceiling) | 0.198 s (2,590) | 0.032 s | ~0.045 s |
| 8192 | 9.98 s (51 tok/s) | 0.325 s (1,575) | 0.050 s | 0.070 s |
| 32768 | 50.0 s (10.2 tok/s) | 0.967 s (530) | 0.126 s | 0.263 s |

Field data (M5 Max, UD-IQ3_XXS, `-fa on`, ubatch 1024): 85 tok/s at 4K, 40 at 10K, 15.6 at 28K, slope 2.39 us per
context token per prompt token. The model here gives 51 tok/s at 8K and 10 at 32K from attention alone, slope
2.63 us (2K->8K) to 3.18 us (8K->32K) with 7 global layers; the 10-20% gap is the bigger ubatch (fewer passes over
K) and measurement load. Conclusion: the collapse is entirely the global layers' banded op running on the CPU's
scalar path at 40-80 GFLOP/s; it is not the bias materialisation (that is the *dense* path, which is 50-60x faster),
not memory traffic, not the KQ matmul (6 ms at 32K on Metal), and the 14 split synchronisations per ubatch are
negligible against seconds of CPU work. The "ctx-size independence" observed in the field (298 vs 284 tok/s for a
2.5K prompt at `-c 4096` vs `65536`) is consistent: the CPU op sees `n_kv_flash = pos_max+1`, not the allocated
cache.

Decode degrades with context on Metal for the same reason, but only through the 7 global layers (section 5.3): the
CPU op's split-KV decode path (`ggml-cpu/ops.cpp:9504-9533`) costs 5-15 ms per layer at 2K-32K plus two
synchronisations, i.e. roughly +0.3-0.5 us per context token per decoded token, against ~6 ns on the Metal vec
kernel. The CUDA reports (flat decode to 2.2K at 6.45 tok/s over RPC) are a different problem (graph re-serialisation
over RPC), not this one.

## 7. Metal banded-FA design spec

### 7.1 Op semantics to implement (`ggml/include/ggml.h:2520-2531`, `ggml/src/ggml.c:5597-5645`)

`dst[dv, h, q, b] = sum_k softmax_k( scale * <q_vec, k_vec> + B(q,h,k,b) + mask[k, q, h % ne32, b % ne33] ) * v[k]`,
with `dist = q + (n_kv - n_q) - k` and `B = rel[dist, h, q, b % ne3]` iff `0 <= dist < E` else 0. The band is
*the bias*, not the attention: the softmax still runs over every unmasked key (the SWA mask bounds SWA layers; global
layers attend to all N keys and get the bias on the last E of them). No ALiBi (`max_bias` 0), no sinks, no softcap,
`op_params[3] = GGML_PREC_F32` (`inkling.cpp:468`), `scale = 1/128` (not 1/sqrt(128), `inkling.cpp:467`).
Sources: `src[0]` q f32 `[128, n_q, 32, 1]` permuted view (nb1 = 128*32*4), `src[1]`/`src[2]` K/V f16 cache views
`[128, n_kv, 8, 1]` with nb1 = 2048 B (row = cell), nb2 = 256 B (head) — i.e. `ns10 = ns20 = 1024` in the kernel's
terms — `src[3]` f16 mask `[n_kv, n_q, 1, 1]` contiguous (the PR `ggml_cont`s it, `inkling.cpp:462-464`), `src[5]`
rel f32 `[E, 32, n_q, 1]` contiguous (always the output of a mul_mat in this model; the CPU op also accepts f16/bf16,
`ops.cpp:8820-8842`; the CUDA MMA path requires f32 contiguous, `fattn.cu:558`). `src[4]` (sinks) is unused. GQA
ratio 4 (`ikv2 = iq2/(ne02/ne_12_2)` as in the existing kernels). Output f32 `[128, 32, n_q, 1]`.

### 7.2 Where the existing kernels add the mask (= where the bias goes)

All three Metal FA families already add a per-(query, key) additive term from the mask with a per-head slope, so the
bias is a second additive term with its own gather:

- `kernels/fa_common.metal` (half8x8 simdgroup kernel, `kernel_flash_attn_ext`, used for n_q >= 20 unless the
  tensor kernel applies): scores for Q = 8 query rows x C = 64 keys are stored to threadgroup `ss` (`:341`, `:405`),
  then the online softmax at `:412-454` reads two keys per lane (`ss2[j*SH/2 + tiisg]`) and adds the mask at
  `:424-431`. Add `rel` there: query `iq1 + j`, keys `ic + 2*tiisg`, `+1`; `dist = (iq1 + j) + q_off - key`; skip the
  whole block when `dist_max = iq1 + Q - 1 + q_off - ic < 0` or `dist_min = iq1 + q_off - (ic + C - 1) >= E`
  (uniform branch, so the bias costs nothing outside the band). Keep the *absolute* `ic` for the band test: the
  kv-pad path resets `ic = 0` for the last partial block (`:194-229`). The `blk_cur == 2` shortcut (mask block all
  zero, `:425`) must not skip the bias. Mask-block precompute (`fa_aux.metal:128-181`) can stay mask-only.
- `kernels/fa_tensor.metal` (Metal 4 tensor API, M5+, n_q >= 20, `ne11 % 64 == 0`): scores `s[ii]` at `:170-174`,
  mask added at `:182-186` for key `ic + ii*NW + tiisg` and query `iq1 + j`; add the bias in the same loop
  (one f32 gather per in-band score; the `rel` row for a fixed query/head is contiguous in `dist`, so 32 lanes read
  32 consecutive floats in reverse order — coalesced). Lazy rescaling (`:196-206`) is unaffected.
- `kernels/fa_vec_common.metal` (decode, n_q < 20, `ggml-metal-ops.cpp:3186-3194`): scores `mqk[qq][tx]` for key
  `ic + NE*tx + ty` combined with the mask at `:376-396`; add the bias there (and in the sparse-index branch using
  `spidx`). With Q = 1 and C = 32 the band test is per block as above; the split-KV reduce (`kernel_flash_attn_ext_vec_reduce`)
  needs no change because the bias only changes the scores.

Plumbing: a new function constant (`FC_FLASH_ATTN_EXT + 6` / `_VEC + 6` / `_TENSOR + 4`; note `_VEC + 5` is already
`has_sparse`, `fa_vec_common.metal:15`), new fields in `ggml_metal_kargs_flash_attn_ext` (`ggml-metal-impl.h:417-450`)
and `_vec` (`:452-486`): `int32_t ne_rel0 (E), ne_rel3, q_off; uint64_t nb_rel1, nb_rel2, nb_rel3`; one more buffer
binding per family (mm: after `blk`, index 9; tensor: index 8; vec: index 9). Host: a `case GGML_OP_FLASH_ATTN_EXT_BANDED`
next to `ggml-metal-ops.cpp:485-488` into `ggml_metal_op_flash_attn_ext` with `rel = op->src[5]`; relax the
`op->op == GGML_OP_FLASH_ATTN_EXT` asserts in `use_vec/use_kv_f16/use_tensor/extra_*` (`:3187, :3199, :3223, :3346,
:3404, :3443, :3472, :3497`) and in the pipeline getters (`ggml-metal-device.cpp:1790` ff.); a `supports_op` case
mirroring `ggml-metal-device.m:1728-1775` plus `rel` checks (f32, contiguous rows, `ne1 == n_head`, `ne2 == n_q`,
`ne3 in {1, ne03}`); the scratch-size hook `ggml-metal.cpp:230-236` must include the new op (the generic
`ggml_op_alloc_size_may_expand` already does, `ggml-backend.cpp:2225`). Reuse: ~95% of the three kernels, 100% of the
pad/blk/reduce auxiliaries and of the quantized-KV handling (vec kernels dequantize inline; mm/tensor kernels use the
dequant-to-f16 pre-pass `ggml-metal-ops.cpp:3198-3220`), so f16, bf16, q8_0, q4_0 KV come for free at the op level.

### 7.3 The alignment problem (worth 2.3x on its own)

The op derives the query offset from `n_kv - n_q` (`ggml.h:2521`, `ops.cpp:8991`, `fattn-banded.cu`), which forces the
model to pass the exact `n_kv_flash = pos_max + 1` K/V view (`inkling.cpp:425-443`). On Metal that is rarely a
multiple of 64, so the tensor kernel is excluded (`ggml-metal-ops.cpp:3260`) and the half8x8 kernel plus pad pass runs
at 2.3x the time (section 5.1 extras). Two fixes, in order of preference: (1) add an explicit `q_offset` op param
(`op_params[5]`; `[4]` is taken by `n_kv_max`, `ggml-metal-ops.cpp:3273`) honoured by the CPU, CUDA and Metal
implementations, and let the model pass the 256-padded view whose padded cells the KQ mask already masks; (2) Metal-only:
run the tensor kernel on the floor-64 part and the mm kernel on the remainder with a merge, or extend the pad buffer
mechanism to the tensor kernel. (1) is a 20-line ggml API change that also removes the fragile reliance on tail
alignment that `get_n_kv_pos_contiguous` enforces.

### 7.4 Expected speedups (attention-only, from section 6; whole-model needs the weights)

- Prefill per global layer at 8K / 32K: 1404 -> 3.5-6.4 ms / 7123 -> 14.4-34 ms (220-400x / 210-495x vs `-fa on`
  today; 4-7x / 3.4-8x vs the dense path). Whole-model attention per 512-token ubatch at 32K: 50 s -> 0.13-0.26 s.
  Model-level prefill at 32K then stops being attention-bound: the MoE forward (12.3 TFLOP per 512 tokens at an
  effective 10-20 TFLOP/s plus ~90 GB of expert reads at ~500 GB/s) is ~0.6-1.4 s per ubatch, so expect roughly
  350-700 tok/s at 32K versus ~10 today (and ~300-400 with `-fa off`).
- Decode at 32K: the 7 global layers go from 5-15 ms + 2 syncs each (section 5.2/5.3) to 0.4 ms each; whole-model
  attention from ~15-25 ms to ~4-8 ms per token. With ~10-15 ms of weight streaming per token, 25-31 tok/s should
  become ~40-55 tok/s at short context and stay flat to 32K (today each 1K of context adds ~0.3-0.5 ms).
- The bias costs the kernel one f32 gather per in-band score: 512 x 1024 x 32 x 4 B = 64 MiB per global layer per
  ubatch (E = 1024), ~5% of the K/V traffic at 32K; expect the banded kernel within 10-20% of the plain FA numbers.
- Effort: 400-600 lines (3 kernels + host + supports_op + alloc size), 3-5 days for someone who knows this code,
  +1-2 days for the `q_offset` change across backends; validation = the PR's 13 `test-backend-ops` cases
  (`tests/test-backend-ops.cpp:12380-12394`, including 512 x 32768 and the 16403 unaligned case), new perf cases
  (n_q 512/1, n_kv 8192/32768, nh 32, nr 4), `verify_dense`-style NMSE vs the CPU op, then `llama-perplexity` at
  8K-32K and `llama-bench -fa 1 -d 0/4096/16384` once the weights are present.

## 8. Other opportunities, ranked

1. **Metal banded FA kernel** (section 7): the only change that fixes prefill, decode and memory at once on Apple
   Silicon; unlocks quantized KV on Metal once the model-side gate (`inkling.cpp:231-233`) and the init guard
   (`llama-context.cpp:4083-4087`) are relaxed.
2. **Immediate operational fix, zero code: run with `-fa off` (or leave `-fa auto`) on Metal.** 31-61x faster
   prefill at 2K-32K than the reported `-fa on` runs, numerically identical (section 5.4). Costs: ~6 GiB of scratch
   per 32K of context at ubatch 512 (scales with ubatch; keep `-ub 512`), and the transposed-V stride penalty with
   large `-c` (section 5.1). The PR thread should be told this explicitly: every Apple user in it ran `-fa on`.
3. **Prompt-cache reuse for the hybrid memory** (section 8a): one-line server change; 5x on multi-turn agent loops
   per the field report. Backend-independent, cheapest high-value item after (2).
4. **`q_offset` + 64/256-aligned KV view** (section 7.3): 2.3x on the global layers' FA at long context on Metal
   (and simplifies CUDA/CPU too).
5. **Decode graph overhead**: ~6,000 nodes per token (field report) at 25-31 tok/s; the shortconv chain
   (CONCAT + CPY + SSM_CONV + ADD, four times per layer), the MoE routing chain (~20 small ops per layer) and the
   per-graph `CONT(TRANSPOSE(attn_rel_proj))` (`inkling.cpp:380`, 42 extra nodes, should be done at load) are
   fusion candidates. Unmeasured here (needs the weights and `GGML_METAL_PROFILE`), but at ~2-5 us per Metal
   dispatch 6,000 nodes are 12-30 ms of the 32-40 ms token budget.
6. **Dense-path hygiene** (if the kernel is delayed): pass `kq_b` as the 3-D soft_max mask to drop the separate ADD
   (saves one 2 GiB pass per global layer at 32K), or build the bias only over the band as a `[E+1, T, 32]` tensor
   plus a window-mask instead of the full `[N, T, 32]`; and avoid the per-layer `ggml_cont` of the mask view
   (`inkling.cpp:462-464`, 7 x 32 MiB per ubatch at 32K).
7. **ROCm/HIP**: give the scalar `flash_attn_ext_banded_f32` kernel a split-KV grid (several warps per (q, head) +
   a reduce) or route HIP to the tile kernels; today one warp walks all N keys serially (`fattn-banded.cu:93-140`,
   grid `(n_q/4, n_head, batch)` at `:229-230`), which is the 15.5 us-per-context-token decode slope (section 8c).
8. **MTP / speculative decoding** (section 8d): large effort, needs the drafter weights in the GGUF first.

### 8a. Prompt-cache / sconv state (F3zz1k's fix)

Mechanism at the PR head: `llama_memory_hybrid_iswa::seq_pos_min` returns `max(attn, recr)`
(`llama-memory-hybrid-iswa.cpp:178-181`); the recurrent side keeps one live cell per sequence whose `pos` is the last
decoded position (`llama-memory-recurrent.cpp:391-404`), so the server's `pos_min` is the *tail*. The server's reuse
gate `pos_min >= pos_min_thold` with `pos_min_thold = pos_eval - n_swa + 2` (`tools/server/server-context.cpp:4105,
4157`) therefore fires on every request, including a strict extension that needs no rewind; the checkpoint search then
requires `cur.pos_min < pos_min_thold || cur.pos_min == 0` (`:4169`), which a checkpoint whose recorded `pos_min` is
also a tail can never satisfy, so `do_reset` -> `n_past = 0` and the whole prompt is re-prefilled (`:4198` logs
"forcing full prompt re-processing ... hybrid/recurrent memory"). Inkling is not in `llm_arch_supports_rs_rollback`
(`llama-arch.cpp:1321-1340`) so its context classifies as `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`
(`common/common.cpp:1596-1640`: partial `seq_rm` fails via `llama-memory-hybrid-iswa.cpp:149-156` ->
`llama-memory-recurrent.cpp:186-201`). F3zz1k's commit `371c74a` (server-context.cpp, +9/-1, https://github.com/F3zz1k/llama.cpp/commit/371c74a0553663d9d6e28b4ebecac9542f076c19) applies the `n_swa` back-off only when `pos_min` is a genuine
window start (seq_rm type PART), not for FULL/RS memories; the surrounding code has moved (the threshold is now
`pos_eval - n_swa + 2`, `:4105`) but the gate and the predicate are the same, so the fix ports as: for FULL/RS
contexts use `pos_min_thold = pos_eval` (strict extension passes because the tail `pos_next - 1 < pos_next`; a rewind
still looks for a checkpoint with `cur.pos_max < pos_next`, `:4166-4169`). Verified by the field (brywil: cached turns
5x faster). Note the server creates checkpoints for FULL/RS memories (`:4296-4309`), so multi-turn rewinds work once
the gate is fixed; `n_rs_seq`-style rollback support for Inkling (`llama-arch.cpp:1321`) would be the fuller fix.

### 8b. The "garbage after ~42K" report

Retracted by its author in the PR thread (02142bbc3 build): it was the q8_0 V-cache path (`@@@@` with a 6-token prompt),
and 35,009- and 63,809-token prompts with f16 KV were clean on both `-fa on/off`. The PR head now refuses a quantized V
cache for Inkling at init (`llama-context.cpp:4083-4087`), so that failure cannot recur. Code review for the asked-for
causes: (i) int32 index math: the CPU banded path is int64 throughout (`ops.cpp:8820-8842, 8991`), the Metal dense
kernels use `int32_t ne*` with `uint64_t nb*` (per-dimension extents stay far below 2^31 up to 1M context; the i32
`rel_idx` values are `i*(E+1)+rel <= 1024*1025`), so no overflow below ~1M context with ubatch 512; (ii) compute-buffer
sizing: the reserve graph sizes the *banded* layout (`llama-kv-cache.cpp:2947-2949`) while a multi-slot/unified or
non-contiguous cache later takes the *dense* layout, which can be 10x larger (section 5.1) — a late `ggml_gallocr`
re-allocation or failure rather than silent garbage; (iii) the fallback path has no state. Confirming experiment (needs
the weights): `llama-server -fa on -c 65536 -ub 512`, send a 42K prompt twice and a 28-token prompt after it, greedy;
repeat with `-fa off`; then `-np 4` (see 10.5).

### 8c. The ROCm/Metal decode slope with context

ROCm (+15.5 us per context token, `-fa 1`): HIP lacks `turing_mma_available`, so `ggml_cuda_flash_attn_ext_banded_use_mma`
is false (`fattn.cu:541-570`) and the scalar `flash_attn_ext_banded_f32` kernel runs: one warp per (query, head),
`for ik in 0..n_kv` serial over keys with a mask load, a 4-element dot, a warp reduce, an `expf` and a V update per
key (`fattn-banded.cu:93-140`), grid `(ceil(n_q/4), n_head, batch)` (`:229-230`). At decode that is 32 warps on a
40-CU gfx1151 with a ~2 us dependent-load chain per key: 7 global layers x 2.2 us = the observed 15.5 us per context
token; `-fa 0` (dense, GPU-wide matmuls) being 3x faster at depth follows. Metal: the slope comes from the CPU split-KV
fallback (section 5.2/5.3), ~0.3-0.5 us per context token per decoded token for the 7 global layers, plus 14 split
synchronisations per token; it is smaller than ROCm's because the CPU path parallelises over keys.

### 8d. MTP head and speculative decoding

The converter skips `model.mtp.*` (`conversion/inkling.py:16`, together with `model.visual.*`/`model.audio.*`),
so the 8-layer drafter is not in the GGUF and `llama-arch.cpp` registers no NEXTN tensors for Inkling. llama.cpp does
have a generic MTP context (`LLAMA_CONTEXT_TYPE_MTP`, `include/llama.h:227`; `LLM_GRAPH_TYPE_DECODER_MTP`,
`llama-graph.h:45`; spec type `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`, `common/common.h:180`) used by deepseek/bailing/
dots/cohere-style single-layer nextn heads (`src/models/{deepseek2,bailingmoe2,...}.cpp`), but Inkling's drafter is a
full 8-layer stack with its own recurrent sconv state, so it would need its own `LLM_GRAPH_TYPE_DECODER_MTP` graph,
converter support and a KV/conv-state story for the drafter. Available today without weights changes: n-gram
self-speculation (`--spec-type ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache`,
`common/speculative.cpp:81-85`), which helps on repetitive agent/coding output, and a standalone draft model only if
one with the same 201,024-token vocabulary exists (none known). Judgement: MTP is a 1-2 week item with the drafter
weights, second-order next to the attention kernel and the prompt cache.

### 8e. Upstream path

- Policy signals in the thread: the bot flags multi-backend + large PR; ngxson asks for 4 PRs (chat parser, text model,
  audio/vision, ggml backend changes); pwilkin rejects the CUDA constants; JohannesGaessler (2026-09-22) "categorically"
  refuses to review the CUDA part of the initial PR and wants CPU-only first; issue #29491 (aetherbird) proposes exactly
  that split (PR 1 CPU incl. the op's CPU implementation and `test-backend-ops` cases, PR 2 CUDA, PR 3+ Vulkan/misc).
- Therefore a Metal banded kernel is **PR 3-class work: a follow-up PR against master once the op exists there**,
  i.e. after PR 1 (or #25731 trimmed to CPU) merges. A standalone `ggml-metal` PR cannot land first (no
  `GGML_OP_FLASH_ATTN_EXT_BANDED` on master, `ggml.h:576`); a PR against `add-inkling` would only live in Daniel's
  branch. Practical plan: develop and validate on `pr-25731` now (this clone), keep the diff confined to
  `ggml/src/ggml-metal/*` + the `supports_op`/alloc-size hooks + perf test cases, rebase onto the merged op, and ask
  ggerganov (Metal owner) for review; mention the `q_offset` API point on #29491 early because it touches all backends.
- Tests the project expects: the op's correctness cases in `tests/test-backend-ops.cpp` (the PR's 13
  `test_flash_attn_ext_banded` cases at `:12380-12394` will run on MTL0 automatically once `supports_op` is true;
  they compare against the CPU op), `perf` cases, and no new standalone test binaries (the PR's
  `test-flash-attn-bias`/`test-flash-attn-generic-hash` executables, `tests/CMakeLists.txt:339-346`, are the kind of
  thing reviewers will ask to fold into `test-backend-ops`).
- Design points reviewers will raise that we should pre-empt: the CUDA path smuggles `rel` through the sinks pointer
  and tags the branch with a negative `max_bias` (`fattn-common.cuh:1236-1242`, `fattn-mma-f16.cuh:1989-1991, 2041-2043`) and
  bumps the running max heuristically (`fattn-mma-f16.cuh:741-751`, the "8.0f*0.6931f" pwilkin objected to); a Metal
  implementation should pass `rel` as a real buffer with its own function constant and should not need the heuristic
  (f32 accumulation in `ss`/`co`, lazy rescaling already handles large logits). Whether the bias should be a 6th
  source of `ggml_flash_attn_ext` instead of a new op is a maintainer call worth asking on #29491.

## 9. Hazards

1. **Moving target.** 45 commits with daily "Fix merge conflicts" merges; the SWA-layer routing described in section 3
   is the PR-head behaviour and differs from the July builds in the thread (86 splits implies all 42 layers fell back
   then). Re-check `inkling.cpp:235-247` and `llama-kv-cache.cpp:1268-1314` before building on them.
2. **CPU numbers are noisy** (load average up to 77 from unrelated jobs); only the first-run prefill figures are
   quoted as point values and they agree with the field slope within 10-30%. GPU figures are min-of-N and stable.
3. **Dense-path memory**: 6.3 GiB per global-layer chain at 32K x 512, linear in both; ubatch 2048 at 64K context is
   ~50 GiB and will fail to allocate (103 GiB at 262K x 1024 on SYCL). The reserve graph sizes the banded layout, so a
   server can start and only later hit the dense layout (multi-slot, non-contiguous cells). Keep `-ub 512` on `-fa off`.
4. **`-fa on` on Metal silently falls back**: there is no warning at runtime (the only hint is the split count); the
   `-fa auto` probe warning is the single message that says the op is unsupported.
5. **Multi-slot `????` output** (field report, ROCm, `-np 4` + unified KV, 18-token prompt; `-np 1 --no-kv-unified`
   fixes it): not reproduced here (no weights). Code reading: with several slots the global cache is no longer
   position-contiguous, so layers switch between the banded and dense paths request by request; the recurrent state
   is addressed by `kv_head` with per-sequence rows (`inkling.cpp:321-331`, `build_rs` at `llama-graph.cpp:3900-3940`),
   which is the standard pattern, so the likelier suspects are the unified-cache mask/`rel_idx` stream handling
   (`inkling.cpp:392-393, 500-506`) rather than the conv state. Experiment: `-np 2` with one active request, `-fa off`
   vs `on`, `--kv-unified` on/off.
6. **q8_0 V**: rejected at init for Inkling; q8_0 K + `-fa on` quietly routes to the dense path. After the Metal kernel
   lands, relaxing `inkling.cpp:231-233` and `llama-context.cpp:4083-4087` must be re-validated with
   `test-backend-ops` q8_0 banded cases (none exist yet in the PR).
7. **Alignment fragility**: the op's `n_kv - n_q` convention ties correctness to the tail-contiguity check; any
   padding of the K view without a `q_offset` silently shifts the bias (section 7.3).
8. **Startup cost**: embedded-library builds recompile ~25 s of Metal shaders at every process start; irrelevant for
   servers, annoying for `llama-bench` sweeps (use a Xcode-built `default.metallib` or keep processes alive).
9. **Weights**: the download was restarted by another process at 21:44 (two `hf download` processes exist); at
   ~11 GB of 91 GiB as of 22:07 it will not finish in this session. `llama-bench` (pp512/tg128 at d 0/4096/16384,
   `-fa 0` vs `1`) and the long-prompt server test are left to a later agent; the commands are in section 7.4 and 8b.

## 10. Files

- Report: `~/code/inkling-contrib/research/R2-llamacpp-metal.md` (this file).
- Clone/build: `~/code/inkling-contrib/upstream/llama.cpp-inkling` (`build/bin/{llama-cli,llama-server,llama-bench,test-backend-ops,llama-gguf-split,llama-mtmd-cli}`).
- Bench sources: `bench/R2/fa_bench.cpp`, `sched_bench.cpp`, `verify_dense.cpp`, runners `run_matrix.sh`, `run_extra.sh`,
  `run_cpu_decode.sh`, `run_sched.sh`, `run_verify.sh`.
- Raw logs: `bench/R2/results/{build.log, build-configure.log, clone.log, support-mtl0-inkling-ops.txt,
  test-mtl0-fa-banded.txt, fa_bench_matrix.log, fa_bench_extra.log, fa_bench_cpu_decode.log, sched_bench.log,
  verify_dense.log, fa_pipelines.log}`; PR material (not redistributed here: the PR #25731 diff, commits, body, comments and reviews, and the F3zz1k fork commit, all fetched from GitHub).
