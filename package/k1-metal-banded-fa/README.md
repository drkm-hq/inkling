# K1: Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED` (Inkling on Apple Silicon)

Branch `metal-banded-fa` in `~/code/inkling-contrib/upstream/llama.cpp-inkling`, three commits on `pr-25731`
(ggml-org/llama.cpp#25731 "Add TML Inkling architecture", head `3582c5ee0`), exported as `patches/`. Nothing is
pushed. The previous four-commit version is kept as `metal-banded-fa-v1` (head `90290efad`) because the measurement
logs cite its hashes. Design spec: `research/R2-llamacpp-metal.md` section 7; adversarial review: `REVIEW.md`.

`PR.md` is a fact sheet for the submitter, not a PR description: llama.cpp forbids AI-written PR text and commit messages
(`AGENTS.md`, `CONTRIBUTING.md`); the commit messages on the branch are placeholders with `Assisted-by: Claude Fable 5.1`.

## What is in the branch

| patch | commit | scope |
|---|---|---|
| `patches/0001-metal-add-GGML_OP_FLASH_ATTN_EXT_BANDED.patch` | `fa06bb282` | the kernel, `ggml/src/ggml-metal/*` only, on the op as #25731 defines it |
| `patches/0002-tests-banded-flash-attention-cases.patch` | `caad009a4` | `tests/test-backend-ops.cpp`: helper extension, Inkling-geometry cases, perf cases; tolerance set by the CUDA MMA kernel's f16 accumulation (1e-4 up to 8192 keys, 5e-4 above) |
| `patches/0003-ggml-q_offset-for-the-banded-flash-attention-proposa.patch` | `7c25a0a9b` | **proposal for the op's own PR**: the `q_offset` op parameter in `ggml.h`/`ggml.c`, CPU, CUDA (scalar + MMA, verified on an H100), one line of the Metal dispatcher, the model's padded cache views, the `kv_pad`/`q_offset` test parameters and cases, `test-flash-attn-bias` |
| `patches/0004-*.patch` | (M2, folded into 0002) | the CUDA tolerance hunk as it was measured on the H100; kept for the record, not to be applied |

0001 + 0002 apply and work on `pr-25731` alone (verified, `bench/test-v2-B-*.log`); 0003 applies on top of them.
See `patches/README.md` and `PR.md` section 1 for the submission order.

## Build

```
cd ~/code/inkling-contrib/upstream/llama.cpp-inkling
git checkout metal-banded-fa
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON   # already configured
cmake --build build --config Release -j 12 --target test-backend-ops test-flash-attn-bias llama-cli llama-server llama-bench
```

The Metal library is embedded and compiled at process start; shader errors show up when the first process starts.

## Verify (op level, no model, no GPU lock needed beyond a few seconds of GPU time)

```
./build/bin/test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT_BANDED   # every case against the CPU reference
./build/bin/test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT          # the plain op is untouched
./build/bin/test-flash-attn-bias                                      # the PR's standalone test (CPU)
./build/bin/test-backend-ops perf -b MTL0 -o FLASH_ATTN_EXT_BANDED
./build/bin/test-backend-ops perf -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\]'   # the ceiling
```

Results: 0001 + 0002 alone 72/72 banded cases OK and plain FA 5020 OK / 0 FAIL (`bench/test-v2-B-*.log`, under the
earlier stricter 1e-6 bound); the full series 85/85, plain FA 5020 OK / 0 FAIL, `test-flash-attn-bias` 11/11
(`bench/test-v2-C-*.log`, same bound) and the re-run at the folded head (`bench/test-v3-*.log`); v1 26/26
(`bench/test-4.log`); A1's 165 adversarial cases 165/165 in `review/`. CUDA (M2, H100, `bench/cuda/CUDA-VERIFY.md`):
the series with the folded tolerance 80/80 banded (5 quantized-KV cases not supported by the PR's CUDA op), every
`test-flash-attn-bias` suite PASS, plain FA 4040/4040 untouched, `q_offset` correct on both CUDA paths. The patches
re-apply with `git am` on `pr-25731` and reproduce the branch trees exactly.

## End to end (needs the UD-IQ3_XXS GGUF and the `bench/GPU_LOCK` protocol of `bench/PROTOCOL.md`)

The end-to-end numbers in `PR.md` come from A1's reference-based runs on v1 head `90290efad` (`review/e2e/`) and from
K1's runs on v1 commit `4ddc6ed6d` (`bench/e2e/`); the restructured branch has the same kernel and model code as
`90290efad` but no end-to-end run of its own yet.

**The decode number (`-fa 1` vs `-fa 0` tg128) is pending** a quiet-machine run of:

```
package/k1-metal-banded-fa/bench/ab_quiet.sh <out-dir> [reps] [depths]
```

three interleaved rounds of `-fa 1` / `-fa 0` llama-bench (pp512 and tg128 at d0/4096/16384/32768, `-r 2` each), the
1-minute load average recorded before every run, min and median per setting printed as a table; it takes and
releases `GPU_LOCK` itself and must run with no other jobs and no renderer process busy. The two existing data
points disagree in sign (`PR.md` section 5).

Other scripts in `bench/`: `run_llama_bench.sh`, `run_greedy.sh` (greedy tokens through llama-server, `-fa on` vs
`off`, server stopped with the lock release), `run_probs.sh` + `greedy_probs.py` (top-k probabilities at given
positions), `run_all.sh` (the lock-aware orchestrator of the first window), `perf_table.py` (markdown table from
test-backend-ops perf logs, min-merged over repeated logs).
