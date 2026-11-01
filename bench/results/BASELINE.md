# BASELINE: Inkling-Small on the M5 Max (MLX 2-bit, llama.cpp UD-IQ3_XXS)

M1, 2026-10-28/29. Protocol: `bench/PROTOCOL.md`, `research/00-method.md` section 2. Scripts:
`bench/baseline/`. Raw logs and JSONL: `bench/results/` (`mlx-*.{log,jsonl}`, `llama-bench-*.{log,jsonl}`,
`llama-server-*.{log,jsonl,client.log}`). Tables below are produced by `python bench/baseline/summarize.py`.
Nothing was pushed or uploaded.

## Headline numbers (median of 3 timed runs unless noted; greedy, 128 new tokens, prompts rendered with reasoning effort "none")

| engine / quant | condition | prompt tokens | prefill tok/s | TTFT s | decode tok/s | peak memory |
|---|---|---|---|---|---|---|
| MLX 2-bit, **stock mlx-vlm 0.7.11 path** | loaded machine, load1 10-30 + GGUF download (22:18-22:36); the only stock rows | 239 / 2085 / 8231 / 32807 | 527 / 550 / 564 / 471 | 0.47 / 3.8 / 14.6 / 69.8 | 21-27 / 21.4 / 20.5 / 22.7 | 89.4 / 94.3 / 100.4 / 112.7 GB |
| MLX 2-bit, stock wheel, 22:09 smoke (idle GPU, 20-token prompt, 48 tokens) | 1 sample | 20 | (compile) | - | 44.9 | 88.5 GB |
| MLX 2-bit, **K4's modified checkout** (`upstream/mlx-vlm` ee1be25: bounded sliding-window caches, quantized global KV, strided sconv, route kernel) | quiet machine, load1 4 (hold 2, 03:48-03:56) | 32807 | 886 [884-897] | 37.1 | 37.8 [35.4-38.6] | 102.3 GB (mx peak) |
| MLX 2-bit, K4's checkout | quiet machine, p_short warm-up (1 sample, 32 tokens) | 239 | (compile) | - | 48.2 | 89.5 GB |
| llama.cpp UD-IQ3_XXS `-fa 0` (dense Metal) | llama-bench pp512 / tg128, depth 0 / 4096 / 16384 / 32768 | 512 | 219 / 275 / 235 / 185 | - | 18.5 / 18.0 / 16.0 / 13.1 | 91.2 GiB weights + KV + scratch |
| llama.cpp `-fa 0` server, `-c 16384` | quiet machine (hold 2) | 239 / 2085 / 8231 | 162* / 211 / 193 | 1.5 / 9.9 / 42.8 | 18.6 / 18.5 / 16.7 | RSS 90.9 GiB |
| llama.cpp `-fa 1` (CPU banded fallback) | llama-bench depth 0 / 4096 | 512 | 71.7 / 34.3 | - | 12.6 / 11.0 | |
| llama.cpp `-fa 1` server, `-c 16384` | quiet machine (hold 2; p_8k single run) | 239 / 2085 / 8231 | 147 / 95 / 39.5 | 1.7 / 22.1 / 208 | 14.0 / 15.2 / 8.4 | RSS 92.4 GiB |

\* first-test warm-up skew (page-in), 173 % spread. The two engines run different quantizations (2-bit MLX vs 3.06 bpw
GGUF), so rows are not an engine-vs-engine comparison; each engine's own before/after is what the campaign measures.
CAVEAT discovered at the end: hold 2's MLX rows ran on K4's modified `upstream/mlx-vlm` checkout (its commits landed
03:23-03:27, minutes before the hold; the per-run cache record shows `InklingCacheList(SlidingWindowKVCache, ...)` for
35 layers and 0.97 GiB of cache at 32K instead of 0.7.11's `CacheList(KVCache, ...)` and 5.6 GiB), so 886 / 37.8 tok/s
is K4's after-number on a quiet machine, not the stock baseline. The stock engine on a quiet machine is known only
from the 22:09 smoke (44.9 tok/s decode at 20 tokens); its loaded-machine rows above are the baseline of record.
llama.cpp server decode moved 4-7x (2.7-4.8 vs 16.7-18.6 tok/s) when its `-c 70000` reserve left no free memory. Before/after claims must be taken
back-to-back in the same quiet window with the load and the competing processes recorded (the JSONL rows here do).

## 0. Machine, software, inputs

- MacBook Pro, Apple M5 Max (18 CPU cores: 6 + 12; 40-core GPU, Metal 4, tensor API on), 128 GB unified memory
  (`recommendedMaxWorkingSetSize` 115.4 GB = 107.5 GiB), macOS 26.5.1 (25F80). Not a quiet machine: other jobs
  ran `test-backend-ops`, cmake `-j16` builds and CPU micro-benchmarks throughout, desktop applications
  sat at 85-115 % CPU each, and the 91 GiB GGUF download (`hf download`, ~3 GB/min) ran until
  02:33Z. Load average 10-30 during the MLX runs. Every run row records `load1` and the competing processes it saw.
- Python 3.12.14 venv `~/.venvs/inkling`: mlx 0.32.4 (mlx-metal 0.32.4), mlx-vlm 0.7.11, mlx-lm 0.32.2,
  transformers 5.21.0, torch 2.15.0, numpy 2.5.4. CAVEAT: at 22:29-22:30 local time a concurrent job replaced the
  venv's `mlx_vlm` / `mlx_lm` wheels with editable checkouts (`upstream/mlx-vlm` at `138d761` = v0.7.11-7-g138d761
  plus an uncommitted chunked-eval change in `mlx_vlm/utils.py`; `upstream/mlx-lm` at `1f7057c` plus an uncommitted
  `ShiftingWindowCache` in `mlx_lm/models/cache.py`). The MLX `baseline` and `kv8` rows below imported the wheel
  before the swap; `rerun`, `kv8-rerun`, `kv8h`, `kv8tq`, `smoke`, `p64k` imported the checkout. `git diff
  --name-only v0.7.11..HEAD` in that checkout touches only `utils.py` (the loader) among `generate/`, `models/inkling`,
  `models/cache.py`, `speculative/`, `kv_quant.py`, `turboquant.py`, so the Inkling generation path is the 0.7.11
  one in every row; `mlx_vlm` does not import `mlx_lm`. `~/.venvs/inkling-base` (created 22:31)
  holds the untouched 0.7.11 wheel (the stock-CLI attempt was later dropped). SECOND SWAP: by 03:27 local the
  checkout had advanced to `ee1be2540c9ec3266dabdd86d46764404a8a80e7` with nine K4 commits on the Inkling path
  (`71eab0a` bounded sliding-window KV caches, `f5f4543` window caches for the sliding layers, `4a951b5` route kernel,
  `982cc5a` strided short-conv kernel, `d0ea67c` logits_to_keep, `fc5504f`, `aca5359` quantized global KV, `ee1be25`),
  and hold 2's MLX run (section 2.8) imported that code. Every other MLX row is on the 0.7.11 generation path.
- MLX checkpoint: `mlx-community/Inkling-Small-mlx-2bit`, snapshot `f8a0af554e168077bdaf48e1fb6682fb395c807c`,
  18 safetensors shards, 82.3 GiB of parameters (`quantization: group_size 64, bits 2, recipe experts_only`),
  1470 tensors, **0 `mtp` tensors** in `model.safetensors.index.json` (config still carries `mtp_config:
  num_nextn_predict_layers 8`).
- GGUF: `unsloth/Inkling-Small-GGUF`, snapshot `1a19ef82883cb7b9c581b93c30ea252dabbf658d`,
  `UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-0000{1,2,3}-of-00003.gguf` (12.4 MiB + 2 x ~45.6 GiB = 91.2 GiB; shard 1 is
  metadata only), `mmproj-F16.gguf` 132 MiB (unused). Download complete at 02:33Z.
- llama.cpp: PR #25731 head `3582c5ee05115892e4e96520be0026c3ceb9f829` ("Fix merge conflicts", 2026-10-27),
  `build: 11983 (3582c5ee0)`, AppleClang 21, Metal (embedded library, no `xcrun metal`), ggml 0.27.0,
  binaries `upstream/llama.cpp-inkling/build/bin/`.
- Prompts (`bench/prompts/`, Moby-Dick, Inkling tokenizer, each ending in "Question: In one sentence, who is the
  narrator of the text above and what is the name of the ship? Answer:"): p_short 226, p_2k 2072, p_8k 8218,
  p_32k 32794, p_64k 65561 tokens of user text. Rendered through the checkpoint's `chat_template.jinja` with
  `reasoning_effort="none"` (-> `<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>` +
  `<|message_user|><|content_text|>...<|end_message|><|message_model|>`), which adds 13-16 tokens: 239 / 2085 /
  8231 / 32807 / 65574 prompt tokens. The same rendered bytes are sent to both engines (llama-server `/completion`
  parses the special tokens), so the token sequences are identical across engines.
- Decoding: greedy (`temperature 0`), exactly 128 new tokens (EOS ignored in both engines so every run has the same
  decode length; the position of the first EOS is recorded, it is token 16 for every direct answer), one warm-up
  (discarded) then 3 timed runs, median and [min-max] reported.

## 1. How the MLX engine was driven (`bench/baseline/bench_mlx.py`)

- `mlx_vlm.load(path, lazy=True)` then `mx.eval` over the flattened parameters in 4 GiB chunks (the one-shot
  `mx.eval(model.parameters())` of the eager load trips the Metal GPU watchdog, see section 5). Load: 0.7-0.9 s
  lazy + 12.3-13.5 s chunked eval for 82.3 GiB (6.3 GiB/s from the page cache).
- `mlx_vlm.generate.stream_generate(model, processor, prompt, max_tokens=128, temperature=0.0, top_p=1.0,
  top_k=0, prompt_cache=make_prompt_cache(...))`, default `prefill_step_size` 2048 (`--prefill-step-size 512` for
  the p_64k run). Reported timings: mlx-vlm's own `GenerationResult.prompt_tps` (= prompt tokens / time to the
  first token, which includes the first decode step) and `generation_tps` (= (n+1) / time since the first token,
  0.8 % optimistic at n=128), plus wall clock: `ttft_wall_s` (first yielded token) and `decode_tps_wall` =
  127 / (last token - first token). Peak memory: `GenerationResult.peak_memory` (= `mx.get_peak_memory()`, reset
  before each run, decimal GB) and `/usr/bin/time -l` (`maximum resident set size` and `peak memory footprint`).
- `tokenizer.stopping_criteria` replaced by a never-stop callable for the timed runs (EOS ignored).
- GPU lock: `bench/GPU_LOCK` with the PID of the process holding the model; `bench/baseline/with_gpu_lock.sh` for
  the llama.cpp commands. `--quiet-wait 600` makes each run wait (up to 10 min) for other jobs' `test-backend-ops`
  / micro-benchmark processes to finish; the processes seen at the start and end of each run are in the JSONL.

## 2. MLX (mlx-vlm 0.7.11, mlx 0.32.4), mlx-community/Inkling-Small-mlx-2bit

### 2.1 Reading the tables

- `baseline` (22:18-22:26 local): the first full pass. Its p_2k / p_8k decode numbers (12.5 / 9.0 tok/s) are NOT
  the engine: they were taken at load average 14-30 while the 3 GB/min GGUF download, two 100 %-CPU Chromium
  renderers and other jobs ran; the same prompts in `rerun` (after the download, load 15-21) decode at
  21-24 tok/s, and p_32k in the same `baseline` pass (load 13-20) decoded at 19-26 tok/s. MLX decode on this model
  is dispatch/CPU-latency bound (42 layers x ~12 kernel launches per token at batch 1), so it tracks the machine's
  CPU load; prefill (long GPU kernels) does not.
- `rerun` (22:31-22:36): p_short, p_2k, p_8k, then p_short again as a drift check, each run gated on no other
  GPU process. The second p_short block is the clean short-context number (TTFT 0.47 s, prefill 511-528 tok/s,
  decode 20.3-21.6 tok/s); the first block overlapped other jobs' cmake builds (2 of 3 runs flagged).
- The warm-up run of every process pays kernel compilation: 8.5-15.9 s "prefill" for 239 tokens.

### 2.2 Findings (MLX)

1. Stock mlx-vlm 0.7.11 (all rows of 2.6): prefill 527-564 tok/s at 0.2K-8K and 464-493 tok/s at 32K (TTFT
   0.47 / 3.8 / 14.6 / 67-71 s for 239 / 2085 / 8231 / 32807 tokens), decode 20-27 tok/s at every context, all taken
   at load1 10-30 with the GGUF download and/or other jobs running. The one stock datum from a quiet GPU is
   the 22:09 smoke run (wheel, 20-token prompt, 48 tokens): 44.9 tok/s decode, i.e. ~2x the loaded-machine decode.
   The stock quiet-machine curve for the fixed prompts was NOT measured (hold 2 ran K4's checkout, see 2.8); it is
   the first thing to measure in a 10-minute hold with `~/.venvs/inkling-base/bin/python` (0.7.11 wheel).
2. Decode is nearly flat with context on the stock engine (21-27 / 21.4 / 20.5 / 22.7 tok/s at 239 / 2085 / 8231 /
   32807 tokens, same load regime) and so is prefill: the MoE weight stream, not attention, sets the rate at 2-bit.
   The 128-token decode phase is dispatch-latency-bound (42 layers x ~12 launches per token) and is what tracks CPU
   load. K4's checkout on a quiet machine (2.8) gives 886 tok/s prefill and 37.8 tok/s decode at 32K and 48.2 tok/s
   decode at p_short; how much of that is the quiet machine and how much K4's bounded caches / kernels is exactly
   the before/after K4's package must show back-to-back in one hold.

3. Memory: 82.4 GiB of weights resident; peak 89.4 GB (decimal, mx) at 239 tokens, 94.3 at 2K, 100.4 at 8K,
   112.7 at 32K in the baseline process (which had run p_short/p_2k/p_8k first) and 102.3 GB at 32K in the clean
   process (p_32k only), both with the default 2048-token prefill chunk; `/usr/bin/time -l` "peak memory footprint" 113.8 GB
   for the baseline process (its RSS, 49 GB, does not count the Metal buffers). The growth is (a) the KV cache:
   every one of the 42 layers keeps the full history in bf16 (`CacheList(KVCache, ArraysCache)`, 172 KB per token
   = 5.6 GiB at 32K, 11.3 GiB at 64K) because the 35 sliding layers only *slice* their cache to the last
   512+L keys instead of rotating it (`mlx_vlm/models/inkling/language.py:445-452`), and (b) prefill transients
   on the 7 global layers: the relative-position bias is materialised as a dense additive mask `[B, 32, LQ, S]`
   in bf16 by a custom Metal kernel (`language.py:150-226`), copied once more for the log-scaling `tau`
   (`:470-473`), then handed to `scaled_dot_product_attention` as an explicit mask (`:475`), i.e. ~4.3 GB per
   copy at LQ=2048, S=32807, ~17 GB in flight. p_64k with the default chunk would need ~129 GiB and was not run
   that way; it is run with `--prefill-step-size 512` (section 2.4).
4. `reasoning_effort="none"` is honoured by the template (header `Thinking effort level: 0`) and the model answers
   directly on p_short/p_8k/p_32k ("The narrator is Ishmael, and the ship is the Pequod." then EOS at token 16),
   but on p_2k it still opens `<|content_thinking|>` and reasons for the whole 128-token budget (answer never
   reached; same greedy output in every run). Effort 0 is a soft signal for this checkpoint, not a hard switch.
5. Outputs are deterministic: identical first-64-token ids across the 3 timed runs of every prompt and across
   `baseline` / `rerun` / the kv variants.

### 2.3 KV-cache quantization and MTP speculative decoding

- `--kv-bits 8` (uniform, `--quantized-kv-start 0`): **silently a no-op for Inkling in mlx-vlm 0.7.11.** The
  homogeneous uniform path of `maybe_quantize_kv_cache` (`mlx_vlm/generate/common.py:235-243`) only converts
  entries that expose `to_quantized` and `offset`; Inkling's per-layer cache is `CacheList(KVCache(),
  ArraysCache(4))` (`models/inkling/language.py:992-993`), which has neither, so every layer stays bf16. Proof in
  the JSONL (`cache_types` recorded after each run): `{'CacheList(KVCache,ArraysCache)': 42}` with `state_gib`
  0.064 for 367 tokens = 2 bytes per element, identical peak memory (89.4 / 100.4 GB) and identical token ids to
  the unquantized runs. The `kv8` / `kv8-rerun` tables therefore measure nothing but run-to-run noise.
  The two code paths that DO descend into a `CacheList` are the split key/value policy (`HybridQuantKVCache`,
  `common.py:158-186`, exercised as `kv8h`: 8-bit keys / 4-bit values, uniform, group 64) and TurboQuant
  (`common.py:190-231`, exercised as `kv8tq`: `--kv-quant-scheme turboquant --kv-bits 8`); see section 2.4 for
  whether they load, what they cost and whether the output changes.
- MTP: the 2-bit checkpoint ships **no drafter**: 0 of 1470 tensors in `model.safetensors.index.json` match `mtp`
  (the config keeps `mtp_config.num_nextn_predict_layers: 8`). mlx-vlm 0.7.11 does have the machinery
  (`speculative/drafters/inkling_mtp/`, `draft-kind mtp`, and `python -m mlx_vlm.split_mtp` with an
  `InklingMTPSplitter` that selects `model.mtp.*` + `model.llm.norm.weight` from a source checkpoint, `supports_mlx_source
  = True`), so a drafter can be produced from the bf16 `thinkingmachines/Inkling-Small` weights, which are not on
  this machine (and nothing on the Hub is a ready-made Inkling-Small MTP drafter: the only drafters listed are
  DSpark/dflash ones, `RadixArk/Inkling-Small-DSpark`, `inference-optimization/dspark-inkling-small`, a different
  kind). Not run; acceptance rate / tok/s with MTP are therefore not measured here.

### 2.4 Variants: see the tables (kv8h, kv8tq, p64k with 512-token prefill chunks, smoke A/B)

(filled from `summarize.py` below)

### 2.5 The stock CLI (`python -m mlx_vlm.generate`)

`dispatch.py:1376-1384` calls `load(model, ...)` with the default `lazy=False`, which ends in
`mx.eval(model.parameters())` (`utils.py:741`): one command buffer over 82 GB, aborted by the Metal watchdog after
~20 s with `RuntimeError: [METAL] Command buffer execution failed: Caused GPU Timeout Error
(00000002:kIOGPUCommandBufferCallbackErrorTimeout)` (same mechanism as mlx-lm issue #1572). There is no CLI flag
or environment variable for a lazy or chunked load in 0.7.11. Recorded twice: with the untouched 0.7.11 wheel
(`~/.venvs/inkling-base`, `results/mlx-cli-wheel-p_short.log`) and with the `upstream/mlx-vlm` checkout whose
uncommitted `utils.py` change evaluates the parameters in 4 GiB chunks (`results/mlx-cli-p_short.log`); the
latter shows what the fix buys.

### 2.6 MLX tables (median [min-max] of the 3 timed runs; `other GPU procs` = runs during which another job's GPU process was seen)

**baseline** (wheel 0.7.11, 22:18-22:26, during the GGUF download, load1 10-30; p_2k/p_8k decode contaminated, see 2.1):

| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 254.5 [219.2-269.3] | 0.94 [0.89-1.09] | 25.48 [22.50-27.87] | 25.68 [22.68-28.09] | 89.4 | 10.4 [10.2-10.9] | n/r | 16,16,16 | ok | yes |
| p_2k | 2085 | 486.6 [441.0-515.8] | 4.30 [4.06-4.74] | 12.52 [10.05-18.83] | 12.61 [10.13-18.98] | 93.9 | 13.6 [11.3-17.5] | n/r | None,None,None | NO (n,n,n) | yes |
| p_8k | 8231 | 464.7 [461.2-472.2] | 17.77 [17.47-17.91] | 9.01 [8.23-9.04] | 9.08 [8.30-9.11] | 100.4 | 27.4 [22.2-30.3] | n/r | 16,16,16 | ok | yes |
| p_32k | 32807 | 470.6 [464.0-493.4] | 69.82 [66.71-70.93] | 22.67 [18.83-25.91] | 22.85 [18.98-26.12] | 112.7 | 14.3 [12.7-20.2] | n/r | 16,16,16 | ok | yes |

load: lazy 0.8 s + chunked eval 13.0 s, 82.3 GiB parameters

- p_short (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|><|content_text|>The user is asking for the narrator and the ship name based on the chapter titles provided, which are from Moby-Dick by Herman Melville. The narrator is Ishmael, and the ship is the Pequod (though "'`
- p_2k (rep 1): `'<|content_thinking|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:"\n\nWe need to identify the narrator of the text above (the excerpt from Moby-Dick). The text is from Chapter 1 "Loomings" and the et'`
- p_8k (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>_<|content_text|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" They want a one-sentence answer identifying the narrator and the ship name from the provided'`
- p_32k (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" The text is clearly Moby-Dick (Melville). The narrator is Ishmael ("Call'`

**rerun** (checkout, 22:31-22:36, after the download; the second p_short block is the clean short-context number):

| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 151.6 [113.6-305.9] | 1.58 [0.78-2.11] | 26.59 [21.07-27.09] | 26.80 [21.23-27.31] | 89.4 | 13.8 [12.0-20.2] | 2/3 runs | 16,16,16 | ok | yes |
| p_2k | 2085 | 549.6 [461.2-558.2] | 3.81 [3.75-4.53] | 21.38 [20.06-24.02] | 21.55 [20.22-24.21] | 94.3 | 17.8 [17.2-18.8] | 0/3 runs | None,None,None | NO (n,n,n) | yes |
| p_8k | 8231 | 563.7 [557.0-564.8] | 14.64 [14.60-14.81] | 20.50 [14.12-21.08] | 20.66 [14.23-21.24] | 100.4 | 17.7 [15.4-21.0] | 0/3 runs | 16,16,16 | ok | yes |
| p_short | 239 | 526.8 [511.2-528.0] | 0.47 [0.46-0.47] | 21.30 [20.34-21.63] | 21.47 [20.50-21.80] | 89.3 | 15.3 [15.1-15.4] | 1/3 runs | 16,16,16 | ok | yes |

load: lazy 0.7 s + chunked eval 12.3 s, 82.3 GiB parameters
process maxrss 56.1 GiB (ru_maxrss; excludes most Metal buffers)
cache after last run: {'kinds': {'CacheList(KVCache,ArraysCache)': 42}, 'offsets': [367], 'state_gib': 0.06360626220703125}

- p_short (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|><|content_text|>The user is asking for the narrator and the ship name based on the chapter titles provided, which are from Moby-Dick by Herman Melville. The narrator is Ishmael, and the ship is the Pequod (though "'`
- p_2k (rep 1): `'<|content_thinking|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:"\n\nWe need to identify the narrator of the text above (the excerpt from Moby-Dick). The text is from Chapter 1 "Loomings" and the et'`
- p_8k (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>_<|content_text|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" They want a one-sentence answer identifying the narrator and the ship name from the provided'`

**kv8** (`--kv-bits 8 --quantized-kv-start 0`, wheel, 22:26-22:29; a no-op, see 2.3; p_short rep1 and the p_8k runs overlapped `test-backend-ops`):

| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 249.2 [56.6-299.9] | 0.96 [0.80-4.25] | 10.36 [10.02-10.48] | 10.44 [10.10-10.57] | 89.3 | 24.7 [24.2-27.9] | n/r | 16,16,16 | ok | yes |
| p_8k | 8231 | 483.6 [477.7-491.6] | 17.11 [16.78-17.26] | 11.35 [8.20-15.69] | 11.44 [8.26-15.81] | 100.4 | 36.7 [30.8-45.4] | n/r | 16,16,16 | ok | yes |

load: lazy 0.9 s + chunked eval 13.5 s, 82.3 GiB parameters
process maxrss 54.3 GiB (ru_maxrss; excludes most Metal buffers)

- p_short (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|><|content_text|>The user is asking for the narrator and the ship name based on the chapter titles provided, which are from Moby-Dick by Herman Melville. The narrator is Ishmael, and the ship is the Pequod (though "'`
- p_8k (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>_<|content_text|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" They want a one-sentence answer identifying the narrator and the ship name from the provided'`

**kv8-rerun** (same flags, checkout, 22:36-22:52; stopped after p_short to hand the GPU to K1; only the cache-type proof matters):

| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 133.0 [46.3-158.8] | 1.80 [1.51-5.17] | 25.22 [14.45-26.38] | 25.41 [14.57-26.58] | 89.4 | 11.6 [10.5-15.0] | 3/3 runs | 16,16,16 | ok | yes |

load: lazy 1.0 s + chunked eval 13.9 s, 82.3 GiB parameters
cache after last run: {'kinds': {'CacheList(KVCache,ArraysCache)': 42}, 'offsets': [367], 'state_gib': 0.06360626220703125}

- p_short (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|><|content_text|>The user is asking for the narrator and the ship name based on the chapter titles provided, which are from Moby-Dick by Herman Melville. The narrator is Ishmael, and the ship is the Pequod (though "'`

### 2.7 Not run (dropped to keep the GPU queue moving)

- kv8h / kv8tq (the two KV-quant paths that do reach a `CacheList`): dropped; `--kv-bits 8` being a no-op is proven
  by the cache-type records (2.3), the cost of the hybrid/TurboQuant paths on this model is unmeasured.
- The two CLI attempts: dropped; the stock CLI's GPU-timeout crash is documented from the 22:06 smoke run
  (`results/smoke-mlx-2bit.log`, traceback in 2.5) and the chunked-load fix in the checkout is what K4 packages.
- p_64k (even with 512-token prefill chunks) and the smoke A/B (why a 20-token prompt decoded at 44.9 tok/s in the
  22:09 smoke run vs 21-27 tok/s for p_short here; the smoke ran on an idle GPU before the GGUF download and the
  other jobs started, which is the only difference on record): not run.
- Hold 2 (02:2x local onwards) runs only the clean post-download p_32k with the default 2048-token chunk
  (`mlx-p32k-clean`, section 2.8) after the llama.cpp `-c 16384` server re-runs (3.5).

### 2.8 Hold 2: p_32k on K4's modified checkout (NOT the stock baseline), default prefill chunk (03:48-03:56, load1 3.5-4.6; the two "other GPU procs" flags are this hold's own `tail` watcher and the zsh wrapper, not a competitor)

The `cache after last run` line below is the proof of which code ran: 35 `SlidingWindowKVCache` layers + 7 `KVCache`
layers and 0.97 GiB of cache state at 32935 tokens (vs 42 x `KVCache`, 5.6 GiB, on 0.7.11). Peak memory 102.3 GB
(vs 112.7 GB stock) follows from that. These are K4's numbers on a quiet machine, recorded here because they were
taken under the baseline protocol; they must not be read as the 0.7.11 engine.

| prompt | prompt tok | prefill tok/s (api) | TTFT s (wall) | decode tok/s (wall) | decode tok/s (api) | peak GB (mx) | load1 | other GPU procs | eos@ | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_32k | 32807 | 886.0 [883.5-897.1] | 37.06 [36.60-37.16] | 37.76 [35.38-38.64] | 38.05 [35.65-38.94] | 102.3 | 3.8 [3.5-3.9] | 3/3 runs | 16,16,16 | ok | yes |

load: lazy 0.6 s + chunked eval 11.2 s, 82.3 GiB parameters
process maxrss 58.5 GiB (ru_maxrss; excludes most Metal buffers)
cache after last run: {'kinds': {'InklingCacheList(SlidingWindowKVCache,ArraysCache)': 35, 'InklingCacheList(KVCache,ArraysCache)': 7}, 'offsets': [32935], 'state_gib': 0.9747886657714844}

- p_32k (rep 1): `'<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" The text is clearly Moby-Dick (Melville). The narrator is Ishmael ("Call'`


## 3. llama.cpp (Metal), unsloth/Inkling-Small-GGUF UD-IQ3_XXS, PR #25731 head, pristine build

Hold 1 of the GPU queue, 05:06-05:49Z. Build: `upstream/llama.cpp-base` = `git worktree add ../llama.cpp-base pr-25731`
(detached at `3582c5ee05115892e4e96520be0026c3ceb9f829`), `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
-DLLAMA_CURL=OFF`, targets llama-bench llama-server, 02:43-02:45Z; `llama-bench --version` = `version: 0.7.0-dev
(build 11983, commit 3582c5ee0)`, AppleClang 21.0.0, ggml 0.27.0, Metal: `has tensor = true`, `has bfloat = true`,
residency sets on. Model path: `.../snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf`
(llama-bench reports `inkling ?B IQ3_XXS - 3.0625 bpw`, 91.19 GiB, "263.66 B params" from the GGUF metadata).
Bounds, as instructed by the lead: `-fa 1` only at depths 0 and 4096 in llama-bench and only p_short / p_2k plus one
p_8k capped at 300 s on the server, because on this PR `-fa 1` runs the 7 global layers' banded attention on the CPU
(no Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED`, R2 section 1) and every deeper point would hold the GPU for tens of
minutes; `-fa 0` (the dense Metal path, what `-fa auto` resolves to for Inkling) at 0 / 4096 / 16384 / 32768 and all four
prompts. Machine state during the hold: load1 12-20 for llama-bench and the `-fa 1` server, but 20-65 (!) during the
`-fa 0` server runs (other jobs' CPU jobs), with PhysMem 126-127 GB used, 110-112 GB wired, 0.1-0.2 GB unused,
swap 4.3 GB and rising pageouts; see 3.4.

### 3.1 llama-bench (`-m <shard1> -ngl 99 -p 512 -n 128 -r 3 -o md`, default 6 threads, ubatch 512, f16 KV)

Command: `bench/baseline/run_llama_bench.sh <fa> <depths> 3`; raw: `results/llama-bench-fa{0,1}.{log,jsonl}`.

`-fa 0` (dense Metal path), 05:06-05:12Z:

| test | depth | fa | t/s avg | t/s stddev | samples |
|---|---|---|---|---|---|
| pp512 | 0 | 0 | 219.38 | 22.88 | 3 |
| tg128 | 0 | 0 | 18.49 | 2.42 | 3 |
| pp512 | 4096 | 0 | 274.67 | 8.63 | 3 |
| tg128 | 4096 | 0 | 17.99 | 0.91 | 3 |
| pp512 | 16384 | 0 | 235.25 | 7.63 | 3 |
| tg128 | 16384 | 0 | 15.99 | 2.92 | 3 |
| pp512 | 32768 | 0 | 185.19 | 3.37 | 3 |
| tg128 | 32768 | 0 | 13.13 | 0.49 | 3 |

`-fa 1` (CPU banded fallback for the global layers), 05:12-05:16Z:

| test | depth | fa | t/s avg | t/s stddev | samples |
|---|---|---|---|---|---|
| pp512 | 0 | 1 | 71.69 | 2.94 | 3 |
| tg128 | 0 | 1 | 12.59 | 1.92 | 3 |
| pp512 | 4096 | 1 | 34.32 | 0.24 | 3 |
| tg128 | 4096 | 1 | 10.97 | 1.13 | 3 |

Readings: (a) at `-fa 0` prefill is 219-275 tok/s for pp512 and drops to 185 tok/s at a 32K depth; decode is
18.5 tok/s at depth 0 and 13.1 tok/s at 32K (the dense attention over 32K keys costs ~22 ms per token on the 7 global
layers). (b) `-fa 1` is 3.1x slower on pp512 at depth 0 and 8.0x at depth 4096 (71.7 -> 34.3 tok/s: the CPU fallback's
cost grows with n_kv, R2 section 5.1/6), and 32-39 % slower on decode (12.6 / 11.0 tok/s); llama-bench's default 6
threads apply (the field ran 18). (c) The pp512 at depth 0 figure has a 10 % stddev (first test after the warm-up;
weights still paging in), the others 1-4 %.

### 3.2 llama-server `/completion` (stream, n_predict 128, temperature 0, ignore_eos, cache_prompt false, 1 warm-up + 3 runs)

Server: `llama-server -m <shard1> -np 1 --no-kv-unified -ngl 99 -c 70000 --jinja --temp 0 --chat-template-kwargs
'{"reasoning_effort":"none"}' -fa {0,1} --host 127.0.0.1 --port 18611 --no-webui -to 3600 --metrics`
(`n_ctx_slot = 70144`, `n_threads = 6`); client `bench/baseline/bench_llama_server.py`; raw
`results/llama-server-fa{0,1}.{jsonl,log,client.log}`. Both servers were healthy 6.0 s after launch (RSS 93.2-93.4
GiB: the weights were already in the page cache). `/apply-template` with `reasoning_effort: none` renders exactly the
bytes the MLX run used, and `/tokenize` gives 239 tokens for p_short, the HF count: the two engines saw identical
token sequences. `timings.prompt_per_second` / `predicted_per_second` are the server's own numbers; TTFT is the wall
clock to the first SSE chunk.

`-fa 0`, `-c 70000`, 05:16-05:41Z (CAVEAT: load1 15-65 and 0.1 GB of free memory, see 3.4; decode here is not the
engine's number, llama-bench measured 18.5 tok/s on the same binary 10 minutes earlier):

| prompt | prompt_n | prefill tok/s (server) | prompt_ms | TTFT s (wall) | decode tok/s (server) | total s | RSS GiB | load1 | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 195.2 [127.2-220.4] | 1224 [1084-1879] | 1.52 [1.35-2.22] | 4.78 [4.49-5.63] | 27.8 [24.4-29.4] | 93.1 [93.1-93.3] | 14.7 [13.2-19.4] | ok | yes |
| p_2k | 2085 | 245.8 [225.3-258.8] | 8481 [8057-9253] | 8.84 [8.32-9.55] | 4.14 [3.70-4.78] | 38.7 [35.1-43.7] | 91.6 [89.2-92.8] | 21.6 [20.6-23.1] | ok | yes |
| p_8k | 8231 | 239.5 [222.7-247.7] | 34363 [33235-36960] | 35.06 [33.69-37.34] | 2.68 [2.62-3.53] | 81.8 [73.1-82.0] | 78.6 [76.7-81.5] | 37.0 [34.1-38.5] | ok | yes |
| p_32k | 32807 | 200.6 [174.1-200.7] | 163530 [163430-188454] | 164.26 [164.05-189.12] | 3.43 [2.50-4.73] | 214.5 [190.7-225.8] | 56.3 [55.9-73.1] | 40.2 [18.2-65.0] | ok | yes |

load: {"load_s": 6.035535875002097, "rss_gib": 93.18070983886719}

template_check: {"same": true, "server_head": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>CHAPTER 1. Loomings.

CHAPTER 2. The Carpet-Bag.

CHAPTER 3. The Spouter-Inn.

CHAPTER 4. The C"}

tokenize_check: {"server": 239, "hf": 239}

prefix_summary: {"p8k_tokens": 8231, "ext_tokens": 8431, "first": {"prompt_n": 8231, "prompt_ms": 32870.512, "tokens_cached": 8246}, "repeat": {"prompt_n": 516, "prompt_ms": 2564.849, "tokens_cached": 8246}, "extended": {"prompt_n": 8431, "prompt_ms": 35499.291, "tokens_cached": 8446}}

end: {"rss_gib": 57.87812805175781}

- p_short (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_2k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator of the text is Ishmael, and the ship is the Pequod (as indicated by the chapter list and the "Epilogue" of *Moby-Dick*).The narrator is Ishmael, and the ship is the Pequod.Based on the chapter headings (e.g., "The Pequod Meets The Virg'`
- p_8k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod (mentioned in Chapter 16).The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator i'`
- p_32k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_8k (prefix_a_first): `'The narrator is Ishmael, and the ship is the Pequod.'`
- p_8k_ext (prefix_b_extended): `'The narrator is Ishmael, and the ship is the Pequod.'`

`-fa 1`, `-c 70000`, 05:41-05:49Z (load1 12-16; p_8k is a single run under the 300 s cap):

| prompt | prompt_n | prefill tok/s (server) | prompt_ms | TTFT s (wall) | decode tok/s (server) | total s | RSS GiB | load1 | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 135.5 [129.3-140.6] | 1764 [1700-1848] | 1.84 [1.78-1.95] | 13.12 [12.64-13.71] | 11.4 [11.1-11.8] | 93.5 [93.5-93.5] | 12.0 [11.7-12.3] | ok | yes |
| p_2k | 2085 | 76.6 [76.1-77.9] | 27203 [26767-27388] | 27.27 [26.84-27.46] | 14.51 [13.64-14.65] | 36.1 [35.4-36.5] | 93.9 [93.9-93.9] | 16.3 [14.3-16.3] | ok | yes |
| p_8k | 8231 | 33.1 | 248299 | 248.41 | 8.35 | 263.5 | 93.9 | 13.1 | ok | yes |

load: {"load_s": 6.022300583004835, "rss_gib": 93.35496520996094}

template_check: {"same": true, "server_head": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>CHAPTER 1. Loomings.

CHAPTER 2. The Carpet-Bag.

CHAPTER 3. The Spouter-Inn.

CHAPTER 4. The C"}

tokenize_check: {"server": 239, "hf": 239}

- p_short (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_2k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator of the text is Ishmael, and the ship is the Pequod (as indicated by the chapter list and the "Epilogue" of *Moby-Dick*).The narrator is Ishmael, and the ship is the Pequod.Based on the chapter headings (e.g., "The Pequod Meets The Virg'`
- p_8k (run_single): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_8k (prefix_a_first): `'The narrator is Ishmael, and the ship is the Pequod.'`

Readings: prefill at `-fa 0` is 195-246 tok/s and flat to 8K (240 tok/s), 201 tok/s at 32K: TTFT 1.5 / 8.8 / 35 / 164 s
for 239 / 2085 / 8231 / 32807 tokens. At `-fa 1` prefill collapses with context, 136 -> 77 -> 33 tok/s (TTFT 1.8 / 27 /
248 s), the field's "prefill collapse", reproduced on the pristine head with the fixed prompts. The `-fa 1` server
decoded at 13.1-14.5 tok/s (8 % spread) while the `-fa 0` server, on a worse-loaded machine, decoded at 2.7-4.8 tok/s;
the `-fa 1` server's footprint is ~14 GB smaller (no dense-path scratch reserve for a 70K context), which is why it
was not thrashing. The greedy outputs of `-fa 0` and `-fa 1` are identical for every prompt (first 300 characters
compared), consistent with R2's NMSE 2e-7 equivalence of the dense and banded paths. Every answer is "The narrator is
Ishmael, and the ship is the Pequod." (with `ignore_eos` the model then restarts the sentence); the special tokens are
stripped from `content` by the server. llama.cpp's IQ3_XXS answers p_2k directly, the MLX 2-bit checkpoint thinks on
it: a quantization/behaviour difference, not an engine one.

### 3.3 Prefix cache (`-fa 0`, `cache_prompt: true`, n_predict 16)

1. p_8k after the p_32k runs: `prompt_n 8231`, 32.9 s (full prefill, expected).
2. The identical p_8k again: `prompt_n 516`, 2.56 s: the server rewound to a checkpoint and re-processed the 512-token
   SWA window plus a few tokens, i.e. reuse works for an exact repeat on this head.
3. p_8k with 200 more tokens inserted before the question (8431 tokens, ~8000 shared): `prompt_n 8431`, 35.5 s,
   `tokens_cached 8446` afterwards: **no reuse of the shared prefix, the whole prompt was re-processed**, as R2 section
   8a predicts for the hybrid/recurrent memory (`pos_min` is the tail; the gate fires and no checkpoint satisfies it).
   This is the 5x-on-multi-turn item that K3's package addresses. The same three requests against the `-fa 1`
   server behave identically (`prompt_n` 516 / 516 / 8431; the extension took 261 s at the fallback's 32 tok/s).

### 3.4 Memory and the `-c 70000` caveat

- With `-c 70000 -fa 0` the server's resident footprint is the 91.2 GiB of mmap'd weights (wired through Metal
  residency sets) plus ~2 GB of f16 KV for the 7 global layers at 70144 cells plus the dense-path compute reserve,
  which scales with n_ctx x n_ubatch (R2: 6.3 GiB per global-layer chain at 32K x 512, so ~14 GiB at 70K). On this
  128 GB machine with desktop applications and other benchmark jobs that left 0.1-0.2 GB unused, 7 GB in the compressor
  and 4.3 GB of swap in use, and the server's per-token CPU work (graph build, sampling, HTTP) competed with a load
  average of 20-65: decode fell from 7.1 tok/s (warm-up) to 2.6 tok/s over 10 minutes while prefill (long GPU
  kernels) stayed at 200-250 tok/s. The RSS reported by `ps` fell from 93 to 56 GiB during those runs (pages
  re-attributed under pressure), so `ps` RSS is not a usable memory metric for this server; use the wired total.
- Hold 2 re-runs both servers at `-c 16384` for p_short/p_2k/p_8k to separate the context-size reserve from the
  engine; the tables are in section 3.5.
- No garbage output in any run (every answer correct, deterministic across runs and across `-fa 0`/`-fa 1`).

### 3.5 Hold 2: `-fa 0` and `-fa 1` servers at `-c 16384`, p_short / p_2k / p_8k (03:33-03:45, quiet machine)

Same server command with `-c 16384` (`n_ctx_slot` 16384), load1 4-6, 66-81 GB unused at start; 1 warm-up + 3 runs for
p_short and p_2k on both modes and for p_8k at `-fa 0`; p_8k at `-fa 1` as one run under a 300 s cap; no prefix test.

`-fa 0`, `-c 16384` (03:33-03:39; the first server start re-read the weights from disk, 27 s, which skews the
p_short prefill spread):

| prompt | prompt_n | prefill tok/s (server) | prompt_ms | TTFT s (wall) | decode tok/s (server) | total s | RSS GiB | load1 | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 162.0 [150.1-430.2] | 1476 [556-1592] | 1.53 [0.59-1.65] | 18.56 [18.30-18.76] | 8.4 [7.4-8.4] | 90.6 [90.6-90.6] | 3.7 [3.4-3.7] | ok | yes |
| p_2k | 2085 | 211.3 [209.4-212.5] | 9866 [9811-9956] | 9.94 [9.88-10.02] | 18.49 [17.66-18.50] | 16.7 [16.7-17.2] | 90.9 [90.9-90.9] | 5.1 [4.3-6.4] | ok | yes |
| p_8k | 8231 | 192.7 [190.5-196.6] | 42721 [41860-43203] | 42.83 [41.96-43.29] | 16.72 [16.60-17.71] | 50.4 [49.6-50.4] | 90.9 [90.9-91.0] | 5.7 [5.3-6.0] | ok | yes |

load: {"load_s": 27.072619416001544, "rss_gib": 90.51290893554688}

template_check: {"same": true, "server_head": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>CHAPTER 1. Loomings.\n\nCHAPTER 2. The Carpet-Bag.\n\nCHAPTER 3. The Spouter-Inn.\n\nCHAPTER 4. The C"}

tokenize_check: {"server": 239, "hf": 239}

end: {"rss_gib": 90.93234252929688}

- p_short (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_2k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator of the text is Ishmael, and the ship is the Pequod (as indicated by the chapter list and the "Epilogue" of *Moby-Dick*).The narrator is Ishmael, and the ship is the Pequod.Based on the chapter headings (e.g., "The Pequod Meets The Virg'`
- p_8k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod (mentioned in Chapter 16).The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator i'`

`-fa 1`, `-c 16384` (03:39-03:45):

| prompt | prompt_n | prefill tok/s (server) | prompt_ms | TTFT s (wall) | decode tok/s (server) | total s | RSS GiB | load1 | answer | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| p_short | 239 | 147.2 [143.1-147.9] | 1624 [1616-1670] | 1.72 [1.70-1.77] | 13.99 [12.36-14.25] | 10.7 [10.6-11.9] | 92.1 [92.1-92.1] | 10.6 [10.4-10.7] | ok | yes |
| p_2k | 2085 | 94.6 [93.5-94.6] | 22049 [22046-22288] | 22.11 [22.11-22.35] | 15.21 [15.10-15.27] | 30.5 [30.4-30.6] | 92.5 [92.5-92.5] | 10.6 [10.5-11.0] | ok | yes |
| p_8k | 8231 | 39.5 | 208241 | 208.37 | 8.44 | 223.3 | 92.4 | 9.6 | ok | yes |

load: {"load_s": 6.023810707993107, "rss_gib": 91.9580078125}

template_check: {"same": true, "server_head": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>CHAPTER 1. Loomings.\n\nCHAPTER 2. The Carpet-Bag.\n\nCHAPTER 3. The Spouter-Inn.\n\nCHAPTER 4. The C"}

tokenize_check: {"server": 239, "hf": 239}

end: {"rss_gib": 92.42288208007812}

- p_short (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`
- p_2k (run): `'The narrator is Ishmael, and the ship is the Pequod.The narrator of the text is Ishmael, and the ship is the Pequod (as indicated by the chapter list and the "Epilogue" of *Moby-Dick*).The narrator is Ishmael, and the ship is the Pequod.Based on the chapter headings (e.g., "The Pequod Meets The Virg'`
- p_8k (run_single): `'The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is the Pequod.The narrator is Ishmael, and the ship is'`

Readings: with the reserve reduced and the machine quiet, `-fa 0` server decode is 18.6 / 18.5 / 16.7 tok/s at 239 /
2085 / 8231 tokens, the llama-bench tg128 numbers (18.5 at depth 0, 18.0 at 4096), so the 2.7-4.8 tok/s of 3.2 was
the `-c 70000` + load artifact, not the engine. Prefill at `-fa 0` is 211 / 193 tok/s at 2K / 8K (vs 246 / 240 at
`-c 70000` on the loaded machine: the dense path's per-ubatch cost does not depend on the reserve, the 15-20 % gap is
unexplained noise between two sessions, both with 1.5-3 % spread). `-fa 1`: prefill 147 / 95 / 39.5 tok/s and decode
14.0 / 15.2 / 8.4 tok/s, confirming the CPU-fallback collapse with context (TTFT 208 s for 8K) with 1-13 % spread.


## 4. What this means (like for like only; the 2-bit MLX and 3-bit GGUF rows are not an engine comparison)

- MLX 2-bit, stock mlx-vlm 0.7.11, on the loaded machine: ~470-560 tok/s prefill and ~21-27 tok/s decode from
  0.2K to 32K context (TTFT 0.5 / 3.8 / 15 / 70 s), 89-113 GB peak for 0.2K-32K; 64K does not fit with the default
  2048-token prefill chunk; the quiet-machine stock numbers are higher (44.9 tok/s decode in the smoke) and should
  be measured first thing in the next quiet hold. Prefill is flat with context, so on MLX the user-felt lever is the MoE prefill path (expert gather and
  2-bit matmul throughput), not attention; decode at 2-bit is latency-bound (42 layers x ~12 launches per token) and
  that is what makes it so sensitive to CPU contention.
- Two zero-risk memory items in mlx-vlm's Inkling: rotate the 35 sliding layers' caches (they hold the full history,
  143 of the 172 KB per token) and build the global layers' relative bias inside the attention instead of as a dense
  `[32, LQ, S]` mask plus a scaled copy (~17 GB of transients at 32K x 2048). Together they make 64K-128K contexts fit
  and remove the chunk-size dependence of peak memory.
- `--kv-bits 8` doing nothing for this model is a correctness/UX bug worth an upstream fix (make the uniform path
  descend into `CacheList`, or warn), independent of speed. MTP speculative decoding needs a drafter split from the
  bf16 release; the 2-bit checkpoint ships none.
- llama.cpp on the pristine PR head: `-fa 0` (what `-fa auto` picks) is the usable mode on Metal today: 219-275 tok/s
  pp512 at <=16K depth, 185 at 32K, 18.5 -> 13.1 tok/s tg128 from 0 to 32K, server decode 16.7-18.6 tok/s at
  `-c 16384`. `-fa 1` reproduces the field's collapse on this machine (pp512 71.7 -> 34.3 tok/s at depth 0 -> 4096;
  server prefill 147 -> 95 -> 39.5 tok/s at 0.2K/2K/8K, TTFT 208 s for 8K) with identical greedy outputs, so the
  Metal banded kernel's target is clear: match the dense path's prefill (and its 18 tok/s decode) while removing the
  dense path's n_ctx x n_ubatch scratch, which at `-c 70000` is what pushed this 128 GB machine into swap.
- The prompt cache on the hybrid memory re-processes a whole 8K prompt when 200 tokens are appended (35.5 s at `-fa 0`,
  261 s at `-fa 1`, instead of ~1-2 s); an exact repeat is served in 2.6 s. That is the before-number for the
  prompt-cache fix.
- Measurement hygiene on a shared machine (the finding that qualifies every other number here): stock MLX decode
  moved ~2x between the idle-GPU smoke and the loaded runs, llama.cpp server decode moved 4-7x when its context
  reserve left no free memory, and an editable venv silently swapped the code under the benchmark twice. A before/after claim is only valid back-to-back in one quiet GPU-queue hold, with load1, free
  memory and the competing processes recorded next to it (the JSONL rows in `results/` do).

## 5. Incidents and caveats

- Hold timeline (GPU queue): hold 1 (llama.cpp) 05:06:40-05:54:36Z (48 min, before the 25-minute rule); hold 2
  03:33:34-03:56:11 local (22.6 min). Hold 1 ran during other jobs' CPU jobs and with the `-c 70000` reserve; its
  server decode rows are superseded by hold 2's `-c 16384` rows.
- Clean-machine gap, stock engine: none of the fixed prompts was measured on a quiet machine with mlx-vlm 0.7.11;
  hold 2's MLX run used K4's checkout (2.8). A 10-minute hold with the pristine wheel
  (`~/.venvs/inkling-base/bin/python baseline/bench_mlx.py --no-lock --tag stock-quiet --prompts
  p_short,p_2k,p_8k,p_32k --repeats 3`) would complete the stock curve; `bench_mlx.py` records the cache classes per
  run, so the code that ran is always provable from the JSONL.
- `summarize.py` regenerates every table in this file from `results/*.jsonl`; `bench/baseline/` holds the exact
  scripts, the queue wrapper (`gpu_turn.sh`) and the hold scripts (`hold_llama.sh`, `hold_mlx.sh`).
- The venv swap at 22:29-22:30 (section 0) means the MLX rows come from two installs of the same generation code;
  stated per table.
- p_64k was not run: with the default prefill chunk it would exceed physical memory (section 2.2), and the
  512-chunk variant was dropped from hold 2.
- The `-fa 0` server rows (3.2) were taken at load1 15-65 with no free memory; treat their decode column as a
  machine-state measurement, not an engine one (llama-bench 3.1 is the engine number).
