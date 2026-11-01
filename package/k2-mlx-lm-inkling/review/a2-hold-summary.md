# A2 hold summary

mlx-lm 0.1.dev8+gcce083427 at `963f755 A2 review: exercise filter/extract/extend in the BatchGenerator test` (dirty: False), <workdir>/upstream/mlx-lm-a2/mlx_lm/__init__.py; load 2.1 + 6.7 s, 82.4 GiB, load1 at start 6.3, peak 90.1 GiB

## Teacher-forced on the mlx-vlm 0.7.11 tokens of K2's hold 2, vs the saved reference logprobs

| run | prompt tok | noise floor (mlx-vlm vs itself, chunk 2048 vs 128) | K2 hold 2: KL mean / median / max, top-1 | A2 re-run (same statistic) | A2 decode tok/s (forced loop) | argmax == fed |
|---|---|---|---|---|---|---|
| p_short_raw | 226 | 0.0207 / 0.0020 / 0.260 (step ?), top-1 95.3 %, |dlp| top-1 max 0.789 | 0.0315 / 0.0019 / 0.538 (step ?), top-1 93.0 %, |dlp| top-1 max 1.614 | 0.0446 / 0.0036 / 1.295 (step 111), top-1 94.5 %, |dlp| top-1 max 3.043 | 15.9 | 121/128 |
| p_2k_raw | 2072 | 0.0570 / 0.0073 / 1.206 (step ?), top-1 92.2 %, |dlp| top-1 max 1.563 | 0.0876 / 0.0076 / 5.134 (step ?), top-1 91.4 %, |dlp| top-1 max 5.243 | 0.0552 / 0.0085 / 1.390 (step 16), top-1 95.3 %, |dlp| top-1 max 1.379 | 21.5 | 122/128 |
| p_8k_raw | 8218 | 0.0174 / 0.0000 / 0.294 (step ?), top-1 95.3 %, |dlp| top-1 max 1.228 | 0.0178 / 0.0000 / 0.322 (step ?), top-1 96.9 %, |dlp| top-1 max 1.408 | 0.0188 / 0.0000 / 0.512 (step 8), top-1 98.4 %, |dlp| top-1 max 0.926 | 18.2 | 126/128 |

## mlx-vlm 0.7.11 re-run on its own tokens (determinism of the reference logprobs)

- p_short_raw: 0.0000 / 0.0000 / 0.000 (step 0), top-1 100.0 %, |dlp| top-1 max 0.000; argmax == fed 128/128; decode 16.9 tok/s
- p_2k_raw: 0.0000 / 0.0000 / 0.000 (step 0), top-1 100.0 %, |dlp| top-1 max 0.000; argmax == fed 128/128; decode 18.2 tok/s

## p_32k (chat template, reasoning effort none)

32807 prompt tokens, TTFT 73.4 s (447 tok/s), 17 tokens at 29.2 tok/s, peak 90.1 GiB (active 82.4), answer_ok=True, finish=stop, load1 6.2

text: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>'

## Back-to-back benchmarks in A2's hold (median of 3, after one warm-up)

| engine | prompt | prompt tok | prefill tok/s (api) | TTFT s | decode tok/s (wall) | peak GiB (mx) | load1 |
|---|---|---|---|---|---|---|---|
| mlx-lm (a2) | p_short | 239 | 193 | 1.24 | 22.6 | 83.0 | 9.1 |
| mlx-lm (a2) | p_2k | 2085 | 408 | 5.11 | 20.6 | 85.7 | 8.9 |
| mlx-lm (a2) | p_8k | 8231 | 430 | 19.16 | 21.8 | 86.4 | 6.1 |
| mlx-vlm 0.7.11 (a2) | p_short | 239 | 224 | 1.07 | 23.5 | 83.2 | 5.6 |
| mlx-vlm 0.7.11 (a2) | p_2k | 2085 | 404 | 5.16 | 22.4 | 88.2 | 4.8 |
| mlx-vlm 0.7.11 (a2) | p_8k | 8231 | 410 | 20.08 | 22.4 | 93.5 | 4.7 |
| mlx-lm attn8 (a2) | p_short | 239 | 209 | 1.15 | 27.8 | 79.9 | 4.4 |

## 8-bit-attention variant (p_short, teacher forced on the same tokens, vs the bf16-attention mlx-lm logprobs)

- p_short_raw: 0.0280 / 0.0033 / 0.478 (step 12), top-1 95.3 %, |dlp| top-1 max 0.947; decode 27.4 tok/s (forced loop); peak 79.9 GiB

## a2-cli-generate.log (tail)

```
$ mlx_lm.generate --model ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/f8a0af554e168077bdaf48e1fb6682fb395c807c/ -p 'The secret to baking a good cake is' -m 64 --temp 0  (bounded 300s)
==========
<|content_thinking|>The user is asking "The secret to baking a good cake is" which seems like they want to know the secret or key to baking a good cake, but it's phrased as a fill-in-the-blank or asking for a specific answer. However, it's somewhat ambiguous - they might be asking for general tips,
==========
Prompt: 23 tokens, 2.583 tokens-per-sec
Generation: 64 tokens, 20.415 tokens-per-sec
Peak memory: 88.680 GB
       30.28 real         4.09 user        47.99 sys
         60155363328  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             5313795  page reclaims
                 732  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                4693  voluntary context switches
              692244  involuntary context switches
        174343070208  instructions retired
        161569645905  cycles elapsed
         89392871848  peak memory footprint
[exit code 0] 2026-10-29T07:26:54Z

```

## a2-server-smoke.log (tail)

```
.completion", "model": "default_model", "created": 1793258824, "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "<|content_thinking|>The user asks: \"In one sentence, who narrates Moby-Dick?\" They want a one-sentence answer identifying the narrator of Moby Dick"}}], "usage": {"prompt_tokens": 27, "completion_tokens": 32, "total_tokens": 59, "prompt_tokens_details": {"cached_tokens": 0}}}
[server stopped] 2026-10-29T07:27:10Z
--- server stdout/stderr (tail)
<workdir>/upstream/mlx-lm-a2/mlx_lm/server.py:1882: UserWarning: mlx_lm.server is not recommended for production as it only implements basic security checks.
  warnings.warn(
2026-10-29 03:26:56,239 - INFO - Starting httpd at 127.0.0.1 on port 8181...
127.0.0.1 - - [29/Oct/2026 03:26:59] "GET /v1/models HTTP/1.1" 200 -
2026-10-29 03:27:03,528 - INFO - Prompt Cache: 0 sequences, 0.00 GB
2026-10-29 03:27:03,528 - INFO - - assistant: 0 sequences, 0.00 GB
2026-10-29 03:27:03,528 - INFO - - user: 0 sequences, 0.00 GB
2026-10-29 03:27:03,528 - INFO - - system: 0 sequences, 0.00 GB
127.0.0.1 - - [29/Oct/2026 03:27:03] "POST /v1/completions HTTP/1.1" 200 -
2026-10-29 03:27:03,608 - INFO - Prompt processing progress: 7/8
2026-10-29 03:27:03,769 - INFO - Prompt processing progress: 8/8
2026-10-29 03:27:04,471 - INFO - Prompt Cache: 2 sequences, 0.01 GB
2026-10-29 03:27:04,471 - INFO - - assistant: 2 sequences, 0.01 GB
2026-10-29 03:27:04,471 - INFO - - user: 0 sequences, 0.00 GB
2026-10-29 03:27:04,471 - INFO - - system: 0 sequences, 0.00 GB
127.0.0.1 - - [29/Oct/2026 03:27:04] "POST /v1/chat/completions HTTP/1.1" 200 -
2026-10-29 03:27:04,640 - INFO - Prompt processing progress: 26/27
2026-10-29 03:27:04,680 - INFO - Prompt processing progress: 27/27

```
