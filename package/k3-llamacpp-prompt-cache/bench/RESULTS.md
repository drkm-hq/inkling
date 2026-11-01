# K3 measurement tables (Inkling-Small UD-IQ3_XXS, Metal, M5 Max 128 GB)

Provenance. The `commit` field in the three `meta` rows says `5edd1f152` for every run because the driver recorded
the worktree HEAD, not the binary. The binaries actually measured (from each `*-server.log` header):
`before-*` = `build-k3-baseline`, `build 11983 (3582c5ee0)`, the unpatched PR head; `after-ub512` = `build-k3`,
`build 11984 (7251878de)`, the pre-amend commit of the fix, whose server code is byte-identical to the final
commit `9b8b6bd93` (the amends changed only the code comment, the commit message and the Python test). The driver
now records `binary_build` from `llama-server --version`.

Load. `load1` at the start of each run: before-ub512 6.5, after-ub512 17.9, before-ub256 26.7 (other jobs
on the same machine). The token counts (`prompt_n` / `cache_n`) are deterministic and are the result; the times
carry about +-15% noise (the uncached p_8k reference took 29.9 s in the before run and 34.5 s in the more loaded
after run) and the after column is the handicapped one.

- `before-ub512`: worktree `5edd1f152`, binary `see the server log`, -ub 512, n_predict 64, load1 at start 6.5, model load 36.1 s, rss 91.8 GiB, p_8k = 8218 tokens, extension = 200 tokens, p_2k = 2072 tokens
- `after-ub512`: worktree `5edd1f152`, binary `see the server log`, -ub 512, n_predict 64, load1 at start 17.9, model load 12.1 s, rss 93.4 GiB, p_8k = 8218 tokens, extension = 200 tokens, p_2k = 2072 tokens
- `before-ub256`: worktree `5edd1f152`, binary `see the server log`, -ub 256, n_predict 64, load1 at start 26.7, model load 6.0 s, rss 93.3 GiB, p_8k = 8218 tokens, extension = 200 tokens, p_2k = 2072 tokens

| request | before-ub512 prompt_n / cache_n / prompt_ms / TTFT | after-ub512 prompt_n / cache_n / prompt_ms / TTFT | before-ub256 prompt_n / cache_n / prompt_ms / TTFT |
|---|---|---|---|
| p_8k, no cache (reference) | 8218 / 0 / 29.86 s / 29.888 s | 8218 / 0 / 34.47 s / 34.495 s | - |
| p_8k, cache on, slot held p_8k+200+gen | 8218 / 0 / 29.49 s / 29.497 s | 316 / 7902 / 1.83 s / 1.862 s | 8218 / 0 / 46.51 s / 46.519 s |
| (a) p_8k + 200 tokens, cache on | 716 / 7702 / 3.60 s / 3.608 s | 204 / 8214 / 1.43 s / 1.432 s | 8418 / 0 / 42.30 s / 42.307 s |
| (a) same, no cache | 8418 / 0 / 28.32 s / 28.334 s | 8418 / 0 / 33.79 s / 33.792 s | - |
| (a2) p_8k + generated + 200, cache on | 779 / 7702 / 4.27 s / 4.274 s | 200 / 8281 / 1.42 s / 1.426 s | 8481 / 0 / 35.37 s / 35.379 s |
| (a2) same, no cache | 8481 / 0 / 32.79 s / 32.83 s | 8481 / 0 / 31.73 s / 31.736 s | - |
| (c) p_8k, last 100 tokens replaced, cache on | 8218 / 0 / 30.25 s / 30.255 s | 253 / 7965 / 1.90 s / 1.909 s | 8218 / 0 / 35.14 s / 35.151 s |
| (c) same, no cache | 8218 / 0 / 29.97 s / 29.979 s | 8218 / 0 / 31.25 s / 31.255 s | - |

| greedy tokens, cached vs uncached | before-ub512 | after-ub512 | before-ub256 |
|---|---|---|---|
| B1_cached_vs_B1_ref | differ at token 19 of 64/64 | differ at token 5 of 64/64 | - |
| B2_cached_vs_B2_ref | differ at token 17 of 64/64 | differ at token 9 of 64/64 | - |
| C_cached_vs_C_ref | identical (64 tokens) | identical (64 tokens) | - |

| chat turn | before-ub512 prompt_n / cache_n / prompt_ms | after-ub512 prompt_n / cache_n / prompt_ms | before-ub256 prompt_n / cache_n / prompt_ms |
|---|---|---|---|
| turn 1 | 2085 / 0 / 9.05 s | 2085 / 0 / 9.24 s | 2085 / 0 / 8.28 s |
| turn 2 | 550 / 1569 / 3.10 s | 18 / 2101 / 0.83 s | 2110 / 9 / 8.38 s |
| turn 3 | 592 / 1603 / 3.49 s | 23 / 2167 / 0.86 s | 2163 / 9 / 8.60 s |
| turn 4 | 574 / 1679 / 3.40 s | 16 / 2233 / 0.76 s | 2217 / 9 / 8.71 s |
