# Equivalence: mlx-lm Inkling vs mlx-vlm 0.7.11 on mlx-community/Inkling-Small-mlx-2bit

Reference: mlx_vlm 0.7.11 from `~/.venvs/inkling-base` (mlx 0.32.4); mlx-lm from `~/.venvs/inkling`. Greedy, 128 new tokens, prefill chunk 2048, wired limit 108 GiB, identical prompt token ids on both sides.

## Teacher-forced comparison (same prefix fed to both; per-step distributions compared)

| run | prompt tok | noise floor: mlx-vlm vs itself (prefill 2048 vs 128) KL mean / median / max, top-1 agree | mlx-lm vs mlx-vlm KL mean / median / max, top-1 agree | mlx-lm |dlogprob| of ref top-1 mean / max |
|---|---|---|---|---|
| p_short_raw | 226 | 0.0207 / 0.0020 / 0.260, 95.3% | 0.0315 / 0.0019 / 0.538, 93.0% | 0.087 / 1.614 |
| p_2k_raw | 2072 | 0.0570 / 0.0073 / 1.206, 92.2% | 0.0876 / 0.0076 / 5.134, 91.4% | 0.123 / 5.243 |
| p_8k_raw | 8218 | 0.0174 / 0.0000 / 0.294, 95.3% | 0.0178 / 0.0000 / 0.322, 96.9% | 0.057 / 1.408 |
| p_short_chat_none | 239 | 0.0338 / 0.0019 / 0.858, 94.5% | 0.0348 / 0.0016 / 0.660, 95.3% | 0.086 / 1.697 |
| p_short_chat_low | 241 | 0.0286 / 0.0018 / 0.533, 94.5% | 0.0392 / 0.0013 / 1.277, 96.9% | 0.081 / 1.443 |

Router audit (p_short_raw, prompt + 128 tokens, fp32 vs bf16 router logits inside mlx-lm): 193/14160 (token, layer) routing decisions differ between fp32 and bf16 router logits (1.36%); 6th/7th selection-score gap when they differ: max 0.00102, median 0.00021 (bf16 resolution near 1 is 0.0039); median gap over all decisions 0.01020

Router audit (p_2k_raw, prompt + 128 tokens, fp32 vs bf16 router logits inside mlx-lm): 1259/88000 (token, layer) routing decisions differ between fp32 and bf16 router logits (1.43%); 6th/7th selection-score gap when they differ: max 0.00129, median 0.00019 (bf16 resolution near 1 is 0.0039); median gap over all decisions 0.00922

Router audit (p_short_chat_none, prompt + 128 tokens, fp32 vs bf16 router logits inside mlx-lm): 219/14680 (token, layer) routing decisions differ between fp32 and bf16 router logits (1.49%); 6th/7th selection-score gap when they differ: max 0.00124, median 0.00020 (bf16 resolution near 1 is 0.0039); median gap over all decisions 0.00986

Router audit (p_short_chat_low, prompt + 128 tokens, fp32 vs bf16 router logits inside mlx-lm): 213/14760 (token, layer) routing decisions differ between fp32 and bf16 router logits (1.44%); 6th/7th selection-score gap when they differ: max 0.00103, median 0.00022 (bf16 resolution near 1 is 0.0039); median gap over all decisions 0.00984

## Greedy runs (first divergence; the top-2 logit gap at that step on each side)

- p_short_raw          prompt    226 tok: DIVERGE at step 0 (A 30985 vs B 730); agree on 0 tokens before
  - mlx-vlm: chose 30985 (22.1250) over 730 (21.1250), gap 1.0000
  - mlx-lm: chose 730 (21.2500) over 30985 (21.2500), gap 0.0000
- p_2k_raw             prompt   2072 tok: DIVERGE at step 13 (A 364 vs B 13); agree on 13 tokens before
  - mlx-vlm: chose 364 (23.6250) over 13 (23.2500), gap 0.3750
  - mlx-lm: chose 13 (23.3750) over 364 (23.3750), gap 0.0000
- p_8k_raw             prompt   8218 tok: DIVERGE at step 11 (A 290 vs B 13808); agree on 11 tokens before
  - mlx-vlm: chose 290 (26.8750) over 13808 (26.7500), gap 0.1250
  - mlx-lm: chose 13808 (26.8750) over 290 (26.7500), gap 0.1250
- p_short_chat_none    prompt    239 tok: DIVERGE at step 17 (A 200004 vs B 976); agree on 17 tokens before
  - mlx-vlm: chose 200004 (21.8750) over 976 (21.1250), gap 0.7500
  - mlx-lm: chose 976 (21.2500) over 200004 (21.0000), gap 0.2500
- p_short_chat_low     prompt    241 tok: DIVERGE at step 5 (A 623 vs B 6960); agree on 5 tokens before
  - mlx-vlm: chose 623 (28.6250) over 6960 (28.1250), gap 0.5000
  - mlx-lm: chose 6960 (28.2500) over 623 (27.8750), gap 0.3750

## Per-layer residual stream and routing on p_short (cache-free forward, 226 tokens)

42 layers; residual stream [tokens, hidden] after each layer (bf16 model, dumped as fp32)
| layer | max|d| / max|ref| | mean|d| / mean|ref| | tokens with a different routed expert set |
|---|---|---|---|
| embed+norm | 0.00e+00 | 0.00e+00 | - |
| 0 | 4.24e-03 | 4.65e-03 | dense |
| 1 | 4.12e-03 | 6.50e-03 | dense |
| 2 | 8.49e-03 | 1.29e-02 | 7/226 (3.1%) |
| 3 | 1.60e-02 | 1.62e-02 | 11/226 (4.9%) |
| 4 | 1.23e-02 | 1.78e-02 | 15/226 (6.6%) |
| 5 | 1.81e-02 | 2.33e-02 | 20/226 (8.8%) |
| 6 | 2.27e-02 | 3.15e-02 | 30/226 (13.3%) |
| 7 | 2.36e-02 | 3.72e-02 | 34/226 (15.0%) |
| 8 | 2.12e-02 | 4.28e-02 | 34/226 (15.0%) |
| 9 | 4.25e-02 | 5.12e-02 | 33/226 (14.6%) |
| 10 | 4.52e-02 | 5.97e-02 | 58/226 (25.7%) |
| 11 | 5.06e-02 | 8.02e-02 | 65/226 (28.8%) |
| 12 | 6.57e-02 | 1.01e-01 | 77/226 (34.1%) |
| 13 | 7.32e-02 | 1.22e-01 | 86/226 (38.1%) |
| 14 | 6.25e-02 | 1.50e-01 | 100/226 (44.2%) |
| 15 | 6.74e-02 | 1.77e-01 | 109/226 (48.2%) |
| 16 | 2.60e-02 | 1.87e-01 | 104/226 (46.0%) |
| 17 | 9.74e-02 | 2.05e-01 | 121/226 (53.5%) |
| 18 | 9.59e-02 | 2.27e-01 | 137/226 (60.6%) |
| 19 | 9.67e-02 | 2.51e-01 | 154/226 (68.1%) |
| 20 | 1.18e-01 | 2.47e-01 | 138/226 (61.1%) |
| 21 | 1.20e-01 | 2.52e-01 | 152/226 (67.3%) |
| 22 | 2.33e-01 | 2.53e-01 | 166/226 (73.5%) |
| 23 | 1.42e-01 | 2.48e-01 | 138/226 (61.1%) |
| 24 | 1.08e-01 | 2.46e-01 | 133/226 (58.8%) |
| 25 | 8.95e-02 | 2.55e-01 | 161/226 (71.2%) |
| 26 | 9.72e-02 | 2.58e-01 | 136/226 (60.2%) |
| 27 | 1.14e-01 | 2.65e-01 | 160/226 (70.8%) |
| 28 | 1.59e-01 | 2.74e-01 | 141/226 (62.4%) |
| 29 | 3.33e-01 | 2.78e-01 | 156/226 (69.0%) |
| 30 | 6.82e-01 | 2.85e-01 | 147/226 (65.0%) |
| 31 | 8.16e-01 | 2.96e-01 | 154/226 (68.1%) |
| 32 | 8.00e-01 | 3.04e-01 | 158/226 (69.9%) |
| 33 | 7.83e-01 | 3.08e-01 | 171/226 (75.7%) |
| 34 | 8.48e-01 | 3.26e-01 | 140/226 (61.9%) |
| 35 | 8.21e-01 | 3.24e-01 | 166/226 (73.5%) |
| 36 | 8.31e-01 | 3.27e-01 | 165/226 (73.0%) |
| 37 | 8.43e-01 | 3.30e-01 | 183/226 (81.0%) |
| 38 | 8.57e-01 | 3.41e-01 | 170/226 (75.2%) |
| 39 | 8.53e-01 | 3.47e-01 | 177/226 (78.3%) |
| 40 | 8.84e-01 | 3.62e-01 | 193/226 (85.4%) |
| 41 | 7.34e-01 | 4.13e-01 | 160/226 (70.8%) |

