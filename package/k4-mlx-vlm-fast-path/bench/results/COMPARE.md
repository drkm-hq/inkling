## Runs

- `before`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 25.8, load lazy+chunked 14.7s, 82.3 GiB
- `after`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git 355ea6f), mlx 0.32.4, load1 at start 13.3, load lazy+chunked 14.9s, 82.3 GiB
- `after-window-only`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git 0abf8bd), mlx 0.32.4, load1 at start 21.5, load lazy+chunked 13.3s, 82.3 GiB
- `after-kernels`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git 21e690a), mlx 0.32.4, load1 at start 13.3, load lazy+chunked 14.7s, 82.3 GiB
- `after-64k`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git 355ea6f), mlx 0.32.4, load1 at start 7.7, load lazy+chunked 14.1s, 82.3 GiB
- `ab-before-1`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 20.8, load lazy+chunked 13.2s, 82.3 GiB
- `ab-after-1`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git eb93183), mlx 0.32.4, load1 at start 12.1, load lazy+chunked 12.5s, 82.3 GiB
- `ab-before-2`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 5.6, load lazy+chunked 11.5s, 82.3 GiB
- `ab-after-2`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git eb93183), mlx 0.32.4, load1 at start 7.2, load lazy+chunked 17.1s, 82.3 GiB
- `after-kv8`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git eb93183), mlx 0.32.4, load1 at start 5.8, load lazy+chunked 11.3s, 82.3 GiB
- `tf-before-2048`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 8.0, load lazy+chunked 12.0s, 82.3 GiB
- `tf-after`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git eb93183), mlx 0.32.4, load1 at start 12.1, load lazy+chunked 15.4s, 82.3 GiB
- `tf-before-512`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 13.2, load lazy+chunked 12.3s, 82.3 GiB
- `tf-after-kv8`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git eb93183), mlx 0.32.4, load1 at start 8.4, load lazy+chunked 12.9s, 82.3 GiB

## Equivalence (greedy token ids vs `before`)

- after / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- after / p_2k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after / p_8k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 5 exact top-2 ties on the path, smallest non-zero margin 0.125
- after / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after-window-only / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- after-window-only / p_2k: no run
- after-window-only / p_8k: no run
- after-window-only / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after-kernels / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- after-kernels / p_2k: no run
- after-kernels / p_8k: no run
- after-kernels / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after-64k / p_short: no run
- after-64k / p_2k: no run
- after-64k / p_8k: no run
- after-64k / p_32k: no run
- ab-before-1 / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- ab-before-1 / p_2k: no run
- ab-before-1 / p_8k: no run
- ab-before-1 / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- ab-after-1 / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- ab-after-1 / p_2k: no run
- ab-after-1 / p_8k: no run
- ab-after-1 / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- ab-before-2 / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- ab-before-2 / p_2k: no run
- ab-before-2 / p_8k: no run
- ab-before-2 / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- ab-after-2 / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- ab-after-2 / p_2k: no run
- ab-after-2 / p_8k: no run
- ab-after-2 / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after-kv8 / p_short: DIVERGES at token 17: 200004 vs 976; top-2 margin before 0.75 / after 0.5; identical prefix 17 tokens
- after-kv8 / p_2k: no run
- after-kv8 / p_8k: DIVERGES at token 17: 62 vs 976; top-2 margin before 0.4375 / after 0.8125; identical prefix 17 tokens
- after-kv8 / p_32k: no run

## Teacher-forced (same 128 tokens fed to every run; KL(reference || run) in nats)

| run | prompt | chunk | reference | identical logprobs | KL mean | KL max (step) | mean abs dlogprob of teacher token | max | top-1 agree |
|---|---|---|---|---|---|---|---|---|---|
| tf-before-2048 | p_short | 2048 | (reference) | md5 2664952957e7 | | | | | teacher top-1 123/128 |
| tf-before-2048 | p_2k | 2048 | (reference) | md5 c047282ba369 | | | | | teacher top-1 125/128 |
| tf-before-2048 | p_8k | 2048 | (reference) | md5 b1fa11abdb9a | | | | | teacher top-1 124/128 |
| tf-before-2048 | p_32k | 2048 | (reference) | md5 c5b786c90196 | | | | | teacher top-1 128/128 |
| tf-after | p_short | 2048 | tf-before-2048 | True | 0.00e+00 | 0.00e+00 (0) | 0.00e+00 | 0.00e+00 | 128/128 |
| tf-after | p_2k | 2048 | tf-before-2048 | True | 0.00e+00 | 0.00e+00 (0) | 0.00e+00 | 0.00e+00 | 128/128 |
| tf-after | p_8k | 2048 | tf-before-2048 | True | 0.00e+00 | 0.00e+00 (0) | 0.00e+00 | 0.00e+00 | 128/128 |
| tf-after | p_32k | 2048 | tf-before-2048 | True | 0.00e+00 | 0.00e+00 (0) | 0.00e+00 | 0.00e+00 | 128/128 |
| tf-before-512 | p_short | 512 | tf-before-2048 | True | 0.00e+00 | 0.00e+00 (0) | 0.00e+00 | 0.00e+00 | 128/128 |
| tf-before-512 | p_2k | 512 | tf-before-2048 | False | 3.26e-02 | 1.62e+00 (120) | 8.09e-02 | 4.83e+00 | 124/128 |
| tf-before-512 | p_8k | 512 | tf-before-2048 | False | 1.84e-02 | 2.64e-01 (18) | 4.52e-02 | 7.70e-01 | 123/128 |
| tf-before-512 | p_32k | 512 | tf-before-2048 | False | 1.30e-02 | 2.95e-01 (17) | 4.53e-02 | 9.54e-01 | 122/128 |
| tf-after-kv8 | p_short | 2048 | tf-before-2048 | False | 1.64e-02 | 3.50e-01 (56) | 4.25e-02 | 4.72e-01 | 126/128 |
| tf-after-kv8 | p_2k | 2048 | tf-before-2048 | False | 1.51e-02 | 3.59e-01 (50) | 4.74e-02 | 5.19e-01 | 126/128 |
| tf-after-kv8 | p_8k | 2048 | tf-before-2048 | False | 2.73e-02 | 7.27e-01 (17) | 6.72e-02 | 1.66e+00 | 123/128 |
| tf-after-kv8 | p_32k | 2048 | tf-before-2048 | False | 6.58e-03 | 1.47e-01 (17) | 2.31e-02 | 4.45e-01 | 128/128 |

## Performance (median of timed runs, min-max in parentheses)

| run | prompt | prompt tok | prefill tok/s | TTFT s | decode tok/s | peak GiB (mx) | total s |
|---|---|---|---|---|---|---|---|
| before | p_short | 239 | 284.1 (215.3-409.1) | 0.85 (0.59-1.11) | 18.43 (16.58-24.31) | 83.2 | 7.7 (6.3-8.3) |
| before | p_2k | 2085 | 470.3 (440.5-476.6) | 4.45 (4.39-4.74) | 17.70 (16.23-18.11) | 87.4 | 11.6 (11.5-12.6) |
| before | p_8k | 8231 | 509.2 (453.8-525.1) | 16.21 (15.73-18.18) | 14.93 (13.33-19.35) | 93.5 | 24.7 (24.7-25.3) |
| before | p_32k | 32807 | 459.4 (414.1-489.0) | 71.52 (67.22-79.53) | 17.82 (10.74-19.95) | 104.9 | 77.9 (74.4-91.4) |
| after | p_short | 239 | 305.9 (231.9-339.2) | 0.78 (0.71-1.03) | 26.78 (25.97-28.09) | 83.3 | 5.6 (5.3-5.8) |
| after | p_2k | 2085 | 566.0 (473.2-567.7) | 3.70 (3.68-4.42) | 23.87 (23.86-25.78) | 87.8 | 9.0 (8.6-9.7) |
| after | p_8k | 8231 | 606.2 (601.0-613.7) | 13.61 (13.45-13.73) | 21.00 (18.94-23.27) | 91.5 | 19.8 (19.1-20.2) |
| after | p_32k | 32807 | 483.7 (433.6-496.9) | 67.94 (66.14-75.74) | 22.95 (21.72-24.86) | 95.3 | 73.5 (71.3-81.6) |
| after-window-only | p_short | 239 | 248.5 (223.0-273.9) | 0.98 (0.88-1.08) | 26.49 (26.16-26.82) | 83.3 | 5.8 (5.7-5.8) |
| after-window-only | p_32k | 32807 | 442.8 (429.8-455.9) | 74.23 (72.04-76.43) | 18.41 (15.70-21.13) | 101.1 | 81.3 (80.1-82.4) |
| after-kernels | p_short | 239 | 266.6 (254.7-278.4) | 0.90 (0.86-0.94) | 24.74 (23.52-25.95) | 83.3 | 6.1 (5.8-6.3) |
| after-kernels | p_32k | 32807 | 509.5 (490.3-528.6) | 64.62 (62.20-67.03) | 18.49 (17.77-19.22) | 95.3 | 71.5 (69.4-73.6) |
| after-64k | p_64k | 65574 | 472.1 (449.2-494.9) | 139.49 (132.87-146.11) | 13.59 (13.47-13.71) | 100.4 | 148.8 (142.3-155.4) |
| ab-before-1 | p_short | 239 | 251.5 (232.3-270.7) | 0.96 (0.89-1.03) | 20.73 (19.57-21.89) | 83.2 | 7.1 (6.8-7.4) |
| ab-before-1 | p_32k | 32807 | 455.7 (449.9-461.5) | 72.17 (71.27-73.08) | 17.40 (15.90-18.90) | 104.9 | 79.5 (78.0-81.1) |
| ab-after-1 | p_short | 239 | 244.2 (210.6-277.8) | 1.00 (0.86-1.14) | 26.58 (25.90-27.26) | 83.3 | 5.8 (5.8-5.8) |
| ab-after-1 | p_32k | 32807 | 497.9 (467.8-528.0) | 66.25 (62.26-70.24) | 24.93 (24.65-25.21) | 95.3 | 71.4 (67.3-75.4) |
| ab-before-2 | p_short | 239 | 180.9 (180.7-181.2) | 1.32 (1.32-1.33) | 22.12 (21.92-22.32) | 83.2 | 7.1 (7.0-7.1) |
| ab-before-2 | p_32k | 32807 | 245.5 (219.3-271.7) | 135.29 (120.80-149.78) | 10.40 (9.29-11.51) | 105.0 | 147.7 (131.8-163.5) |
| ab-after-2 | p_short | 239 | 125.0 (122.4-127.7) | 1.92 (1.88-1.96) | 13.74 (13.09-14.40) | 83.3 | 11.2 (10.7-11.7) |
| ab-after-2 | p_32k | 32807 | 242.2 (213.1-271.2) | 137.66 (121.21-154.10) | 13.72 (10.36-17.08) | 95.3 | 147.5 (128.6-166.4) |
| after-kv8 | p_short | 239 | 199.1 (178.4-219.9) | 1.21 (1.09-1.34) | 25.15 (24.65-25.66) | 83.3 | 6.3 (6.0-6.5) |
| after-kv8 | p_8k | 8231 | 433.9 (427.3-440.5) | 18.99 (18.70-19.27) | 24.26 (24.05-24.46) | 92.1 | 24.2 (24.0-24.5) |

## Ratios vs `before` (medians)

| run | prompt | prefill | TTFT | decode | peak GiB delta |
|---|---|---|---|---|---|
| after | p_short | 1.076x | 0.927x | 1.453x | +0.1 |
| after | p_2k | 1.204x | 0.831x | 1.349x | +0.4 |
| after | p_8k | 1.190x | 0.840x | 1.407x | -2.1 |
| after | p_32k | 1.053x | 0.950x | 1.288x | -9.6 |
| after-window-only | p_short | 0.874x | 1.155x | 1.437x | +0.1 |
| after-window-only | p_32k | 0.964x | 1.038x | 1.033x | -3.8 |
| after-kernels | p_short | 0.938x | 1.069x | 1.342x | +0.1 |
| after-kernels | p_32k | 1.109x | 0.903x | 1.038x | -9.6 |
| ab-before-1 | p_short | 0.885x | 1.136x | 1.125x | -0.0 |
| ab-before-1 | p_32k | 0.992x | 1.009x | 0.976x | +0.0 |
| ab-after-1 | p_short | 0.859x | 1.184x | 1.442x | +0.1 |
| ab-after-1 | p_32k | 1.084x | 0.926x | 1.399x | -9.6 |
| ab-before-2 | p_short | 0.637x | 1.565x | 1.200x | +0.0 |
| ab-before-2 | p_32k | 0.534x | 1.892x | 0.584x | +0.1 |
| ab-after-2 | p_short | 0.440x | 2.267x | 0.746x | +0.1 |
| ab-after-2 | p_32k | 0.527x | 1.925x | 0.770x | -9.6 |
| after-kv8 | p_short | 0.701x | 1.436x | 1.365x | +0.1 |
| after-kv8 | p_8k | 0.852x | 1.172x | 1.625x | -1.4 |
