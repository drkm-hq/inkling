## Runs

- `before`: mlx_vlm 0.7.11 from `~/.venvs/inkling-base/lib/python3.12/site-packages/mlx_vlm/__init__.py` (git None), mlx 0.32.4, load1 at start 25.8, load lazy+chunked 14.7s, 82.3 GiB
- `after`: mlx_vlm 0.7.11 from `<workdir>/upstream/mlx-vlm/mlx_vlm/__init__.py` (git 355ea6f), mlx 0.32.4, load1 at start 13.3, load lazy+chunked 14.9s, 82.3 GiB

## Equivalence (greedy token ids vs `before`)

- after / p_short: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.250
- after / p_2k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125
- after / p_8k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 5 exact top-2 ties on the path, smallest non-zero margin 0.125
- after / p_32k: IDENTICAL 129 tokens (runs self-consistent: True); top-2 logit margins identical at every step; 1 exact top-2 ties on the path, smallest non-zero margin 0.125

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

## Ratios vs `before` (medians)

| run | prompt | prefill | TTFT | decode | peak GiB delta |
|---|---|---|---|---|---|
| after | p_short | 1.076x | 0.927x | 1.453x | +0.1 |
| after | p_2k | 1.204x | 0.831x | 1.349x | +0.4 |
| after | p_8k | 1.190x | 0.840x | 1.407x | -2.1 |
| after | p_32k | 1.053x | 0.950x | 1.288x | -9.6 |
