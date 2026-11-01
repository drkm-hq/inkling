| engine | tag | load (lazy + eval) | params GiB |
|---|---|---|---|
| mlx-vlm 0.7.11 | K2-same-hold | 0.8 + 11.3 s | 82.3 |
| mlx-lm 0.1.dev1+g1f7057ca5 | mlxlm-2bit-same-hold | 1.1 + 10.9 s | 82.4 |

| prompt | tokens | engine (tag) | prefill tok/s | TTFT s | decode tok/s (wall) | decode tok/s (api) | peak GiB | load1 |
|---|---|---|---|---|---|---|---|---|
| p_short | 239 | mlx-vlm 0.7.11 (K2-same-hold) | 148 (137.0-184.7) | 1.62 | 45.57 (43.1-46.1) | 45.93 | 83.2 | 8 |
| p_short | 239 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit-same-hold) | 373 (321.3-402.0) | 0.64 | 35.31 (30.7-35.7) | 35.58 | 83.0 | 8 |
| p_2k | 2085 | mlx-vlm 0.7.11 (K2-same-hold) | 700 (687.4-799.9) | 2.98 | 40.58 (39.4-41.3) | 40.90 | 87.8 | 8 |
| p_2k | 2085 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit-same-hold) | 568 (530.5-584.5) | 3.67 | 32.01 (31.1-32.8) | 32.27 | 85.6 | 13 |
| p_8k | 8231 | mlx-vlm 0.7.11 (K2-same-hold) | 758 (747.9-759.9) | 10.87 | 38.15 (36.2-38.7) | 38.45 | 93.5 | 7 |
