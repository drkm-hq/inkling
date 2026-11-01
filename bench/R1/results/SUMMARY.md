
#### 2-bit tier as shipped (experts q2, attention bf16, lm_head bf16) (`e2_abf16_hbf16_decode2`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.638 | 0.569 | 0.611 | 0.284 / 0.288 / 0.023 | 0.319 / 0.299 / 0.022 | 4.93 / 5.97 / 0.39 | 35.7 / 36.5 | 8.07 | 59.5 |
| 2048 | 0.665 | 0.773 | 0.781 | 0.359 / 0.381 / 0.026 | 0.452 / 0.370 / 0.023 | 5.13 / 6.00 / 0.37 | 27.8 / 28.4 | 8.21 | 58.5 |
| 8192 | 0.847 | 0.916 | 1.368 | 0.520 / 0.440 / 0.030 | 0.924 / 1.037 / 0.092 | 5.32 / 6.66 / 0.58 | 22.1 / 22.9 | 8.38 | 57.3 |
| 32768 | 1.452 | 1.643 | 2.706 | 1.066 / 0.663 / 0.122 | 1.946 / 0.911 / 0.097 | 6.90 / 9.25 / 1.31 | 12.5 / 13.0 | 9.09 | 52.8 |

head: lm_head 3.150 ms (1.65 GB), norm+lm_head+argmax 3.326 ms, embed 0.328 ms; residual dtype mlx.core.bfloat16; peak memory 68.6 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.228, route_kernel_only_ms=0.195, shared_experts_ms=0.078, routed_experts_ms=0.193, attn_no_tau_ms=0.295; ctx 2048: attn_no_kv_slice_ms=0.396, router_logits+route_kernel_ms=0.221, route_kernel_only_ms=0.194, shared_experts_ms=0.084, routed_experts_ms=0.228, attn_no_tau_ms=0.410; ctx 8192: attn_no_kv_slice_ms=0.642, router_logits+route_kernel_ms=4.084, route_kernel_only_ms=3.556, shared_experts_ms=0.395, routed_experts_ms=0.708, attn_no_tau_ms=1.320; ctx 32768: attn_no_kv_slice_ms=1.600, router_logits+route_kernel_ms=1.114, route_kernel_only_ms=0.927, shared_experts_ms=0.175, routed_experts_ms=0.541, attn_no_tau_ms=1.767; 

#### 2-bit tier as shipped, first clean run (no GPU warm-up) (`e2_abf16_hbf16`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.034 | 0.720 | 0.606 | 0.303 / 0.285 / 0.033 | 0.310 / 0.266 / 0.032 | 9.66 / 8.65 / 1.13 | 27.2 / 26.8 | 8.07 | 59.5 |
| 2048 | 0.981 | 0.742 | 0.919 | 0.983 / 0.676 / 0.066 | 0.500 / 0.457 / 0.035 | 4.15 / 6.16 / 0.41 | 25.3 / 29.1 | 8.21 | 58.5 |
| 8192 | 0.743 | 0.863 | 1.051 | 0.418 / 0.372 / 0.036 | 0.618 / 0.360 / 0.032 | 4.73 / 5.54 / 0.32 | 22.7 / 25.4 | 8.38 | 57.3 |
| 32768 | 0.946 | 1.104 | 1.781 | 0.751 / 0.390 / 0.035 | 1.459 / 0.447 / 0.047 | 4.74 / 5.43 / 0.29 | 17.4 / 19.3 | 9.09 | 52.8 |

head: lm_head 3.673 ms (1.65 GB), norm+lm_head+argmax 5.285 ms, embed 1.418 ms; residual dtype mlx.core.bfloat16; peak memory 68.6 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.251, route_kernel_only_ms=0.235, shared_experts_ms=0.088, routed_experts_ms=0.189, attn_no_tau_ms=0.319; ctx 2048: attn_no_kv_slice_ms=1.085, router_logits+route_kernel_ms=0.226, route_kernel_only_ms=0.198, shared_experts_ms=0.123, routed_experts_ms=0.252, attn_no_tau_ms=0.453; ctx 8192: attn_no_kv_slice_ms=0.532, router_logits+route_kernel_ms=0.227, route_kernel_only_ms=0.206, shared_experts_ms=0.093, routed_experts_ms=0.214, attn_no_tau_ms=0.538; ctx 32768: attn_no_kv_slice_ms=1.291, router_logits+route_kernel_ms=0.273, route_kernel_only_ms=0.323, shared_experts_ms=0.137, routed_experts_ms=0.309, attn_no_tau_ms=1.598; 

#### 4-bit tier as shipped (experts q4, attention bf16, lm_head bf16) (`e4_abf16_hbf16_decode2`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.753 | 0.745 | 1.190 | 0.354 / 0.433 / 0.055 | 0.540 / 0.838 / 0.290 | 5.20 / 7.20 / 1.07 | 25.9 / 27.1 | 10.19 | 47.1 |
| 2048 | 0.767 | 1.005 | 1.347 | 0.501 / 0.431 / 0.061 | 0.724 / 0.479 / 0.058 | 5.20 / 7.26 / 0.92 | 20.7 / 21.6 | 10.32 | 46.5 |
| 8192 | 0.913 | 1.082 | 1.543 | 0.554 / 0.462 / 0.055 | 0.799 / 0.711 / 0.143 | 5.35 / 7.48 / 1.06 | 19.0 / 19.9 | 10.50 | 45.7 |
| 32768 | 1.358 | 2.548 | 2.572 | 1.753 / 0.787 / 0.056 | 1.869 / 1.891 / 0.773 | 6.30 / 8.16 / 1.73 | 9.2 / 9.6 | 11.20 | 42.9 |

head: lm_head 3.805 ms (1.65 GB), norm+lm_head+argmax 3.818 ms, embed 0.450 ms; residual dtype mlx.core.bfloat16; peak memory 70.2 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.641, router_logits+route_kernel_ms=0.440, route_kernel_only_ms=0.325, shared_experts_ms=0.129, routed_experts_ms=0.266, attn_no_tau_ms=0.544; ctx 2048: attn_no_kv_slice_ms=0.540, mlp_gather_down_ms=0.524, router_logits+route_kernel_ms=0.428, route_kernel_only_ms=0.417, shared_experts_ms=0.115, routed_experts_ms=0.450, attn_no_tau_ms=0.482; ctx 8192: attn_no_kv_slice_ms=0.734, mlp_gather_down_ms=0.977, router_logits+route_kernel_ms=0.403, route_kernel_only_ms=0.429, shared_experts_ms=0.141, routed_experts_ms=0.467, attn_no_tau_ms=0.792; ctx 32768: attn_no_kv_slice_ms=1.461, mlp_gather_down_ms=0.831, router_logits+route_kernel_ms=0.499, route_kernel_only_ms=0.609, shared_experts_ms=0.205, routed_experts_ms=0.562, attn_no_tau_ms=2.193; 

#### 4-bit tier as shipped, first clean run (no GPU warm-up) (`e4_abf16_hbf16`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.476 | 0.516 | 0.559 | 0.245 / 0.283 / 0.026 | 0.268 / 0.287 / 0.024 | 4.41 / 4.82 / 0.27 | 39.4 / 40.4 | 10.19 | 47.1 |
| 2048 | 0.532 | 0.589 | 0.632 | 0.292 / 0.294 / 0.028 | 0.354 / 0.299 / 0.029 | 4.50 / 4.78 / 0.31 | 35.2 / 36.1 | 10.32 | 46.5 |
| 8192 | 0.643 | 0.754 | 1.133 | 0.393 / 0.372 / 0.035 | 0.670 / 0.436 / 0.032 | 4.60 / 4.87 / 0.48 | 26.6 / 27.6 | 10.50 | 45.7 |
| 32768 | 1.623 | 1.475 | 2.095 | 0.961 / 0.499 / 0.051 | 1.470 / 0.470 / 0.050 | 8.77 / 23.97 / 0.41 | 14.3 / 14.2 | 11.20 | 42.9 |

head: lm_head 2.880 ms (1.65 GB), norm+lm_head+argmax 3.218 ms, embed 0.247 ms; residual dtype mlx.core.bfloat16; peak memory 70.2 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.315, router_logits+route_kernel_ms=0.166, route_kernel_only_ms=0.172, shared_experts_ms=0.064, routed_experts_ms=0.200, attn_no_tau_ms=0.257; ctx 2048: attn_no_kv_slice_ms=0.319, mlp_gather_down_ms=0.314, router_logits+route_kernel_ms=0.194, route_kernel_only_ms=0.175, shared_experts_ms=0.070, routed_experts_ms=0.205, attn_no_tau_ms=0.318; ctx 8192: attn_no_kv_slice_ms=0.512, mlp_gather_down_ms=0.469, router_logits+route_kernel_ms=0.240, route_kernel_only_ms=0.207, shared_experts_ms=0.118, routed_experts_ms=0.330, attn_no_tau_ms=0.668; ctx 32768: attn_no_kv_slice_ms=1.514, mlp_gather_down_ms=0.478, router_logits+route_kernel_ms=0.253, route_kernel_only_ms=0.339, shared_experts_ms=0.111, routed_experts_ms=0.297, attn_no_tau_ms=1.426; 

#### hypothetical: experts q2 + attention q8 + lm_head q8 (`e2_a8_h8_decode2`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.405 | 0.461 | 0.499 | 0.180 / 0.249 / 0.022 | 0.208 / 0.251 / 0.023 | 3.28 / 3.97 / 0.68 | 43.7 / 46.6 | 5.57 | 86.2 |
| 2048 | 0.503 | 0.654 | 0.708 | 0.289 / 0.348 / 0.032 | 0.382 / 0.385 / 0.037 | 3.34 / 3.97 / 0.45 | 32.3 / 34.5 | 5.70 | 84.2 |
| 8192 | 0.706 | 0.875 | 1.143 | 0.472 / 0.426 / 0.036 | 0.691 / 0.452 / 0.036 | 4.17 / 5.00 / 0.60 | 24.0 / 25.2 | 5.88 | 81.7 |
| 32768 | 1.113 | 1.180 | 1.977 | 0.842 / 0.343 / 0.038 | 1.590 / 0.495 / 0.048 | 3.74 / 4.89 / 0.54 | 17.1 / 18.4 | 6.58 | 72.9 |

head: lm_head 2.283 ms (0.87 GB), norm+lm_head+argmax 2.765 ms, embed 0.604 ms; residual dtype mlx.core.bfloat16; peak memory 50.6 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.192, route_kernel_only_ms=0.173, shared_experts_ms=0.055, routed_experts_ms=0.140, attn_no_tau_ms=0.189; ctx 2048: attn_no_kv_slice_ms=0.304, router_logits+route_kernel_ms=0.268, route_kernel_only_ms=0.219, shared_experts_ms=0.096, routed_experts_ms=0.226, attn_no_tau_ms=0.343; ctx 8192: attn_no_kv_slice_ms=0.602, router_logits+route_kernel_ms=0.319, route_kernel_only_ms=0.282, shared_experts_ms=0.118, routed_experts_ms=0.266, attn_no_tau_ms=0.644; ctx 32768: attn_no_kv_slice_ms=1.399, router_logits+route_kernel_ms=0.382, route_kernel_only_ms=0.548, shared_experts_ms=0.135, routed_experts_ms=0.370, attn_no_tau_ms=1.493; 

#### hypothetical: experts q4 + attention q8 + lm_head q8 (no warm-up) (`e4_a8_h8`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.742 | 2.088 | 1.483 | 0.446 / 0.503 / 0.040 | 0.915 / 1.598 / 0.491 | 4.41 / 5.75 / 1.13 | 11.6 / 12.2 | 7.68 | 62.5 |
| 2048 | 0.892 | 2.243 | 0.984 | 0.399 / 0.477 / 0.051 | 0.547 / 1.250 / 0.067 | 4.41 / 4.50 / 0.75 | 11.6 / 12.0 | 7.81 | 61.4 |
| 8192 | 1.254 | 0.747 | 1.040 | 1.290 / 0.363 / 0.035 | 1.838 / 0.833 / 0.098 | 3.16 / 4.57 / 0.77 | 26.5 / 28.9 | 7.99 | 60.1 |
| 32768 | 3.388 | 1.111 | 1.678 | 1.116 / 0.412 / 0.044 | 1.510 / 0.950 / 0.069 | 3.53 / 6.86 / 0.69 | 17.1 / 19.0 | 8.69 | 55.2 |

head: lm_head 3.936 ms (0.87 GB), norm+lm_head+argmax 2.899 ms, embed 0.408 ms; residual dtype mlx.core.bfloat16; peak memory 52.2 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.466, router_logits+route_kernel_ms=0.236, route_kernel_only_ms=0.219, shared_experts_ms=0.112, routed_experts_ms=0.266, attn_no_tau_ms=0.457; ctx 2048: attn_no_kv_slice_ms=0.374, mlp_gather_down_ms=0.499, router_logits+route_kernel_ms=0.466, route_kernel_only_ms=0.265, shared_experts_ms=0.125, routed_experts_ms=0.748, attn_no_tau_ms=0.420; ctx 8192: attn_no_kv_slice_ms=0.447, mlp_gather_down_ms=1.897, router_logits+route_kernel_ms=5.341, route_kernel_only_ms=5.462, shared_experts_ms=0.659, routed_experts_ms=0.566, attn_no_tau_ms=0.667; ctx 32768: attn_no_kv_slice_ms=1.216, mlp_gather_down_ms=0.730, router_logits+route_kernel_ms=0.380, route_kernel_only_ms=0.294, shared_experts_ms=0.129, routed_experts_ms=0.337, attn_no_tau_ms=1.280; 

#### hypothetical: everything q4 (`e4_a4_h4_decode2`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.570 | 0.587 | 0.689 | 0.186 / 0.358 / 0.036 | 0.282 / 0.402 / 0.032 | 2.17 / 3.51 / 0.60 | 37.2 / 39.0 | 6.34 | 75.7 |
| 2048 | 0.671 | 0.695 | 0.840 | 0.276 / 0.441 / 0.038 | 0.368 / 0.404 / 0.049 | 2.29 / 3.72 / 0.63 | 31.6 / 33.1 | 6.48 | 74.1 |
| 8192 | 0.731 | 0.763 | 1.298 | 0.418 / 0.406 / 0.044 | 0.669 / 0.522 / 0.051 | 2.20 / 3.83 / 0.75 | 26.8 / 28.5 | 6.65 | 72.2 |
| 32768 | 1.209 | 1.447 | 2.419 | 0.992 / 0.489 / 0.055 | 1.540 / 0.582 / 0.090 | 3.39 / 5.00 / 0.93 | 14.6 / 15.3 | 7.36 | 65.2 |

head: lm_head 1.189 ms (0.46 GB), norm+lm_head+argmax 1.312 ms, embed 0.224 ms; residual dtype mlx.core.bfloat16; peak memory 51.8 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.405, router_logits+route_kernel_ms=0.317, route_kernel_only_ms=0.283, shared_experts_ms=0.099, routed_experts_ms=0.253, attn_no_tau_ms=0.223; ctx 2048: attn_no_kv_slice_ms=0.316, mlp_gather_down_ms=0.433, router_logits+route_kernel_ms=0.318, route_kernel_only_ms=0.369, shared_experts_ms=0.100, routed_experts_ms=0.289, attn_no_tau_ms=0.280; ctx 8192: attn_no_kv_slice_ms=0.551, mlp_gather_down_ms=0.607, router_logits+route_kernel_ms=0.531, route_kernel_only_ms=0.372, shared_experts_ms=0.120, routed_experts_ms=0.235, attn_no_tau_ms=0.496; ctx 32768: attn_no_kv_slice_ms=1.534, mlp_gather_down_ms=1.174, router_logits+route_kernel_ms=0.800, route_kernel_only_ms=2.175, shared_experts_ms=0.317, routed_experts_ms=0.704, attn_no_tau_ms=2.078; 

#### hypothetical: experts q2 + attention q4 + lm_head q4 (no warm-up) (`e2_a4_h4`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.676 | 0.690 | 0.784 | 0.228 / 0.439 / 0.038 | 0.357 / 2.442 / 0.276 | 3.40 / 4.20 / 0.99 | 22.7 / 32.4 | 4.23 | 113.5 |
| 2048 | 0.863 | 0.776 | 0.836 | 0.902 / 0.614 / 0.060 | 0.582 / 0.665 / 0.057 | 2.47 / 5.04 / 0.80 | 21.0 / 30.1 | 4.36 | 110.0 |
| 8192 | 0.669 | 0.882 | 3.427 | 0.479 / 1.104 / 0.073 | 0.921 / 2.754 / 0.061 | 6.15 / 3.68 / 0.75 | 14.5 / 18.0 | 4.54 | 105.8 |
| 32768 | 0.952 | 1.164 | 1.964 | 0.871 / 0.377 / 0.057 | 1.358 / 0.468 / 0.048 | 4.05 / 4.48 / 0.95 | 14.6 / 18.5 | 5.24 | 91.5 |

head: lm_head 2.970 ms (0.46 GB), norm+lm_head+argmax 3.150 ms, embed 11.207 ms; residual dtype mlx.core.bfloat16; peak memory 50.2 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.418, route_kernel_only_ms=0.385, shared_experts_ms=0.197, routed_experts_ms=0.233, attn_no_tau_ms=0.375; ctx 2048: attn_no_kv_slice_ms=0.446, router_logits+route_kernel_ms=0.347, route_kernel_only_ms=0.340, shared_experts_ms=0.168, routed_experts_ms=0.378, attn_no_tau_ms=0.504; ctx 8192: attn_no_kv_slice_ms=0.897, router_logits+route_kernel_ms=0.405, route_kernel_only_ms=0.220, shared_experts_ms=0.141, routed_experts_ms=0.280, attn_no_tau_ms=0.609; ctx 32768: attn_no_kv_slice_ms=3.300, router_logits+route_kernel_ms=0.668, route_kernel_only_ms=0.444, shared_experts_ms=0.103, routed_experts_ms=0.218, attn_no_tau_ms=2.758; 

#### FINAL PASS (contended, canary 8.5 ms): 2-bit tier (`e2_abf16_hbf16_final`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.963 | 4.845 | 2.550 | 2.080 / 2.816 / 0.104 | 2.080 / 2.256 / 0.122 | 24.42 / 20.94 / 0.53 | 5.3 / 5.1 | 8.07 | 59.5 |
| 2048 | 2.097 | 6.806 | 1.207 | 2.176 / 0.892 / 0.043 | 1.859 / 0.598 / 0.071 | 10.96 / 11.28 / 0.78 | 4.1 / 4.2 | 8.21 | 58.5 |
| 8192 | 4.739 | 2.393 | 3.792 | 1.273 / 1.267 / 0.161 | 2.106 / 2.352 / 0.394 | 16.05 / 22.35 / 1.08 | 8.3 / 8.3 | 8.38 | 57.3 |
| 32768 | 3.856 | 6.479 | 7.580 | 3.353 / 2.789 / 0.417 | 5.529 / 3.366 / 0.389 | 11.18 / 15.02 / 1.70 | 3.6 / 3.7 | 9.09 | 52.8 |

head: lm_head 4.771 ms (1.65 GB), norm+lm_head+argmax 5.362 ms, embed 0.334 ms; residual dtype mlx.core.bfloat16; peak memory 68.6 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.900, route_kernel_only_ms=0.994, shared_experts_ms=0.700, routed_experts_ms=4.510, attn_no_tau_ms=1.495; ctx 2048: attn_no_kv_slice_ms=0.847, router_logits+route_kernel_ms=0.489, route_kernel_only_ms=0.393, shared_experts_ms=0.346, routed_experts_ms=1.523, attn_no_tau_ms=0.521; ctx 8192: attn_no_kv_slice_ms=1.485, router_logits+route_kernel_ms=2.283, route_kernel_only_ms=3.045, shared_experts_ms=0.373, routed_experts_ms=1.771, attn_no_tau_ms=2.225; ctx 32768: attn_no_kv_slice_ms=5.537, router_logits+route_kernel_ms=4.877, route_kernel_only_ms=4.942, shared_experts_ms=0.520, routed_experts_ms=1.497, attn_no_tau_ms=3.379; 

#### FINAL PASS (contended, canary 9.0 ms): 4-bit tier (`e4_abf16_hbf16_final`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.814 | 0.768 | 0.831 | 0.377 / 0.421 / 0.037 | 0.466 / 0.428 / 0.045 | 4.87 / 7.62 / 1.06 | 26.5 / 28.4 | 10.19 | 47.1 |
| 2048 | 0.984 | 1.297 | 1.298 | 0.629 / 0.834 / 0.099 | 0.722 / 0.530 / 0.185 | 7.62 / 13.31 / 1.02 | 17.0 / 17.3 | 10.32 | 46.5 |
| 8192 | 0.977 | 1.251 | 1.847 | 0.985 / 0.728 / 0.109 | 0.776 / 0.615 / 0.084 | 5.51 / 7.71 / 0.99 | 16.4 / 17.4 | 10.50 | 45.7 |
| 32768 | 1.582 | 1.469 | 2.106 | 1.099 / 0.445 / 0.307 | 1.466 / 0.558 / 0.459 | 5.82 / 8.17 / 0.84 | 14.0 / 14.9 | 11.20 | 42.9 |

head: lm_head 3.532 ms (1.65 GB), norm+lm_head+argmax 4.594 ms, embed 0.379 ms; residual dtype mlx.core.bfloat16; peak memory 70.2 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.414, router_logits+route_kernel_ms=0.235, route_kernel_only_ms=0.268, shared_experts_ms=0.092, routed_experts_ms=0.234, attn_no_tau_ms=0.366; ctx 2048: attn_no_kv_slice_ms=0.928, mlp_gather_down_ms=0.457, router_logits+route_kernel_ms=0.538, route_kernel_only_ms=0.435, shared_experts_ms=0.151, routed_experts_ms=0.376, attn_no_tau_ms=0.576; ctx 8192: attn_no_kv_slice_ms=0.728, mlp_gather_down_ms=0.623, router_logits+route_kernel_ms=0.348, route_kernel_only_ms=0.307, shared_experts_ms=0.105, routed_experts_ms=0.558, attn_no_tau_ms=0.695; ctx 32768: attn_no_kv_slice_ms=1.844, mlp_gather_down_ms=0.889, router_logits+route_kernel_ms=0.254, route_kernel_only_ms=0.418, shared_experts_ms=0.099, routed_experts_ms=0.277, attn_no_tau_ms=2.773; 

#### FINAL PASS (contended, canary 8.7 ms): q2 + q8 attn + q8 head (`e2_a8_h8_final`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.641 | 0.661 | 0.799 | 0.296 / 0.510 / 0.048 | 0.460 / 0.403 / 0.069 | 3.00 / 5.08 / 0.74 | 31.8 / 33.8 | 5.57 | 86.2 |
| 2048 | 0.701 | 0.963 | 1.237 | 1.411 / 0.344 / 0.066 | 0.685 / 1.069 / 0.311 | 3.84 / 5.95 / 0.67 | 22.4 / 23.4 | 5.70 | 84.2 |
| 8192 | 0.699 | 0.926 | 1.169 | 0.905 / 0.439 / 0.040 | 0.651 / 0.393 / 0.069 | 3.30 / 5.09 / 0.82 | 23.3 / 24.6 | 5.88 | 81.7 |
| 32768 | 1.657 | 1.440 | 2.263 | 0.871 / 0.366 / 0.032 | 1.829 / 0.724 / 0.059 | 3.87 / 6.42 / 0.64 | 14.4 / 15.3 | 6.58 | 72.9 |

head: lm_head 3.895 ms (0.87 GB), norm+lm_head+argmax 2.500 ms, embed 0.245 ms; residual dtype mlx.core.bfloat16; peak memory 50.6 GB

extras (ms): ctx 0: router_logits+route_kernel_ms=0.436, route_kernel_only_ms=0.410, shared_experts_ms=0.085, routed_experts_ms=0.210, attn_no_tau_ms=0.417; ctx 2048: attn_no_kv_slice_ms=0.334, router_logits+route_kernel_ms=0.424, route_kernel_only_ms=0.452, shared_experts_ms=0.119, routed_experts_ms=0.373, attn_no_tau_ms=0.531; ctx 8192: attn_no_kv_slice_ms=1.737, router_logits+route_kernel_ms=0.418, route_kernel_only_ms=0.385, shared_experts_ms=0.121, routed_experts_ms=0.263, attn_no_tau_ms=1.080; ctx 32768: attn_no_kv_slice_ms=1.449, router_logits+route_kernel_ms=5.127, route_kernel_only_ms=1.700, shared_experts_ms=0.103, routed_experts_ms=0.360, attn_no_tau_ms=1.354; 

#### FINAL PASS (canary 5.7 ms): q4 + q8 attn + q8 head (`e4_a8_h8_final`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.772 | 0.808 | 1.467 | 0.688 / 0.452 / 0.051 | 0.547 / 0.505 / 0.525 | 4.24 / 5.13 / 0.66 | 24.3 / 25.2 | 7.68 | 62.5 |
| 2048 | 0.721 | 0.866 | 1.022 | 0.547 / 0.466 / 0.082 | 0.553 / 0.512 / 0.049 | 3.13 / 5.87 / 0.69 | 25.1 / 26.5 | 7.81 | 61.4 |
| 8192 | 0.967 | 1.533 | 1.373 | 0.929 / 0.665 / 0.126 | 0.716 / 0.700 / 0.047 | 3.54 / 10.53 / 0.78 | 15.4 / 16.2 | 7.99 | 60.1 |
| 32768 | 1.173 | 1.185 | 2.083 | 0.817 / 0.451 / 0.034 | 1.601 / 0.475 / 0.048 | 4.44 / 5.44 / 0.73 | 17.0 / 17.8 | 8.69 | 55.2 |

head: lm_head 1.997 ms (0.87 GB), norm+lm_head+argmax 2.394 ms, embed 0.348 ms; residual dtype mlx.core.bfloat16; peak memory 52.2 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.797, router_logits+route_kernel_ms=0.422, route_kernel_only_ms=0.706, shared_experts_ms=0.178, routed_experts_ms=0.335, attn_no_tau_ms=0.296; ctx 2048: attn_no_kv_slice_ms=0.444, mlp_gather_down_ms=0.444, router_logits+route_kernel_ms=0.340, route_kernel_only_ms=0.189, shared_experts_ms=0.093, routed_experts_ms=0.345, attn_no_tau_ms=0.340; ctx 8192: attn_no_kv_slice_ms=0.721, mlp_gather_down_ms=0.840, router_logits+route_kernel_ms=1.003, route_kernel_only_ms=0.396, shared_experts_ms=0.177, routed_experts_ms=0.364, attn_no_tau_ms=0.875; ctx 32768: attn_no_kv_slice_ms=1.897, mlp_gather_down_ms=1.114, router_logits+route_kernel_ms=0.363, route_kernel_only_ms=0.303, shared_experts_ms=0.143, routed_experts_ms=1.412, attn_no_tau_ms=1.987; 

#### FINAL PASS (quiet, canary 4.6 ms): all q4 (`e4_a4_h4_final`): decode, ms per call (median), M5 Max
| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.660 | 0.686 | 0.905 | 0.280 / 0.425 / 0.054 | 1.108 / 0.490 / 0.051 | 4.00 / 4.78 / 0.71 | 30.9 / 31.2 | 6.34 | 75.7 |
| 2048 | 0.710 | 0.743 | 0.768 | 0.287 / 0.442 / 0.048 | 0.424 / 0.441 / 0.051 | 4.03 / 4.85 / 1.21 | 29.9 / 30.2 | 6.48 | 74.1 |
| 8192 | 0.715 | 0.769 | 0.962 | 0.375 / 0.417 / 0.035 | 0.563 / 0.463 / 0.046 | 2.89 / 3.74 / 0.55 | 28.0 / 29.4 | 6.65 | 72.2 |
| 32768 | 1.104 | 1.178 | 1.882 | 0.752 / 0.386 / 0.041 | 1.430 / 0.471 / 0.053 | 2.73 / 4.30 / 0.56 | 17.7 / 18.9 | 7.36 | 65.2 |

head: lm_head 1.144 ms (0.46 GB), norm+lm_head+argmax 1.533 ms, embed 0.568 ms; residual dtype mlx.core.bfloat16; peak memory 51.8 GB

extras (ms): ctx 0: mlp_gather_down_ms=0.508, router_logits+route_kernel_ms=10.451, route_kernel_only_ms=9.576, shared_experts_ms=0.542, routed_experts_ms=0.762, attn_no_tau_ms=0.223; ctx 2048: attn_no_kv_slice_ms=0.328, mlp_gather_down_ms=0.498, router_logits+route_kernel_ms=0.474, route_kernel_only_ms=0.367, shared_experts_ms=0.109, routed_experts_ms=0.297, attn_no_tau_ms=0.347; ctx 8192: attn_no_kv_slice_ms=0.469, mlp_gather_down_ms=0.476, router_logits+route_kernel_ms=0.364, route_kernel_only_ms=0.302, shared_experts_ms=0.104, routed_experts_ms=0.296, attn_no_tau_ms=0.544; ctx 32768: attn_no_kv_slice_ms=1.295, mlp_gather_down_ms=0.667, router_logits+route_kernel_ms=0.435, route_kernel_only_ms=0.413, shared_experts_ms=0.127, routed_experts_ms=0.342, attn_no_tau_ms=1.491; 

#### 2-bit tier (`e2_abf16_hbf16`): prefill, ms per layer call (median)
| T | L0 (attn/mlp/sconv) | L1 (attn/mlp/sconv) | L2 (attn/mlp/sconv) | L2 attn without tau | L1 routed / shared | 42-layer ms | pred. tok/s |
|---|---|---|---|---|---|---|---|
| 512 | 14.0 (3.6/11.5/0.5) | 15.4 (2.3/12.9/0.4) | 25.5 (7.9/18.3/0.5) | 3.4 | 14.7 / 1.9 | 714 | 717 |
| 2048 | 56.4 (22.5/39.9/7.4) | 76.6 (25.1/54.6/1.2) | 79.6 (22.6/55.5/3.2) | 19.7 | 45.1 / 21.1 | 3197 | 641 |
  chunk-2048 steady state: L1 chunk2048_at_offset2048_ms=70.9, L1 chunk2048_at_offset6144_ms=63.8, L2 chunk2048_at_offset2048_ms=89.2, L2 chunk2048_at_offset6144_ms=107.6
| 8192 | 284.3 (136.8/190.6/7.5) | 361.5 (180.8/230.4/6.0) | 460.0 (227.0/199.6/6.2) | 176.8 | 120.4 / 46.9 | 15718 | 521 |

#### 4-bit tier (`e4_abf16_hbf16`): prefill, ms per layer call (median)
| T | L0 (attn/mlp/sconv) | L1 (attn/mlp/sconv) | L2 (attn/mlp/sconv) | L2 attn without tau | L1 routed / shared | 42-layer ms | pred. tok/s |
|---|---|---|---|---|---|---|---|
| 512 | 12.7 (3.3/8.8/0.6) | 24.7 (3.6/21.7/0.6) | 27.1 (4.2/22.1/0.5) | 3.6 | 20.5 / 2.9 | 1030 | 497 |
| 2048 | 54.0 (16.3/34.9/1.4) | 69.1 (14.7/52.6/1.3) | 70.4 (19.9/52.7/1.1) | 17.9 | 43.7 / 8.4 | 2882 | 711 |
  chunk-2048 steady state: L1 chunk2048_at_offset2048_ms=69.8, L1 chunk2048_at_offset6144_ms=73.4, L2 chunk2048_at_offset2048_ms=81.2, L2 chunk2048_at_offset6144_ms=102.1
| 8192 | 292.0 (138.8/176.3/5.8) | 279.2 (129.0/179.1/7.5) | 395.6 (234.0/205.5/9.3) | 147.9 | 136.0 / 36.7 | 12567 | 652 |

#### kernels_bench.json
| measurement | ms |
|---|---|
| decode SWA S=512: banded_mask_v2 [1,32,1,512] | 0.1063 writes 0.03 MB |
| decode SWA S=512: SDPA with additive mask | 0.1193 K+V read 2.10 MB |
| decode SWA S=512: SDPA mask=None | 0.0636  |
| decode SWA S=512: mask_v2 + SDPA (as in InklingAttention) | 0.0383  |
| decode global S=2048: banded_mask_v2 [1,32,1,2048] | 0.0371 writes 0.13 MB |
| decode global S=2048: SDPA with additive mask | 0.0442 K+V read 8.39 MB |
| decode global S=2048: SDPA mask=None | 0.0404  |
| decode global S=2048: mask_v2 + SDPA (as in InklingAttention) | 0.0559  |
| decode global S=2048: tau where() over the mask (L473) | 0.0276  |
| decode global S=8192: banded_mask_v2 [1,32,1,8192] | 0.0565 writes 0.52 MB |
| decode global S=8192: SDPA with additive mask | 0.0897 K+V read 33.55 MB |
| decode global S=8192: SDPA mask=None | 0.0740  |
| decode global S=8192: mask_v2 + SDPA (as in InklingAttention) | 0.1113  |
| decode global S=8192: tau where() over the mask (L473) | 0.0276  |
| decode global S=32768: banded_mask_v2 [1,32,1,32768] | 0.1253 writes 2.10 MB |
| decode global S=32768: SDPA with additive mask | 0.3150 K+V read 134.22 MB |
| decode global S=32768: SDPA mask=None | 0.2868  |
| decode global S=32768: mask_v2 + SDPA (as in InklingAttention) | 0.4704  |
| decode global S=32768: tau where() over the mask (L473) | 0.0338  |
| prefill SWA LQ=2048 S=2048: banded_mask_v2 [1,32,2048,2048] | 2.2651 writes 0.27 GB |
| prefill SWA LQ=2048 S=2048: SDPA with additive mask | 1.8570  |
| prefill SWA LQ=2048 S=2048: SDPA mask='causal' (no bias; reference) | 0.8966  |
| prefill SWA LQ=2048 S=2559: banded_mask_v2 [1,32,2048,2559] | 2.7026 writes 0.34 GB |
| prefill SWA LQ=2048 S=2559: SDPA with additive mask | 2.2344  |
| prefill global LQ=2048 S=2048: banded_mask_v2 [1,32,2048,2048] | 2.8285 writes 0.27 GB |
| prefill global LQ=2048 S=2048: SDPA with additive mask | 1.8643  |
| prefill global LQ=2048 S=2048: SDPA mask='causal' (no bias; reference) | 0.8924  |
| prefill global LQ=2048 S=2048: tau where() over the mask (L473) | 3.9455  |
| prefill global LQ=2048 S=8192: banded_mask_v2 [1,32,2048,8192] | 7.3490 writes 1.07 GB |
| prefill global LQ=2048 S=8192: SDPA with additive mask | 7.3752  |
| prefill global LQ=2048 S=8192: tau where() over the mask (L473) | 15.9781  |
| prefill global LQ=8192 S=8192: banded_mask_v2 [1,32,8192,8192] | 28.5418 writes 4.29 GB |
| prefill global LQ=8192 S=8192: SDPA with additive mask | 27.4411  |
| prefill global LQ=8192 S=8192: SDPA mask='causal' (no bias; reference) | 10.5935  |
| prefill global LQ=8192 S=8192: tau where() over the mask (L473) | 62.2408  |
| sconv decode fused kernel C=1024 (+residual) | 0.1289  |
| sconv decode UNFUSED path C=1024 (mask forces mx ops) | 0.1317  |
| sconv prefill T=2048 C=1024 (conv1d path) | 0.4463  |
| sconv decode fused kernel C=4096 (+residual) | 0.0233  |
| sconv decode UNFUSED path C=4096 (mask forces mx ops) | 0.0685  |
| sconv prefill T=2048 C=4096 (conv1d path) | 1.4904  |
| route kernel N=1 (inkling_moe_route) | 0.1418  |
| route pure-MLX N=1 (L772-793 chain) | 0.2337  |
| router gemv N=1 (xf @ gate_weight.T, bf16 258x4096) | 0.1417  |
| route kernel N=8 (inkling_moe_route) | 0.1431  |
| route pure-MLX N=8 (L772-793 chain) | 0.2020  |
| router gemv N=8 (xf @ gate_weight.T, bf16 258x4096) | 0.1553  |
| route kernel N=2048 (inkling_moe_route) | 0.1609  |
| route pure-MLX N=2048 (L772-793 chain) | 0.2612  |
| router gemv N=2048 (xf @ gate_weight.T, bf16 258x4096) | 0.2726  |
| MoE block q4 N=1: default path (down_combine kernel if q4) | 0.2821  |
| MoE block q4 N=1: gather_qmm down path (_DOWN_COMBINE=False) | 0.2992  |
| MoE block q4 N=3: default path (down_combine kernel if q4) | 0.5318  |
| MoE block q4 N=3: gather_qmm down path (_DOWN_COMBINE=False) | 0.6085  |
| MoE block q4 N=8: default path (down_combine kernel if q4) | 1.1667  |
| MoE block q4 N=8: gather_qmm down path (_DOWN_COMBINE=False) | 1.1631  |
| MoE block q4 N=2048 (prefill, sorted gather_qmm) | 30.5216  |
|   gather_qmm q4 up_proj N=2048 sorted (x gathered [12288,1,4096]) | 10.3584  |
|   gather_qmm q4 up_proj N=2048 UNsorted | 567.8464  |
|   gather_qmm q4 down_proj N=2048 sorted | 31.2766  |
|   reference bf16 GEMM [2048,4096]x[4096,2048] (1/18 of the MoE FLOPs) | 1.7426  |
|   reference q4 quantized_matmul [2048,4096]x[4096,2048] | 3.5614  |
| MoE block q2 N=1: default path (down_combine kernel if q4) | 0.3878  |
| MoE block q2 N=1: gather_qmm down path (_DOWN_COMBINE=False) | 0.3725  |
| MoE block q2 N=3: default path (down_combine kernel if q4) | 0.7109  |
| MoE block q2 N=3: gather_qmm down path (_DOWN_COMBINE=False) | 0.7486  |
| MoE block q2 N=8: default path (down_combine kernel if q4) | 1.9043  |
| MoE block q2 N=8: gather_qmm down path (_DOWN_COMBINE=False) | 2.2379  |
| MoE block q2 N=2048 (prefill, sorted gather_qmm) | 79.5036  |
|   gather_qmm q2 up_proj N=2048 sorted (x gathered [12288,1,4096]) | 21.6147  |
|   gather_qmm q2 up_proj N=2048 UNsorted | 172.6264  |
|   gather_qmm q2 down_proj N=2048 sorted | 22.1088  |
|   reference q2 quantized_matmul [2048,4096]x[4096,2048] | 3.0431  |
| tiny-shape 3-layer decode step: wall per token (async loop) ~ dispatch floor | 0.5638 same op graph as the real model, GPU work negligible |
| tiny-shape 3-layer decode step: python graph-build time per token | 0.1648  |

#### sdpa_probe.json
```json
{
 "LQ=1,S=512": {
  "array_mask": [
   0.18724999972619116,
   8.192e-06
  ],
  "array_mask_force_fused": [
   0.16825000056996942,
   8.192e-06
  ],
  "bool_mask": [
   0.18679200002225116,
   1.6386e-05
  ],
  "none": [
   0.16474999938509427,
   8.192e-06
  ],
  "unfused_reference": [
   0.27429200054029934,
   2e-09
  ],
  "max_abs_diff_vs_unfused": 0.0009765625
 },
 "LQ=1,S=8192": {
  "array_mask": [
   0.2557079969847109,
   8.192e-06
  ],
  "array_mask_force_fused": [
   0.27712500013876706,
   0.00053248
  ],
  "bool_mask": [
   0.20858299831161276,
   0.000262146
  ],
  "none": [
   0.19800000154646114,
   0.000540672
  ],
  "unfused_reference": [
   0.877291000506375,
   0.134758402
  ],
  "max_abs_diff_vs_unfused": 0.00048828125
 },
 "LQ=1,S=32768": {
  "array_mask": [
   0.5300419979903381,
   0.000540672
  ],
  "array_mask_force_fused": [
   0.5010419990867376,
   1.6384e-05
  ],
  "bool_mask": [
   0.3865409998979885,
   4e-09
  ],
  "none": [
   0.5010419990867376,
   0.000540672
  ],
  "unfused_reference": [
   2.849292002792936,
   0.26844365
  ],
  "max_abs_diff_vs_unfused": 0.00048828125
 },
 "LQ=8,S=2048": {
  "array_mask": [
   0.3357089990458917,
   0.0
  ],
  "array_mask_force_fused": [
   0.30104100005701184,
   6.5536e-05
  ],
  "bool_mask": [
   0.25962500149034895,
   0.00052429
  ],
  "none": [
   0.25933299912139773,
   0.0
  ],
  "causal_str": [
   0.34666700230445713,
   0.008388608
  ],
  "unfused_reference": [
   0.3822920007223729,
   0.0
  ],
  "max_abs_diff_vs_unfused": 0.00048828125
 },
 "LQ=512,S=512": {
  "array_mask": [
   0.3085829994233791,
   0.004194304
  ],
  "array_mask_force_fused": [
   0.30158400113577954,
   0.004194304
  ],
  "bool_mask": [
   0.33308400088571943,
   0.00838861
  ],
  "none": [
   0.2688330023374874,
   0.004194304
  ],
  "causal_str": [
   0.25837500288616866,
   0.004194304
  ],
  "unfused_reference": [
   0.4502499978116248,
   0.0
  ],
  "max_abs_diff_vs_unfused": 0.015625
 },
 "LQ=2048,S=2048": {
  "array_mask": [
   1.9264579968876205,
   0.0
  ],
  "array_mask_force_fused": [
   1.9084170016867574,
   0.0
  ],
  "bool_mask": [
   2.595999998447951,
   2e-09
  ],
  "none": [
   1.5700830008427147,
   0.0
  ],
  "causal_str": [
   0.8941659980337135,
   0.0
  ],
  "unfused_reference": [
   4.830582998692989,
   0.335544322
  ],
  "max_abs_diff_vs_unfused": 0.015625
 },
 "LQ=2048,S=2559": {
  "array_mask": [
   2.3199999995995313,
   0.0
  ],
  "array_mask_force_fused": [
   2.334042001166381,
   0.0
  ],
  "bool_mask": [
   3.3903750008903444,
   2e-09
  ],
  "none": [
   1.8263330021000002,
   0.0
  ],
  "causal_str": [
   1.2652500008698553,
   0.0
  ],
  "unfused_reference": [
   6.271666999964509,
   0.394133506
  ]
 },
 "LQ=2048,S=8192": {
  "array_mask": [
   7.845209001970943,
   0.0
  ],
  "array_mask_force_fused": [
   7.899708998593269,
   0.0
  ],
  "bool_mask": [
   10.596666998026194,
   2e-09
  ],
  "none": [
   6.504332999611506,
   0.0
  ],
  "causal_str": [
   5.9184579986322206,
   0.0
  ],
  "unfused_reference": [
   21.0640000004787,
   1.241513986
  ]
 },
 "LQ=8192,S=8192": {
  "array_mask": [
   55.03645800126833,
   0.0
  ],
  "array_mask_force_fused": [
   127.98525000107475,
   0.0
  ],
  "bool_mask": [
   111.10137500145356,
   2e-09
  ],
  "none": [
   81.76291700146976,
   0.0
  ],
  "causal_str": [
   36.85304200189421,
   0.067108864
  ]
 }
}
```

#### mtp_probe.json
```json
{
 "ctx2048": {
  "verify_N1_3layers_ms": 3.035624999029096,
  "verify_N1_plus_argmax_head_ms": 6.522958999994444,
  "verify_N3_3layers_ms": 3.1316250024246983,
  "verify_N3_plus_argmax_head_ms": 6.391625000105705,
  "verify_N5_3layers_ms": 3.5786250009550713,
  "verify_N5_plus_argmax_head_ms": 6.90625000061118,
  "verify_N8_3layers_ms": 4.384250001749024,
  "verify_N8_plus_argmax_head_ms": 8.442792001005728,
  "snapshot_3layers_ms": 0.43250000089756213,
  "restore_3layers_ms": 0.3074590022151824,
  "kv_bytes_3layers": 25165824
 },
 "ctx32768": {
  "verify_N1_3layers_ms": 2.8255840006750077,
  "verify_N1_plus_argmax_head_ms": 6.327125000098022,
  "verify_N3_3layers_ms": 3.7417499988805503,
  "verify_N3_plus_argmax_head_ms": 7.428542001434835,
  "verify_N5_3layers_ms": 4.884291000053054,
  "verify_N5_plus_argmax_head_ms": 9.098333001020364,
  "verify_N8_3layers_ms": 8.791124997514999,
  "verify_N8_plus_argmax_head_ms": 17.54116700249142,
  "snapshot_3layers_ms": 2.859125001123175,
  "restore_3layers_ms": 1.1173750026500784,
  "kv_bytes_3layers": 402653184
 },
 "draft_step_dense_layer_plus_lm_head_argmax_ms": 6.129000001237728,
 "lm_head_argmax_only_ms": 5.154250000487082
}
```

#### diag_loop.json
```json
{
 "bf16_scale": {
  "residual_dtype": "mlx.core.bfloat16",
  "layer0_out_dtype": "mlx.core.bfloat16",
  "full_model_ms": 5.238915997324511,
  "model_no_head_ms": 1.9625830027507618,
  "head_on_pre_ms": 3.49975000062841,
  "head_on_bf16_ms": 3.4864170011132956,
  "loop": {
   "async_ms": 4.384750000099302,
   "graph_build_ms": 0.5479789997480111,
   "sync_ms": 5.04285400165827
  }
 },
 "fp32_scale": {
  "residual_dtype": "mlx.core.float32",
  "layer0_out_dtype": "mlx.core.float32",
  "full_model_ms": 24.65416600171011,
  "model_no_head_ms": 8.037125000555534,
  "head_on_pre_ms": 16.296542002237402,
  "head_on_bf16_ms": 3.2344170031137764,
  "loop": {
   "async_ms": 23.717604501143796,
   "graph_build_ms": 0.9422709990758449,
   "sync_ms": 29.48110400029691
  }
 }
}
```

#### bw_probe2.json
| op | ms/op | MB | GB/s |
|---|---|---|---|
| gemv M=1 4096x4096 bf16 (16 copies, chain 32) | 0.0743 | 33.6 | 452 |
| gemv M=1 4096x4096 q8 (16 copies, chain 32) | 0.0430 | 17.8 | 415 |
| gemv M=1 4096x4096 q4 (16 copies, chain 32) | 0.0319 | 9.4 | 296 |
| gemv M=1 4096x4096 q2 (16 copies, chain 32) | 0.0221 | 5.2 | 237 |
| gemv M=1 6656x4096 bf16 (+slice) (12 copies, chain 24) | 0.1207 | 54.5 | 452 |
| gemv M=1 6656x4096 q8 (+slice) (12 copies, chain 24) | 0.0705 | 29.0 | 411 |
| gemv M=1 6656x4096 q4 (+slice) (12 copies, chain 24) | 0.0454 | 15.3 | 337 |
| gemv M=1 dense up+down 16384x4096 x2 bf16 (4 copies, chain 12) | 0.5614 | 268.4 | 478 |
| gemv M=1 dense up+down 16384x4096 x2 q8 (4 copies, chain 12) | 0.3167 | 142.6 | 450 |
| gemv M=1 dense up+down 16384x4096 x2 q4 (4 copies, chain 12) | 0.1823 | 75.5 | 414 |
| gemv M=1 dense up+down 16384x4096 x2 q2 (4 copies, chain 12) | 0.1357 | 41.9 | 309 |
| gemv M=1 lm_head 201024x4096 bf16 (chain 6) | 3.3491 | 1646.8 | 492 |
| gemv M=1 lm_head 201024x4096 q8 (chain 6) | 1.7991 | 874.9 | 486 |
| gemv M=1 lm_head 201024x4096 q4 (chain 6) | 0.9393 | 463.2 | 493 |
| gemv M=1 lm_head 201024x4096 q2 (chain 6) | 0.5331 | 257.3 | 483 |
| SwitchGLU gather_qmm q2 N=1 K=6 up+gate+down (chain 16) | 0.1687 | 47.2 | 280 |
| gather_qmm q2 down_proj only N=1 K=6 (chain 16) | 0.0631 | 15.7 | 249 |
| gather_qmm q2 up_proj only N=1 K=6 (chain 16) | 0.0687 | 15.7 | 229 |
| SwitchGLU gather_qmm q4 N=1 K=6 up+gate+down (chain 16) | 0.2564 | 84.9 | 331 |
| gather_qmm q4 down_proj only N=1 K=6 (chain 16) | 0.0951 | 28.3 | 298 |
| gather_qmm q4 up_proj only N=1 K=6 (chain 16) | 0.0891 | 28.3 | 318 |
