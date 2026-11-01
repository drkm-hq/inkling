| engine | tag | load (lazy + eval) | params GiB |
|---|---|---|---|
| mlx-vlm 0.7.11 | rerun | 0.7 + 12.3 s | 82.3 |
| mlx-lm 0.1.dev1+g1f7057ca5 | mlxlm-2bit | 0.9 + 12.7 s | 82.4 |

| prompt | tokens | engine (tag) | prefill tok/s | TTFT s | decode tok/s (wall) | decode tok/s (api) | peak GiB | load1 |
|---|---|---|---|---|---|---|---|---|
| p_short | 239 | mlx-vlm 0.7.11 (rerun) | 409 (113.6-528.0) | 0.63 | 21.46 (20.3-27.1) | 21.63 | 83.2 | 15 |
| p_short | 239 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) | 301 (223.2-302.2) | 0.80 | 24.68 (16.0-25.7) | 24.87 | 83.0 | 16 |
| p_2k | 2085 | mlx-vlm 0.7.11 (rerun) | 550 (461.2-558.2) | 3.81 | 21.38 (20.1-24.0) | 21.55 | 87.7 | 18 |
| p_2k | 2085 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) | 500 (462.6-511.1) | 4.17 | 17.62 (16.1-18.9) | 17.76 | 85.6 | 24 |
| p_8k | 8231 | mlx-vlm 0.7.11 (rerun) | 564 (557.0-564.8) | 14.64 | 20.50 (14.1-21.1) | 20.66 | 93.5 | 18 |
| p_8k | 8231 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) | 582 (571.3-598.2) | 14.15 | 18.61 (16.5-21.9) | 18.76 | 86.5 | 24 |
| p_32k | 32807 | mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) | 554 (504.2-557.6) | 59.23 | 15.70 (9.9-19.8) | 15.83 | 90.1 | 30 |

mlx-vlm 0.7.11 (rerun) p_short first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|><|content_text|>The user is asking for the narrator and the ship name based on the chapter titles provided, which are from Moby-Dick by Herman Melville. The narrator is Ishmael, and the ship is the Pequod (though "'
mlx-vlm 0.7.11 (rerun) p_2k first64: '<|content_thinking|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:"\n\nWe need to identify the narrator of the text above (the excerpt from Moby-Dick). The text is from Chapter 1 "Loomings" and the et'
mlx-vlm 0.7.11 (rerun) p_8k first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>_<|content_text|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" They want a one-sentence answer identifying the narrator and the ship name from the provided'
mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) p_short first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" The text lists chapter titles from Moby-Dick (e.g., "Loomings",'
mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) p_2k first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" The text is from Moby-Dick, Chapter 1 "Loomings" lists chapter'
mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) p_8k first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod (though it is only mentioned later; based on the "Call me Ishmael" opening, the answer is Ishmael).<|end_message|><|content_model_end_sampling|>The user asks for the answer in one sentence. The narrator is Ishmael ("Call me Ishmael"). The ship'
mlx-lm 0.1.dev1+g1f7057ca5 (mlxlm-2bit) p_32k first64: '<|content_text|>The narrator is Ishmael, and the ship is the Pequod.<|end_message|><|content_model_end_sampling|>The user asks: "In one sentence, who is the narrator of the text above and what is the name of the ship? Answer:" The text is clearly Moby-Dick (Melville). The narrator is Ishmael ("Call'
