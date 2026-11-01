# Benchmark protocol (Apple Silicon)

Machine: MacBook Pro M5 Max, 128 GB, macOS 26.5.1. Close other heavy processes; check with
`ps aux | sort -nrk 6 | head`. One engine resident at a time (each model is 80-95 GB).

Fixed inputs (bench/prompts/): p_short (~200 tokens), p_8k, p_32k, p_64k built from the same
public-domain text so every engine tokenizes the same bytes; greedy decoding (temp 0), 128 new
tokens unless the test is TTFT-only. Reasoning effort fixed via chat-template kwargs to "none"
when a chat template is used, and the raw completion endpoint preferred for timing.

Metrics per run: prefill tok/s (prompt tokens / prompt time), TTFT (s), decode tok/s at that
depth (new tokens / generation time), peak resident memory (process RSS and, for MLX,
mx.get_peak_memory), first 64 generated tokens (to compare outputs across runs/engines).

Procedure: warm the model with the same prompt once (discarded); then 3 timed runs; report
the median and the spread. llama.cpp: `llama-bench` for pp512/tg128 at depths 0/4096/16384/
32768, plus `llama-server` /completion timings for the fixed prompts. MLX: a script that
drives the model with the same prompts and reads the generation timings.

Report format (bench/results/<engine>-<quant>-<date>.md): a table per metric with the exact
command lines and the commit/wheel versions used.
