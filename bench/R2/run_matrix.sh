#!/bin/bash
# Runs the fa_bench matrix at Inkling-Small shapes (32 q heads, 8 kv heads, d=128, rel extent 1024, SWA 512).
# Output: one RESULT line per case in results/fa_bench_matrix.log (plus Metal init noise filtered out).
set -u
B=~/code/inkling-contrib/bench/R2/fa_bench
OUT=~/code/inkling-contrib/bench/R2/results/fa_bench_matrix.log
export GGML_METAL_NDEBUG=1
: > "$OUT"
run() { "$B" "$@" 2>&1 | grep "^RESULT" >> "$OUT"; }

echo "# started $(date -u +%FT%TZ) on $(sysctl -n machdep.cpu.brand_string)" >> "$OUT"
# --- Metal: existing flash attention (no bias) = ceiling for a banded Metal kernel
for kv in 2048 8192 32768; do run MTL0 fa 512 $kv --reps 8; done
for kv in 2048 8192 32768; do run MTL0 fa 1   $kv --reps 20; done
run MTL0 fa 512 1024 --swa 512 --reps 8      # SWA layer, prefill: cache view = n_swa + ubatch
run MTL0 fa 1   768  --swa 512 --reps 20     # SWA layer, decode: cache view padded to 256
# --- Metal: the -fa off (dense) path that llama.cpp uses when FA is auto-disabled
for kv in 2048 8192 32768; do run MTL0 qk 512 $kv --reps 5; done
for kv in 2048 8192 32768; do run MTL0 bias 512 $kv --reps 5; done
for kv in 2048 8192 32768; do run MTL0 dense-nobias 512 $kv --reps 5; done
for kv in 2048 8192 32768; do run MTL0 dense 512 $kv --reps 5; done
for kv in 2048 8192 32768; do run MTL0 dense 1 $kv --reps 20; done
run MTL0 dense 512 1024 --swa 512 --reps 8
run MTL0 dense 1   768  --swa 512 --reps 20
# --- CPU: the banded op that the -fa on path falls back to on Metal
for kv in 2048 8192 32768; do run CPU banded 512 $kv --threads 18 --reps 3; done
for t in 6 12; do run CPU banded 512 8192 --threads $t --reps 3; done
for kv in 2048 8192 32768; do run CPU banded 1 $kv --threads 18 --reps 10; done
run CPU banded 512 1024 --swa 512 --threads 18 --reps 5
run CPU banded 1   768  --swa 512 --threads 18 --reps 10
# --- CPU reference: the plain (tiled) CPU flash attention at one shape
run CPU fa 512 8192 --threads 18 --reps 3
echo "# finished $(date -u +%FT%TZ)" >> "$OUT"
