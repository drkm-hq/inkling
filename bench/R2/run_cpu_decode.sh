#!/bin/bash
# clean re-run of the CPU banded decode cases (nothing else running on the machine)
B=~/code/inkling-contrib/bench/R2/fa_bench
OUT=~/code/inkling-contrib/bench/R2/results/fa_bench_cpu_decode.log
export GGML_METAL_NDEBUG=1
: > "$OUT"
run() { "$B" "$@" 2>&1 | grep "^RESULT" >> "$OUT"; }
echo "# started $(date -u +%FT%TZ)" >> "$OUT"
for t in 6 12 18; do for p in 0 1; do run CPU banded 1 8192 --threads $t --pool $p --reps 30; done; done
for kv in 2048 32768; do run CPU banded 1 $kv --threads 12 --pool 1 --reps 30; done
run CPU banded 1 768 --swa 512 --threads 12 --pool 1 --reps 30
run CPU banded 1 768 --swa 512 --threads 12 --pool 0 --reps 30
run CPU banded 512 8192 --threads 18 --pool 1 --reps 3
echo "# finished $(date -u +%FT%TZ)" >> "$OUT"
