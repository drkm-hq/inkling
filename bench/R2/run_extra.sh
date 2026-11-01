#!/bin/bash
# extra cases: CPU decode with a persistent threadpool; Metal FA with an unaligned n_kv (as the banded path passes it)
B=~/code/inkling-contrib/bench/R2/fa_bench
OUT=~/code/inkling-contrib/bench/R2/results/fa_bench_extra.log
export GGML_METAL_NDEBUG=1
: > "$OUT"
run() { "$B" "$@" 2>&1 | grep "^RESULT" >> "$OUT"; }
echo "# started $(date -u +%FT%TZ)" >> "$OUT"
for kv in 2048 8192 32768; do run CPU banded 1 $kv --threads 18 --pool 1 --reps 20; done
run CPU banded 1 768 --swa 512 --threads 18 --pool 1 --reps 20
run CPU banded 512 8192 --threads 18 --pool 1 --reps 3
run CPU banded 512 1024 --swa 512 --threads 18 --pool 1 --reps 5
for t in 6 12; do run CPU banded 1 8192 --threads $t --pool 1 --reps 20; done
run MTL0 fa 512 8209 --reps 8
run MTL0 fa 512 32785 --reps 8
run MTL0 fa 1 32785 --reps 20
run MTL0 fa 512 32768 --kv-size 65536 --reps 8
run MTL0 dense 512 32768 --kv-size 65536 --reps 5
echo "# finished $(date -u +%FT%TZ)" >> "$OUT"
