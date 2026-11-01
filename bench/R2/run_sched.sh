#!/bin/bash
B=~/code/inkling-contrib/bench/R2/sched_bench
OUT=~/code/inkling-contrib/bench/R2/results/sched_bench.log
export GGML_METAL_NDEBUG=1
: > "$OUT"
run() { "$B" "$@" 2>&1 | grep "^RESULT" >> "$OUT"; }
echo "# started $(date -u +%FT%TZ) loadavg=$(sysctl -n vm.loadavg)" >> "$OUT"
run fa     1 8192 768 --threads 12 --reps 30
run banded 1 8192 768 --threads 12 --reps 30
run fa     1 32768 768 --threads 12 --reps 30
run banded 1 32768 768 --threads 12 --reps 30
run fa     512 8192 1024 --threads 18 --reps 5
run banded 512 8192 1024 --threads 18 --reps 3
echo "# finished $(date -u +%FT%TZ) loadavg=$(sysctl -n vm.loadavg)" >> "$OUT"
