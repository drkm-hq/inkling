#!/bin/bash
B=~/code/inkling-contrib/bench/R2/verify_dense
OUT=~/code/inkling-contrib/bench/R2/results/verify_dense.log
export GGML_METAL_NDEBUG=1
: > "$OUT"
echo "# started $(date -u +%FT%TZ)" >> "$OUT"
for a in "512 2048" "512 8192" "512 1024 --swa 512" "1 8192" "17 4100" "1 768 --swa 512"; do bash -c "$B $a" 2>&1 | grep "^VERIFY" >> "$OUT"; done
echo "# finished $(date -u +%FT%TZ)" >> "$OUT"
