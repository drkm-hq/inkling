#!/bin/zsh
# llama-bench pp512/tg128 at depths 0/4096/16384/32768, -fa 1 (Metal banded kernel) vs -fa 0 (dense chain).
# usage: run_llama_bench.sh <out-dir> [reps]
set -u
ROOT=<workdir>/upstream/llama.cpp-inkling
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
OUT=$1
REPS=${2:-2}
mkdir -p "$OUT"
for FA in on off; do
    echo "=== llama-bench -fa $FA ($(date -u +%H:%M:%SZ))"
    /usr/bin/time -l "$ROOT/build/bin/llama-bench" -m "$MODEL" -ngl 99 -fa $FA -p 512 -n 128 -d 0,4096,16384,32768 -r $REPS -o md \
        > "$OUT/llama-bench-fa$FA.md" 2> "$OUT/llama-bench-fa$FA.err"
    echo "exit=$?"
    grep -E "maximum resident|real" "$OUT/llama-bench-fa$FA.err" | head -3
done
