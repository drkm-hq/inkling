#!/bin/zsh
# llama-bench pp512 / tg128 at increasing KV depth, Metal, for one -fa setting.
# Usage: run_llama_bench.sh <fa:0|1> <depths e.g. 0,4096,16384,32768> [reps]
set -u
FA=$1; DEPTHS=$2; REPS=${3:-3}
SRC=${LLAMA_SRC:-$HOME/code/inkling-contrib/upstream/llama.cpp-base}
BIN=${LLAMA_BIN:-$SRC/build/bin}
MODEL=$(ls $HOME/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/*/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf | head -1)
OUT=$HOME/code/inkling-contrib/bench/results
echo "start $(date -u +%FT%TZ) fa=$FA depths=$DEPTHS reps=$REPS model=$MODEL"
echo "commit $(git -C $SRC rev-parse HEAD) ($SRC)"; $BIN/llama-bench --version 2>/dev/null | grep -v ggml_metal
echo "cmd: $BIN/llama-bench -m $MODEL -ngl 99 -fa $FA -p 512 -n 128 -d $DEPTHS -r $REPS -o md -oe jsonl --progress"
$BIN/llama-bench -m "$MODEL" -ngl 99 -fa "$FA" -p 512 -n 128 -d "$DEPTHS" -r "$REPS" -o md -oe jsonl --progress 2> "$OUT/llama-bench-fa$FA.jsonl"
rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc"
