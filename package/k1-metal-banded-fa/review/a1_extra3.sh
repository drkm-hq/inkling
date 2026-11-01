#!/bin/zsh
# A1 review, third extension of the same GPU window: per-tensor dump (llama-eval-callback) of a short prompt on the
# three attention paths, to localize the kernel/dense-vs-CPU discrepancy to a layer and an op.
set -u
LOCK=<workdir>/bench/GPU_LOCK
QUEUE=<workdir>/bench/GPU_QUEUE
A1=<workdir>/upstream/llama.cpp-a1/build-a1/bin
BASE=<workdir>/upstream/llama.cpp-a1-base/build-a1-base/bin
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
P=<workdir>/bench/prompts
OUT=<workdir>/package/k1-metal-banded-fa/review/e2e
PREV=$1
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
while kill -0 $PREV 2>/dev/null; do sleep 1; done
for i in 1 2 3 4 5; do
    if ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) A1 review a1_extra3.sh (same window, eval-callback)" > "$LOCK" ) 2>/dev/null; then break; fi
    pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then log "lock taken by pid $pid meanwhile; giving up"; exit 1; fi
    sleep 1
done
grep -q "^$$ " "$LOCK" || { log "could not take the lock"; exit 1; }
{ echo A1; grep -v '^A1$' "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
release() { rm -f "$LOCK"; sed -i '' '/^A1$/d' "$QUEUE"; log "lock released, queue line removed"; }
trap 'release' EXIT
log "extension-3 lock acquired (pid $$)"
T0=$SECONDS
for i in 1 2 3 4 5 6; do [ -x "$A1/llama-eval-callback" ] && [ -x "$BASE/llama-eval-callback" ] && break; sleep 10; done
PROMPT="$(cat $P/p_short.txt)"
"$BASE/llama-eval-callback" -m "$MODEL" -ngl 99 -fa on  -c 2048 -n 1 --temp 0 -p "$PROMPT" > "$OUT/evalcb-base-faon.log" 2> "$OUT/evalcb-base-faon.err"; log "evalcb base -fa on exit=$? ($(wc -l < "$OUT/evalcb-base-faon.log") lines) at $(( (SECONDS - T0) / 60 )) min"
"$A1/llama-eval-callback"   -m "$MODEL" -ngl 99 -fa on  -c 2048 -n 1 --temp 0 -p "$PROMPT" > "$OUT/evalcb-k1-faon.log"   2> "$OUT/evalcb-k1-faon.err";   log "evalcb k1 -fa on exit=$? at $(( (SECONDS - T0) / 60 )) min"
"$A1/llama-eval-callback"   -m "$MODEL" -ngl 99 -fa off -c 2048 -n 1 --temp 0 -p "$PROMPT" > "$OUT/evalcb-k1-faoff.log"  2> "$OUT/evalcb-k1-faoff.err";  log "evalcb k1 -fa off exit=$? at $(( (SECONDS - T0) / 60 )) min"
log "extension-3 done at $(( (SECONDS - T0) / 60 )) min"
