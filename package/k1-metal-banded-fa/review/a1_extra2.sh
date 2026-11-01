#!/bin/zsh
# A1 review, second extension of the same GPU window: interleaved llama-bench A/B, -fa 1 / 0 / 1 / 0 at d0 and d4096.
set -u
LOCK=<workdir>/bench/GPU_LOCK
QUEUE=<workdir>/bench/GPU_QUEUE
A1=<workdir>/upstream/llama.cpp-a1/build-a1/bin
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
OUT=<workdir>/package/k1-metal-banded-fa/review/e2e
PREV=$1
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
while kill -0 $PREV 2>/dev/null; do sleep 1; done
for i in 1 2 3 4 5; do
    if ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) A1 review a1_extra2.sh (same window, interleaved bench)" > "$LOCK" ) 2>/dev/null; then
        break
    fi
    pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then log "lock taken by pid $pid meanwhile; giving up"; exit 1; fi
    sleep 1
done
grep -q "^$$ " "$LOCK" || { log "could not take the lock"; exit 1; }
{ echo A1; grep -v '^A1$' "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
release() { rm -f "$LOCK"; sed -i '' '/^A1$/d' "$QUEUE"; log "lock released, queue line removed"; }
trap 'release' EXIT
log "extension-2 lock acquired (pid $$)"
T0=$SECONDS
for round in 1 2; do
    for FA in 1 0; do
        "$A1/llama-bench" -m "$MODEL" -ngl 99 -fa $FA -p 512 -n 128 -d 0,4096 -r 2 -o md > "$OUT/abab-fa$FA-$round.md" 2> "$OUT/abab-fa$FA-$round.err"
        log "abab round $round -fa $FA exit=$? at $(( (SECONDS - T0) / 60 )) min; load: $(uptime | sed 's/.*load averages: //')"
    done
done
log "extension-2 done at $(( (SECONDS - T0) / 60 )) min"
