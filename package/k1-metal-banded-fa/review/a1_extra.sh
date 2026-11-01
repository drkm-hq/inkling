#!/bin/zsh
# A1 review, extension of the same GPU window: KL divergence of the kernel (-fa on) and of the dense chain (-fa off)
# against the CPU banded reference (pristine build, -fa on) on p_8k with n_ctx 2048.
set -u
LOCK=<workdir>/bench/GPU_LOCK
QUEUE=<workdir>/bench/GPU_QUEUE
A1=<workdir>/upstream/llama.cpp-a1/build-a1/bin
BASEPPL=<workdir>/upstream/llama.cpp-a1-base/build-a1-base/bin
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
P=<workdir>/bench/prompts
OUT=<workdir>/package/k1-metal-banded-fa/review/e2e
ORCH=$1
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
# wait for the orchestrator (my own process) to exit, then take the lock back immediately
while kill -0 $ORCH 2>/dev/null; do sleep 1; done
for i in 1 2 3 4 5; do
    if ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) A1 review a1_extra.sh (same window, KLD)" > "$LOCK" ) 2>/dev/null; then
        break
    fi
    pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then log "lock taken by pid $pid meanwhile; giving up the extension"; exit 1; fi
    sleep 1
done
grep -q "^$$ " "$LOCK" || { log "could not take the lock"; exit 1; }
# bookkeeping: the holder is first in the queue
{ echo A1; grep -v '^A1$' "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
release() { rm -f "$LOCK"; sed -i '' '/^A1$/d' "$QUEUE"; log "lock released, queue line removed"; }
trap 'release' EXIT
log "extension lock acquired (pid $$)"
T0=$SECONDS
BASEF="$OUT/ref-logits-p8k-c2048.bin"
/usr/bin/time -l "$BASEPPL/llama-perplexity" -m "$MODEL" -ngl 99 -fa on -c 2048 -b 2048 -f "$P/p_8k.txt" --kl-divergence-base "$BASEF" > "$OUT/kld-base-faon.log" 2> "$OUT/kld-base-faon.err"
log "reference log-probs written: exit=$? $(ls -la "$BASEF" | awk '{print $5}') bytes, $(grep -E 'Final estimate' "$OUT/kld-base-faon.err" | sed 's/.*Final/Final/'), $(( (SECONDS - T0) / 60 )) min"
for FA in on off; do
    /usr/bin/time -l "$A1/llama-perplexity" -m "$MODEL" -ngl 99 -fa $FA -c 2048 -b 2048 -f "$P/p_8k.txt" --kl-divergence --kl-divergence-base "$BASEF" > "$OUT/kld-k1-fa$FA.log" 2> "$OUT/kld-k1-fa$FA.err"
    log "kld -fa $FA exit=$? at $(( (SECONDS - T0) / 60 )) min"
done
rm -f "$BASEF"
log "extension done at $(( (SECONDS - T0) / 60 )) min"
