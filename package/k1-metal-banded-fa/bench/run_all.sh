#!/bin/zsh
# K1 end-to-end window: wait for bench/GPU_LOCK (free or held by a dead pid), take it, run the quiet-GPU op perf,
# llama-bench and the greedy equivalence check under a ~25 minute budget, release the lock on exit.
set -u
LOCK=<workdir>/bench/GPU_LOCK
B=<workdir>/package/k1-metal-banded-fa/bench
ROOT=<workdir>/upstream/llama.cpp-inkling
EXPB=<scratch>
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
OUT=$B/e2e
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# 1. the lock: free, or its holder is gone
while true; do
    if [ ! -e "$LOCK" ]; then break; fi
    pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        log "lock held by dead pid $pid, removing it"
        rm -f "$LOCK"
        break
    fi
    sleep 10
done
if ! ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) k1-metal-banded-fa" > "$LOCK" ) 2>/dev/null; then
    log "lost the lock race"
    exit 1
fi
trap 'rm -f "$LOCK"; log "lock released"' EXIT
log "lock acquired (pid $$)"

# the previous holder's model may still be unmapping
for i in {1..9}; do
    big=$(ps -axo pid,rss,comm | awk -v me=$$ '$2 > 20*1024*1024 && $1 != me {print $3"("int($2/1048576)"G)"}' | tr '\n' ' ')
    [ -z "$big" ] && break
    log "waiting for large processes to exit: $big"
    sleep 10
done

T0=$SECONDS
elapsed() { echo $(( (SECONDS - T0) / 60 )); }

# 2. quiet-GPU op-level perf
log "perf: banded (committed build)"
"$ROOT/build/bin/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT_BANDED > "$OUT/perf-quiet-banded.log" 2>&1
log "perf: plain FA at the same geometry"
"$ROOT/build/bin/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=(2048|8192|32768|8209),nb=(512|1),' > "$OUT/perf-quiet-fa.log" 2>&1
if [ -e "$EXPB/DONE" ]; then
    log "perf: banded (single-pointer experiment build)"
    "$EXPB/bin/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT_BANDED > "$OUT/perf-quiet-banded-exp.log" 2>&1
else
    log "experiment build not ready, skipping its perf"
fi
log "perf done at $(elapsed) min"

# 3. llama-bench
bench() {
    log "llama-bench -fa $1 -d $2 -r $3"
    /usr/bin/time -l "$ROOT/build/bin/llama-bench" -m "$MODEL" -ngl 99 -fa $1 -p 512 -n 128 -d $2 -r $3 -o md > "$OUT/llama-bench-$4.md" 2> "$OUT/llama-bench-$4.err"
    log "exit=$? $(grep -E 'maximum resident' "$OUT/llama-bench-$4.err" | sed 's/^ *//')"
}
bench on  0,4096,16384,32768 2 fa-on
bench off 0,4096,16384       2 fa-off
log "llama-bench done at $(elapsed) min"

# 4. greedy equivalence (llama-server)
if [ $(elapsed) -lt 20 ]; then
    "$B/run_greedy.sh" "$OUT" > "$OUT/greedy.log" 2>&1
    log "greedy done at $(elapsed) min"
    grep -E "IDENTICAL|DIVERGE|maximum resident|prompt_n" "$OUT/greedy.log"
else
    log "SKIPPED greedy (time budget)"
fi

# 5. the dense path at 32K depth if time remains
if [ $(elapsed) -lt 23 ]; then
    bench off 32768 1 fa-off-d32k
else
    log "SKIPPED llama-bench -fa off -d 32768 (time budget)"
fi
log "all done at $(elapsed) min"
