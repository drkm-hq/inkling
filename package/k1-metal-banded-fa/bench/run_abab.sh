#!/bin/zsh
# Interleaved op-level perf (3 rounds of: banded committed / banded single-pointer / plain FA) for min-of-N ratios.
# Waits for the e2e orchestrator to release bench/GPU_LOCK, re-takes it for the ~3 minutes this needs, releases it.
set -u
LOCK=<workdir>/bench/GPU_LOCK
B=<workdir>/package/k1-metal-banded-fa/bench
ROOT=<workdir>/upstream/llama.cpp-inkling
EXPB=<scratch>
OUT=$B/e2e
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
# wait for our own orchestrator to finish (it holds the lock until its trap runs)
while pgrep -f "run_all.sh" > /dev/null; do sleep 1; done
while [ -e "$LOCK" ]; do sleep 1; done
if ! ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) k1-metal-banded-fa-abab" > "$LOCK" ) 2>/dev/null; then
    log "lost the lock race, running without it"
fi
trap 'rm -f "$LOCK"; log "abab lock released"' EXIT
log "abab start"
for i in 1 2 3; do
    "$ROOT/build/bin/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT_BANDED > "$OUT/abab-banded-$i.log" 2>&1
    "$EXPB/bin/test-backend-ops"       perf -b MTL0 -o FLASH_ATTN_EXT_BANDED > "$OUT/abab-exp-$i.log" 2>&1
    "$ROOT/build/bin/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=(2048|8192|32768|8209),nb=(512|1),' > "$OUT/abab-fa-$i.log" 2>&1
    log "abab round $i done"
done
log "abab done"
