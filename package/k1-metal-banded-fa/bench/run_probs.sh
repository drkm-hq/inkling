#!/bin/zsh
# Top-k probabilities at the first greedy divergence (p_short pos 11, p_8k pos 12) for -fa on and -fa off.
# Takes bench/GPU_LOCK only if it is free right now (exits otherwise), holds it ~3 minutes, releases it with the server.
set -u
LOCK=<workdir>/bench/GPU_LOCK
B=<workdir>/package/k1-metal-banded-fa/bench
ROOT=<workdir>/upstream/llama.cpp-inkling
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
PROMPTS=<workdir>/bench/prompts
OUT=$B/e2e
PORT=8112
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
if ! ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) k1-metal-banded-fa (probs, ~3 min)" > "$LOCK" ) 2>/dev/null; then
    log "lock busy, not running"; exit 1
fi
SRV=""
cleanup() { [ -n "$SRV" ] && kill -INT $SRV 2>/dev/null; for i in $(seq 1 20); do [ -n "$SRV" ] && kill -0 $SRV 2>/dev/null || break; sleep 1; done; rm -f "$LOCK"; log "server stopped and lock released"; }
trap cleanup EXIT
for FA in on off; do
    log "server -fa $FA"
    "$ROOT/build/bin/llama-server" -m "$MODEL" -np 1 --no-kv-unified -ngl 99 --jinja --temp 0 -c 70000 -fa $FA --port $PORT --host 127.0.0.1 > "$OUT/probs-server-fa$FA.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 2; done
    echo "== -fa $FA p_short (divergence pos 11)"; python3 -I "$B/greedy_probs.py" $PORT "$PROMPTS/p_short.txt" 11 14
    echo "== -fa $FA p_8k (divergence pos 12)";    python3 -I "$B/greedy_probs.py" $PORT "$PROMPTS/p_8k.txt" 12 15
    kill -INT $SRV; wait $SRV; SRV=""
done
