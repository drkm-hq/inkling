#!/bin/zsh
# A1 review end-to-end window. Waits for the GPU lock protocol (lock absent or dead, no >20 GB process, A1 first in
# the queue), then runs: greedy equivalence (kernel vs the CPU banded reference), perplexity, llama-bench, stability.
set -u
LOCK=<workdir>/bench/GPU_LOCK
QUEUE=<workdir>/bench/GPU_QUEUE
A1=<workdir>/upstream/llama.cpp-a1/build-a1/bin      # metal-banded-fa code + A1 tests
BASE=<workdir>/upstream/llama.cpp-base/build/bin     # pristine pr-25731 (3582c5ee0)
BASEPPL=<workdir>/upstream/llama.cpp-a1-base/build-a1-base/bin  # pristine code, A1 build (llama-perplexity only)
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
P=<workdir>/bench/prompts
OUT=<workdir>/package/k1-metal-banded-fa/review/e2e
PORT=8123
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# 1. wait for the lock protocol
while true; do
    first=$(head -1 "$QUEUE" 2>/dev/null)
    big=$(ps -axo pid,rss,comm | awk -v me=$$ 'NR > 1 && $2 > 20*1024*1024 && $1 != me {print $1}' | head -1)
    free=0
    if [ ! -e "$LOCK" ]; then free=1; else
        pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then free=1; fi
    fi
    if [ "$first" = "A1" ] && [ $free = 1 ] && [ -z "$big" ]; then break; fi
    sleep 15
done
if ! ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) A1 review a1_e2e.sh" > "$LOCK" ) 2>/dev/null; then
    log "lost the lock race"; exit 1
fi
release() { rm -f "$LOCK"; sed -i '' '/^A1$/d' "$QUEUE"; log "lock released, queue line removed"; }
trap 'release' EXIT
log "lock acquired (pid $$)"
T0=$SECONDS
elapsed() { echo $(( (SECONDS - T0) / 60 )); }

SRV=""; SRVPID=""
start_server() { # <bin-dir> <tag> <extra args...>
    local bin=$1; local tag=$2; shift 2
    /usr/bin/time -l "$bin/llama-server" -m "$MODEL" -np 1 --no-kv-unified -ngl 99 --jinja --temp 0 \
        --port $PORT --host 127.0.0.1 "$@" > "$OUT/server-$tag.log" 2> "$OUT/server-$tag.err" &
    SRV=$!
    for i in $(seq 1 900); do
        curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
        if ! kill -0 $SRV 2>/dev/null; then log "server $tag died"; return 1; fi
        sleep 2
    done
    SRVPID=$(pgrep -P $SRV | head -1)
    log "server $tag up after ~$((i*2)) s (time-wrapper $SRV, server $SRVPID)"
}
stop_server() {
    [ -n "$SRVPID" ] && kill -INT $SRVPID 2>/dev/null
    wait $SRV 2>/dev/null
    log "server stopped: $(grep -E 'maximum resident' "$OUT/server-$1.err" | sed 's/^ *//')"
    SRV=""; SRVPID=""
}

# 1b. op-level perf, banded vs plain at 512x8192 and 1x32768 (and the other two combinations), 3 interleaved repeats
for i in 1 2 3; do
    "$A1/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT_BANDED -p 'n_q=(512|1),n_kv=(8192|32768),rel_extent=1024,mask_kind=1,kv_type=f16,rel_type=f32,strided=0,kv_pad=0' > "$OUT/perf-banded-$i.log" 2>&1
    "$A1/test-backend-ops" perf -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=(8192|32768),nb=(512|1),' > "$OUT/perf-plain-$i.log" 2>&1
    log "perf repeat $i done at $(elapsed) min"
done

# 2. greedy equivalence: kernel (-fa on, metal-banded-fa) vs the reference (-fa on, pristine = CPU banded op)
start_server "$A1" k1-faon-c8k -c 8192 -fa on && \
    python3 -I "$OUT/../a1_greedy.py" $PORT "$OUT/greedy-k1-faon.json" 96 "$P/p_short.txt" "$P/p_2k.txt" > "$OUT/greedy-k1-faon.txt" 2>&1
stop_server k1-faon-c8k
log "k1 greedy done at $(elapsed) min"
start_server "$BASE" base-faon-c8k -c 8192 -fa on && \
    python3 -I "$OUT/../a1_greedy.py" $PORT "$OUT/greedy-base-faon.json" 96 "$P/p_short.txt" "$P/p_2k.txt" > "$OUT/greedy-base-faon.txt" 2>&1
stop_server base-faon-c8k
log "base greedy done at $(elapsed) min"

# 3. perplexity, p_8k (8218 tokens, 2 chunks of 4096), kernel -fa on vs the dense chain -fa off
for FA in on off; do
    /usr/bin/time -l "$A1/llama-perplexity" -m "$MODEL" -ngl 99 -fa $FA -c 4096 -b 2048 -f "$P/p_8k.txt" > "$OUT/ppl-k1-fa$FA.log" 2> "$OUT/ppl-k1-fa$FA.err"
    log "ppl -fa $FA exit=$? $(grep -E 'Final estimate' "$OUT/ppl-k1-fa$FA.log") $(grep -E 'maximum resident' "$OUT/ppl-k1-fa$FA.err" | sed 's/^ *//')"
done
log "ppl done at $(elapsed) min"

# 4. llama-bench pp512/tg128 at d0 and d4096, -r 3
for FA in 1 0; do
    /usr/bin/time -l "$A1/llama-bench" -m "$MODEL" -ngl 99 -fa $FA -p 512 -n 128 -d 0,4096 -r 3 -o md > "$OUT/llama-bench-fa$FA.md" 2> "$OUT/llama-bench-fa$FA.err"
    log "llama-bench -fa $FA exit=$? $(grep -E 'maximum resident' "$OUT/llama-bench-fa$FA.err" | sed 's/^ *//')"
done
log "llama-bench done at $(elapsed) min"

# 5. stability: p_32k (and p_64k if the budget allows) on the kernel, 64 new tokens
start_server "$A1" k1-faon-c70k -c 70000 -fa on
python3 -I "$OUT/../a1_greedy.py" $PORT "$OUT/greedy-k1-faon-32k.json" 64 "$P/p_32k.txt" > "$OUT/greedy-k1-faon-32k.txt" 2>&1
log "p_32k done at $(elapsed) min"
if [ $(elapsed) -lt 22 ]; then
    python3 -I "$OUT/../a1_greedy.py" $PORT "$OUT/greedy-k1-faon-64k.json" 64 "$P/p_64k.txt" > "$OUT/greedy-k1-faon-64k.txt" 2>&1
    log "p_64k done at $(elapsed) min"
else
    log "SKIPPED p_64k (budget)"
fi
stop_server k1-faon-c70k

# 6. the reference perplexity (pristine -fa on = CPU banded op) if the budget allows
if [ $(elapsed) -lt 28 ] && [ -x "$BASEPPL/llama-perplexity" ]; then
    /usr/bin/time -l "$BASEPPL/llama-perplexity" -m "$MODEL" -ngl 99 -fa on -c 4096 -b 2048 -f "$P/p_8k.txt" > "$OUT/ppl-base-faon.log" 2> "$OUT/ppl-base-faon.err"
    log "base ppl exit=$? $(grep -E 'Final estimate' "$OUT/ppl-base-faon.log")"
else
    log "SKIPPED base ppl (budget)"
fi
log "all done at $(elapsed) min"
