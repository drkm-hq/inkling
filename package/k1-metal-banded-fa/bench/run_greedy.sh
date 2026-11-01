#!/bin/zsh
# Greedy equivalence -fa 0 vs -fa 1 with llama-server (-np 1 --no-kv-unified -ngl 99 --jinja --temp 0 -c 70000).
# usage: run_greedy.sh <out-dir>
set -u
ROOT=<workdir>/upstream/llama.cpp-inkling
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
PROMPTS=<workdir>/bench/prompts
OUT=$1
PORT=8111
mkdir -p "$OUT"
for FA in on off; do
    echo "=== llama-server -fa $FA ($(date -u +%H:%M:%SZ))"
    /usr/bin/time -l "$ROOT/build/bin/llama-server" -m "$MODEL" -np 1 --no-kv-unified -ngl 99 --jinja --temp 0 -c 70000 -fa $FA \
        --port $PORT --host 127.0.0.1 > "$OUT/server-fa$FA.log" 2> "$OUT/server-fa$FA.err" &
    SRV=$!
    for i in $(seq 1 600); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 2
    done
    echo "server up after ~$((i*2)) s"
    # warm-up (discarded), then the three prompts
    python3 -I "$(dirname "$0")/greedy_check.py" $PORT "$OUT/warm-fa$FA.json" "$PROMPTS/p_short.txt" > /dev/null
    python3 -I "$(dirname "$0")/greedy_check.py" $PORT "$OUT/greedy-fa$FA.json" "$PROMPTS/p_short.txt" "$PROMPTS/p_8k.txt" "$PROMPTS/p_32k.txt"
    # $SRV is the /usr/bin/time wrapper, which does not forward signals: stop the server itself (its child)
    kill -INT $(pgrep -P $SRV)
    wait $SRV
    grep -E "maximum resident" "$OUT/server-fa$FA.err"
done
python3 -I - "$OUT/greedy-faoff.json" "$OUT/greedy-faon.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
for k in a:
    ta, tb = a[k]["tokens"], b[k]["tokens"]
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    if div is None and len(ta) == len(tb):
        print(f"{k}: IDENTICAL ({len(ta)} tokens)")
    else:
        print(f"{k}: DIVERGE at token {div} of {len(ta)}/{len(tb)}")
        print("  fa0:", repr(a[k]["content"][:300]))
        print("  fa1:", repr(b[k]["content"][:300]))
PY
