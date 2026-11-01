#!/bin/zsh
# kv8h re-run (hybrid uniform 8-bit keys / 4-bit values, the HybridQuantKVCache path that does reach a CacheList).
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
until grep -q 'kv8tq done' results/phase3.log 2>/dev/null; do sleep 20; done
/usr/bin/time -l python baseline/bench_mlx.py --tag kv8h --max-tokens 128 --quiet-wait 600 --prompts p_short,p_8k --repeats 3 --kv-bits 8 --kv-key-bits 8 --kv-value-bits 4 --kv-group-size 64 --quantized-kv-start 0 > results/mlx-kv8h.log 2>&1
echo "[$(date -u +%FT%TZ)] kv8h done rc=$?"
