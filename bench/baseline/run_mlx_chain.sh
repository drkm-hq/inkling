#!/bin/zsh
# Runs after the baseline: (1) 8-bit KV cache on p_short/p_8k, (2) the stock CLI (expected to hit the GPU timeout).
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
echo "chain start $(date -u +%FT%TZ)"
/usr/bin/time -l python baseline/bench_mlx.py --tag kv8 --prompts p_short,p_8k --repeats 3 --max-tokens 128 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0 > results/mlx-kv8.log 2>&1
echo "kv8 done rc=$? $(date -u +%FT%TZ)"
baseline/with_gpu_lock.sh /usr/bin/time -l python -m mlx_vlm.generate --model "$MODEL" --prompt "$(cat prompts/p_short.txt)" \
  --max-tokens 128 --temperature 0 --verbose > results/mlx-cli-p_short.log 2>&1
echo "cli done rc=$? $(date -u +%FT%TZ)"
echo "chain end $(date -u +%FT%TZ)"
