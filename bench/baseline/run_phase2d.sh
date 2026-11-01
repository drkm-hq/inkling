#!/bin/zsh
# Stock-wheel CLI attempt (mlx-vlm 0.7.11 wheel via ~/.venvs/inkling-base): expected to hit the Metal GPU timeout
# in the one-shot mx.eval of the eager load. Runs after phase2b's own CLI step (which uses the chunked-loader checkout).
cd ~/code/inkling-contrib/bench
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
until grep -q 'cli done' results/phase3.log 2>/dev/null; do sleep 20; done
baseline/with_gpu_lock.sh /usr/bin/time -l ~/.venvs/inkling-base/bin/python -m mlx_vlm.generate --model "$MODEL" --prompt "$(cat prompts/p_short.txt)" --max-tokens 128 --temperature 0 --verbose > results/mlx-cli-wheel-p_short.log 2>&1
echo "[$(date -u +%FT%TZ)] stock-wheel cli done rc=$?"
