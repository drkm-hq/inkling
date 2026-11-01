#!/bin/zsh
cd ~/code/inkling-contrib/bench
baseline/gpu_turn.sh llama baseline/hold_llama.sh
baseline/gpu_turn.sh mlx-variants baseline/hold_mlx.sh
echo "[$(date -u +%FT%TZ)] all holds done"
