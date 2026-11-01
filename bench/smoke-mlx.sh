#!/bin/zsh
source ~/.venvs/inkling/bin/activate
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
echo "model: $MODEL"; echo "start: $(date -u +%FT%TZ)"
/usr/bin/time -l python -m mlx_vlm.generate --model "$MODEL" --prompt "The capital of France is" --max-tokens 48 --temperature 0 --verbose 2>&1
echo "end: $(date -u +%FT%TZ)"
