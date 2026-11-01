#!/bin/zsh
source ~/.venvs/inkling/bin/activate
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
echo "model: $MODEL"; echo "start: $(date -u +%FT%TZ)"
/usr/bin/time -l python ~/code/inkling-contrib/bench/load_mlx_chunked.py "$MODEL" 48 2>&1
echo "end: $(date -u +%FT%TZ)"
