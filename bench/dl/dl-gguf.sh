#!/bin/zsh
# Download Inkling-Small UD-IQ3_XXS GGUF (+ mmproj) into the HF cache with retries, plain HTTP (no xet).
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
for i in {1..30}; do
  echo "[$(date -u +%FT%TZ)] attempt $i"
  if uvx --from huggingface_hub hf download unsloth/Inkling-Small-GGUF --include "*UD-IQ3_XXS*" --include "mmproj-F16.gguf"; then
    echo "[$(date -u +%FT%TZ)] DONE"; exit 0
  fi
  echo "[$(date -u +%FT%TZ)] attempt $i failed; retrying in 30s"; sleep 30
done
echo FAILED; exit 1
