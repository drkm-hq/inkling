#!/bin/zsh
# Prints COMPLETE when the UD-IQ3_XXS shards are all real files summing to ~91.2 GiB and no *.incomplete blobs remain.
setopt null_glob
REPO=$HOME/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF
inc=$(find "$REPO/blobs" -name "*.incomplete" 2>/dev/null | wc -l | tr -d " ")
snap=$(ls -d "$REPO"/snapshots/*/ | head -1)
n=0; total=0
for i in 1 2 3; do
  f="$snap/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-0000$i-of-00003.gguf"
  if [[ -e $f ]]; then s=$(stat -Lf %z "$f"); n=$((n+1)); total=$((total+s)); fi
done
gib=$(python3 -c "print(round($total/2**30,1))")
if [[ $inc -eq 0 && $n -eq 3 ]]; then echo "COMPLETE shards=$n total=${gib}GiB"; else echo "INCOMPLETE shards=$n total=${gib}GiB incomplete_blobs=$inc blobs=$(du -sh $REPO/blobs | cut -f1)"; fi
