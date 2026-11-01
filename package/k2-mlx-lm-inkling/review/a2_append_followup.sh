#!/bin/zsh
# Waits for the follow-up mini hold (a2_hold2.sh) to finish, then appends its headline
# results to REVIEW.md section 10 so the report completes itself if the queue is slow.
set -u
R=~/code/inkling-contrib/package/k2-mlx-lm-inkling/review
REVIEW=~/code/inkling-contrib/package/k2-mlx-lm-inkling/REVIEW.md
stamp() { date -u +%FT%TZ; }
t=0
until grep -q "follow-up hold done" $R/a2-hold2.log 2>/dev/null; do
  if [ $t -ge 21600 ]; then echo "[$(stamp)] gave up after 6 h"; exit 1; fi
  sleep 30; t=$((t+30))
done
echo "[$(stamp)] follow-up hold finished; appending to REVIEW.md"
{
  echo
  echo "### 10.1 Results (appended automatically by \`review/a2_append_followup.sh\` at $(stamp))"
  echo
  echo "Fold tree (\`cce0834\` model file = the head \`dc0a8b1\`'s) teacher forced on p_2k, vs the mlx-vlm reference logprobs (\`a2-forced-mlx_lm-fold.log\`):"
  echo
  grep -E "vs reference|argmax agrees" $R/a2-forced-mlx_lm-fold.log 2>/dev/null | sed 's/^/    /'
  echo
  echo "Per-layer comparison of the fold tree with mlx-vlm on the dump steps (\`a2-outlier-compare-fold.md\`; dump step 46 = the step-47 distribution):"
  echo
  grep -E "^== p_2k_raw step|^first layer" $R/a2-outlier-compare-fold.md 2>/dev/null | cut -c1-260 | sed 's/^/    /'
  echo
  echo "Dense tree (\`00cdad1\`) run a second time against this hold's own logprobs (\`a2-forced-mlx_lm-rerun.log\`; KL 0 = run-to-run deterministic):"
  echo
  grep -E "vs reference|argmax agrees" $R/a2-forced-mlx_lm-rerun.log 2>/dev/null | sed 's/^/    /'
  echo
  echo "Driver log: \`a2-hold2.log\`."
} >> $REVIEW
echo "[$(stamp)] appended"
