#!/bin/zsh
# A2 follow-up mini hold (<= 7 min): the p_2k step-47 outlier at K2's fold commit
# (cce0834, shared experts folded into the routed gather) with per-layer dumps,
# and a repeat of the dense-tree run to measure run-to-run determinism. Waits for
# the main A2 hold (pid in a2-hold.pid) to exit before queueing, so the main
# hold's cleanup cannot remove this entry.
set -u
C=~/code/inkling-contrib
B=$C/bench
R=$C/package/k2-mlx-lm-inkling/review
FOLD=$C/upstream/mlx-lm-a2-fold
PY=~/.venvs/inkling-a2/bin/python
BUDGET=${BUDGET:-420}
stamp() { date -u +%FT%TZ; }
child=""
run() { "$@" & child=$!; wait $child; local rc=$?; child=""; return $rc; }
cleanup() {
  if [ -n "$child" ]; then kill $child 2>/dev/null; sleep 2; kill -9 $child 2>/dev/null; fi
  if [ "$(cut -d' ' -f1 $B/GPU_LOCK 2>/dev/null)" = "$$" ]; then rm -f $B/GPU_LOCK; echo "[$(stamp)] GPU_LOCK released"; fi
  sed -i '' '/^A2$/d' $B/GPU_QUEUE
  echo "[$(stamp)] A2 left the queue; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
}
trap 'cleanup; exit 1' TERM INT HUP
trap 'cleanup' EXIT
MAIN=$(awk '{print $NF}' $R/a2-hold.pid)
echo "[$(stamp)] follow-up hold script start (pid $$); waiting for the main hold (pid $MAIN) to exit"
while kill -0 $MAIN 2>/dev/null; do sleep 10; done
echo "[$(stamp)] main hold gone; queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
grep -qx A2 $B/GPU_QUEUE || echo A2 >> $B/GPU_QUEUE
cd $B/K2 && run $PY gpulock.py wait A2 || exit 1
echo "$$ $(date +%FT%T) A2-review-2 (fold outlier check, bounded ${BUDGET}s)" > $B/GPU_LOCK
START=$(date +%s)
echo "[$(stamp)] follow-up hold started; load=$(uptime | sed 's/.*load averages: //')"
cd $R
# 1. the fold commit (cce0834) on the same tokens with dumps at 45-48
mkdir -p $R/a2-dumps-fold
PYTHONPATH=$FOLD run $PY a2_forced.py --impl mlx_lm --runs p_2k_raw --dump "p_2k_raw:45,46,47,48" \
    --dump-dir $R/a2-dumps-fold --out $R/a2-forced-mlx_lm-fold.json > $R/a2-forced-mlx_lm-fold.log 2>&1
echo "[$(stamp)] fold forced rc=$? ($(( BUDGET - $(date +%s) + START ))s left)"
cp $R/a2-dumps/mlx_vlm_p_2k_raw_step0*.npz $R/a2-dumps-fold/ 2>/dev/null
run $PY a2_outlier_compare.py $R/a2-dumps-fold p_2k_raw 45 46 47 48 > $R/a2-outlier-compare-fold.md 2>&1
# 2. the dense tree again: run-to-run determinism against this hold's own logprobs
if [ $(( BUDGET - $(date +%s) + START )) -gt 120 ]; then
  run $PY a2_forced.py --impl mlx_lm --runs p_2k_raw --ref-logprobs $R/a2-logprobs-mlx_lm --out $R/a2-forced-mlx_lm-rerun.json > $R/a2-forced-mlx_lm-rerun.log 2>&1
  echo "[$(stamp)] dense rerun rc=$? ($(( BUDGET - $(date +%s) + START ))s left)"
fi
echo "[$(stamp)] follow-up hold done after $(( $(date +%s) - START ))s"
