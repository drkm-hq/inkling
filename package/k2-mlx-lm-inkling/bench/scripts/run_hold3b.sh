#!/bin/zsh
# Re-take K2's hold 3 the instant run_hold3.sh releases it, re-run the fixed decode profile
# (about 2 minutes, within the 40-minute budget that started at 03:13:56), release.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/K2
B=~/code/inkling-contrib/bench
R=$B/results
stamp() { date -u +%FT%TZ; }
child=""
run() { "$@" & child=$!; wait $child; local rc=$?; child=""; return $rc; }
cleanup() {
  if [ -n "$child" ]; then kill $child 2>/dev/null; sleep 2; kill -9 $child 2>/dev/null; fi
  if [ "$(cut -d' ' -f1 $B/GPU_LOCK 2>/dev/null)" = "$$" ]; then rm -f $B/GPU_LOCK; echo "[$(stamp)] lock released by trap"; fi
  sed -i '' '/^K2$/d' $B/GPU_QUEUE
}
trap 'cleanup; exit 1' TERM INT HUP
trap 'cleanup' EXIT
until grep -q "hold 3 released" $R/K2-hold3.log; do sleep 0.2; done
if [ -f $B/GPU_LOCK ]; then echo "[$(stamp)] someone else took the lock: $(cat $B/GPU_LOCK); not profiling"; exit 0; fi
if [ $(date +%s) -gt $(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-29 03:50:00" +%s) ]; then echo "[$(stamp)] no budget left for the profile"; exit 0; fi
echo "$$ $(date +%FT%T) K2-chain-3b (decode profile, end of hold 3)" > $B/GPU_LOCK
grep -qx K2 $B/GPU_QUEUE || { { echo K2; cat $B/GPU_QUEUE; } > $B/GPU_QUEUE.new && mv $B/GPU_QUEUE.new $B/GPU_QUEUE; }
echo "[$(stamp)] hold 3b started (lock pid $$); queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
run python equiv.py --impl mlx_lm --no-lock --prompts p_short --chat-efforts "" --out $R/K2-equiv-mlx_lm-dense.json > $R/K2-equiv-mlx_lm-dense.log 2>&1; echo "dense greedy rc=$? $(stamp)"
run python equiv.py --impl mlx_lm --no-lock --model $HOME/models/inkling-small/mlx-2bit-attn8 --prompts p_short --chat-efforts "" --teacher $R/K2-equiv-mlx_lm-dense.json --out $R/K2-equiv-attn8-forced.json > $R/K2-equiv-attn8-forced.log 2>&1; echo "attn8 forced rc=$? $(stamp)"
run python equiv_forced_diff.py $R/K2-equiv-mlx_lm-dense.json $R/K2-equiv-attn8-forced.json > $R/K2-equiv-forced-diff-attn8.txt 2>&1; cat $R/K2-equiv-forced-diff-attn8.txt
run python equiv.py --impl mlx_lm --no-lock --teacher $R/K2-equiv-mlx_vlm.json --router-audit --out $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-mlx_lm-forced.log 2>&1; echo "forced dense rc=$? $(stamp)"
run python equiv_forced_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-forced-diff.txt 2>&1
run python equiv_report.py > $R/K2-equivalence.md 2>&1
run python profile_decode.py > $R/K2-profile-decode.log 2>&1; echo "profile rc=$? $(stamp)"
rm -f $B/GPU_LOCK
sed -i '' '/^K2$/d' $B/GPU_QUEUE
echo "[$(stamp)] hold 3b released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
