#!/bin/zsh
# Continuation of K2's hold 2 (budget: 40 min from 01:54:50 local = 02:34): the teacher-forced
# comparison first (acceptance evidence), then the back-to-back benchmarks if time allows.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/K2
B=~/code/inkling-contrib/bench
R=$B/results
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-29 02:33:00" +%s)
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
OLD=$1
echo "[$(stamp)] hold 2 continuation: taking over the lock from pid $OLD"
echo "$$ $(date +%FT%T) K2-chain-2b (continuation of hold 2)" > $B/GPU_LOCK
grep -qx K2 $B/GPU_QUEUE || { { echo K2; cat $B/GPU_QUEUE; } > $B/GPU_QUEUE.new && mv $B/GPU_QUEUE.new $B/GPU_QUEUE; }
kill $OLD 2>/dev/null
for i in $(seq 1 40); do
  big=$(ps -axo pid,rss,command | awk -v me=$$ '$2>20000000 && $1!=me {print $1}' | wc -l | tr -d ' ')
  [ "$big" = "0" ] && break
  sleep 2
done
echo "[$(stamp)] previous model process gone; queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
# 1. the acceptance evidence
run python equiv.py --impl mlx_lm --no-lock --teacher $R/K2-equiv-mlx_vlm.json --router-audit --out $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-mlx_lm-forced.log 2>&1; echo "forced rc=$? $(stamp)"
run python equiv_forced_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-forced-diff.txt 2>&1; cat $R/K2-equiv-forced-diff.txt
run python equiv_report.py > $R/K2-equivalence.md 2>&1
# 2. back-to-back benchmarks while the budget allows (p_32k only if there is time)
if [ $(date +%s) -lt $((DEADLINE - 600)) ]; then
  run ~/.venvs/inkling-base/bin/python ../baseline/bench_mlx.py --tag K2-same-hold --no-lock --prompts p_short,p_2k,p_8k --repeats 3 --max-tokens 128 --out $R/K2-bench-mlxvlm-same-hold.jsonl > $R/K2-bench-mlxvlm-same-hold.log 2>&1; echo "bench mlx-vlm rc=$? $(stamp)"
  run python bench_mlx_lm.py --tag mlxlm-2bit-same-hold --no-lock --prompts p_short,p_2k,p_8k --repeats 3 > $R/K2-bench-mlxlm-2bit-same-hold.log 2>&1; echo "bench mlx-lm rc=$? $(stamp)"
else
  echo "[$(stamp)] no time left for the benchmarks in this hold"
fi
rm -f $B/GPU_LOCK
sed -i '' '/^K2$/d' $B/GPU_QUEUE
echo "[$(stamp)] hold 2 released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
