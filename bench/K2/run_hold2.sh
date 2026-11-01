#!/bin/zsh
# K2 hold 2 under the GPU queue protocol: the mlx-vlm 0.7.11 reference generation (base venv,
# wired limit set), the 8-bit-attention variant (build, equivalence, benchmark) and the stock
# mlx_lm.generate CLI transcript. Same trap as run_chain.sh: a signal stops the model process
# by PID, releases our lock and leaves the queue.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/K2
B=~/code/inkling-contrib/bench
R=$B/results
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
ATTN8=~/models/inkling-small/mlx-2bit-attn8
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
echo "hold2 script start $(stamp)"
grep -qx K2 $B/GPU_QUEUE || echo K2 >> $B/GPU_QUEUE
run python gpulock.py wait K2 || exit 1
echo "$$ $(date +%FT%T) K2-chain-2" > $B/GPU_LOCK
echo "[$(stamp)] hold 2 started (lock pid $$)"
# 0. the bf16-attention greedy run again with the final build (the hold-1 run used the compiled router)
[ -f $R/K2-equiv-mlx_lm.json ] && mv $R/K2-equiv-mlx_lm.json $R/K2-equiv-mlx_lm-hold1.json
run python equiv.py --impl mlx_lm --no-lock --out $R/K2-equiv-mlx_lm.json > $R/K2-equiv-mlx_lm.log 2>&1; echo "equiv mlx_lm rc=$? $(stamp)"
# 1. the reference: mlx-vlm 0.7.11 wheel from ~/.venvs/inkling-base
run ~/.venvs/inkling-base/bin/python equiv.py --impl mlx_vlm --no-lock --out $R/K2-equiv-mlx_vlm.json > $R/K2-equiv-mlx_vlm.log 2>&1; echo "equiv mlx_vlm rc=$? $(stamp)"
run python equiv_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm.json > $R/K2-equiv-diff.txt 2>&1; echo "diff rc=$?"; cat $R/K2-equiv-diff.txt
# mlx-lm teacher-forced on the reference's tokens (+ the fp32 vs bf16 router audit)
run python equiv.py --impl mlx_lm --no-lock --teacher $R/K2-equiv-mlx_vlm.json --router-audit --out $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-mlx_lm-forced.log 2>&1; echo "forced rc=$? $(stamp)"
run python equiv_forced_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-forced-diff.txt 2>&1; cat $R/K2-equiv-forced-diff.txt
# 2. like-for-like benchmark under the same load: M1's mlx-vlm script (0.7.11 wheel, base venv)
#    immediately followed by the mlx-lm benchmark, same prompts, 3 timed runs each
run ~/.venvs/inkling-base/bin/python ../baseline/bench_mlx.py --tag K2-same-hold --no-lock --prompts p_short,p_2k,p_8k,p_32k --repeats 3 --max-tokens 128 --out $R/K2-bench-mlxvlm-same-hold.jsonl > $R/K2-bench-mlxvlm-same-hold.log 2>&1; echo "bench mlx-vlm rc=$? $(stamp)"
run python bench_mlx_lm.py --tag mlxlm-2bit-same-hold --no-lock --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > $R/K2-bench-mlxlm-2bit-same-hold.log 2>&1; echo "bench mlx-lm rc=$? $(stamp)"
rm -f $B/GPU_LOCK
sed -i '' '/^K2$/d' $B/GPU_QUEUE
echo "[$(stamp)] hold 2 released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
