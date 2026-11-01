#!/bin/zsh
# K2 real-weights chain under the GPU queue protocol (bench/GPU_QUEUE + bench/GPU_LOCK).
# Hold 1 (<= 40 min): greedy equivalence mlx-vlm vs mlx-lm, then the mlx-lm benchmark.
# Hold 2 (re-queued at the end): the 8-bit-attention variant (build, equivalence, benchmark)
# and the stock mlx_lm.generate CLI transcript.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/K2
B=~/code/inkling-contrib/bench
R=$B/results
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
ATTN8=~/models/inkling-small/mlx-2bit-attn8
stamp() { date -u +%FT%TZ; }
# Every step runs as a tracked child so that a signal to this shell stops the model
# process (by PID), releases our lock and leaves the queue, all in one trap.
child=""
run() { "$@" & child=$!; wait $child; local rc=$?; child=""; return $rc; }
cleanup() {
  if [ -n "$child" ]; then kill $child 2>/dev/null; sleep 2; kill -9 $child 2>/dev/null; fi
  if [ "$(cut -d' ' -f1 $B/GPU_LOCK 2>/dev/null)" = "$$" ]; then rm -f $B/GPU_LOCK; echo "[$(stamp)] lock released by trap"; fi
  sed -i '' '/^K2$/d' $B/GPU_QUEUE
}
trap 'cleanup; exit 1' TERM INT HUP
trap 'cleanup' EXIT
take_lock() {
  run python gpulock.py wait K2 || exit 1
  echo "$$ $(date +%FT%T) K2-chain-$1" > $B/GPU_LOCK
  echo "[$(stamp)] hold $1 started (lock pid $$)"
}
release_lock() {
  rm -f $B/GPU_LOCK
  sed -i '' '/^K2$/d' $B/GPU_QUEUE
  echo "[$(stamp)] hold $1 released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
}
echo "chain start $(stamp) model=$MODEL"
grep -qx K2 $B/GPU_QUEUE || echo K2 >> $B/GPU_QUEUE

# ---------------------------------------------------------------- hold 1
take_lock 1
# the mlx-vlm REFERENCE runs from the untouched 0.7.11 wheel in ~/.venvs/inkling-base
run ~/.venvs/inkling-base/bin/python equiv.py --impl mlx_vlm --no-lock --out $R/K2-equiv-mlx_vlm.json > $R/K2-equiv-mlx_vlm.log 2>&1; echo "equiv mlx_vlm rc=$? $(stamp)"
run python equiv.py --impl mlx_lm --no-lock --out $R/K2-equiv-mlx_lm.json > $R/K2-equiv-mlx_lm.log 2>&1; echo "equiv mlx_lm rc=$? $(stamp)"
run python equiv_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm.json > $R/K2-equiv-diff.txt 2>&1; echo "diff rc=$?"; cat $R/K2-equiv-diff.txt
run python bench_mlx_lm.py --tag mlxlm-2bit --no-lock --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > $R/K2-bench-mlxlm-2bit.log 2>&1; echo "bench 2bit rc=$? $(stamp)"
# per-token dispatch cost at tiny widths, each side from its own venv (about a minute)
run ~/.venvs/inkling-base/bin/python dispatch_bench.py --iters 30 --impl mlx_vlm > $R/K2-dispatch-bench-base.log 2>&1
run python dispatch_bench.py --iters 30 --impl mlx_lm >> $R/K2-dispatch-bench-base.log 2>&1
release_lock 1

# ---------------------------------------------------------------- hold 2
echo K2 >> $B/GPU_QUEUE
take_lock 2
if [ ! -d $ATTN8 ]; then
  mkdir -p ~/models/inkling-small
  run python make_attn8.py --no-lock --out $ATTN8 > $R/K2-make-attn8.log 2>&1; echo "make attn8 rc=$? $(stamp)"
fi
run python equiv.py --impl mlx_lm --no-lock --model $ATTN8 --out $R/K2-equiv-mlx_lm-attn8.json > $R/K2-equiv-mlx_lm-attn8.log 2>&1; echo "equiv attn8 rc=$? $(stamp)"
run python equiv_diff.py $R/K2-equiv-mlx_lm.json $R/K2-equiv-mlx_lm-attn8.json > $R/K2-equiv-diff-attn8.txt 2>&1; cat $R/K2-equiv-diff-attn8.txt
run python bench_mlx_lm.py --tag mlxlm-attn8 --no-lock --model $ATTN8 --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > $R/K2-bench-mlxlm-attn8.log 2>&1; echo "bench attn8 rc=$? $(stamp)"
# stock CLI with its one-shot load (may trip the Metal watchdog: a finding either way)
{ echo "\$ mlx_lm.generate --model $MODEL -p 'The secret to baking a good cake is' -m 200"
  run /usr/bin/time -l python -m mlx_lm generate --model "$MODEL" -p "The secret to baking a good cake is" -m 200 2>&1
  echo "[exit code $?]"; } > $R/K2-cli-generate.log 2>&1
echo "cli done $(stamp)"
release_lock 2
echo "chain end $(stamp)"
