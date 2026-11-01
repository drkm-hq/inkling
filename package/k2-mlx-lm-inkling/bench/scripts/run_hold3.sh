#!/bin/zsh
# K2 hold 3 (budget 40 min), in this order:
#  1. fold build vs dense-shared-expert build, same hold, p_short + p_8k, 3 reps each
#  2. real-width decode profiles (mlx-lm dense build, then mlx-vlm 0.7.11), the forced
#     equivalence re-check of the dense build, and its greedy run (teacher for attn8)
#  3. the 8-bit-attention variant on p_short (build, greedy + forced vs bf16 attention, bench)
#  4. the stock mlx_lm.generate CLI transcript (one-shot load), with a lazy-load fallback
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
echo "hold3 script start $(stamp)"
run python gpulock.py wait K2 || exit 1
echo "$$ $(date +%FT%T) K2-chain-3 (budget 40 min)" > $B/GPU_LOCK
T0=$(date +%s)
echo "[$(stamp)] hold 3 started (lock pid $$); queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
left() { echo $(( 2400 - ($(date +%s) - T0) )); }
# 1. fold vs dense
run python bench_mlx_lm.py --tag mlxlm-2bit-fold-h3 --no-lock --model-type inkling_mm_model_fold --prompts p_short,p_8k --repeats 3 > $R/K2-bench-mlxlm-fold-h3.log 2>&1; echo "bench fold rc=$? $(stamp) left $(left)s"
run python bench_mlx_lm.py --tag mlxlm-2bit-dense-h3 --no-lock --prompts p_short,p_8k --repeats 3 > $R/K2-bench-mlxlm-dense-h3.log 2>&1; echo "bench dense rc=$? $(stamp) left $(left)s"
# 2. profiles, forced re-check, dense greedy (teacher for attn8)
run python profile_decode.py > $R/K2-profile-decode.log 2>&1; echo "profile mlx-lm rc=$? $(stamp) left $(left)s"
run ~/.venvs/inkling-base/bin/python profile_decode_vlm.py > $R/K2-profile-decode-vlm.log 2>&1; echo "profile mlx-vlm rc=$? $(stamp) left $(left)s"
[ -f $R/K2-equiv-mlx_lm-forced.json ] && mv $R/K2-equiv-mlx_lm-forced.json $R/K2-equiv-mlx_lm-forced-fold.json
run python equiv.py --impl mlx_lm --no-lock --teacher $R/K2-equiv-mlx_vlm.json --router-audit --out $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-mlx_lm-forced.log 2>&1; echo "forced dense rc=$? $(stamp) left $(left)s"
run python equiv_forced_diff.py $R/K2-equiv-mlx_vlm.json $R/K2-equiv-mlx_lm-forced.json > $R/K2-equiv-forced-diff.txt 2>&1
run python equiv.py --impl mlx_lm --no-lock --prompts p_short --chat-efforts "" --out $R/K2-equiv-mlx_lm-dense.json > $R/K2-equiv-mlx_lm-dense.log 2>&1; echo "dense greedy rc=$? $(stamp) left $(left)s"
run python equiv_report.py > $R/K2-equivalence.md 2>&1
# 3. attn8 on p_short
if [ ! -d $ATTN8 ]; then
  mkdir -p ~/models/inkling-small
  run python make_attn8.py --no-lock --out $ATTN8 > $R/K2-make-attn8.log 2>&1; echo "make attn8 rc=$? $(stamp) left $(left)s"
fi
run python equiv.py --impl mlx_lm --no-lock --model $ATTN8 --prompts p_short --chat-efforts "" --out $R/K2-equiv-attn8.json > $R/K2-equiv-attn8.log 2>&1; echo "attn8 greedy rc=$? $(stamp)"
run python equiv.py --impl mlx_lm --no-lock --model $ATTN8 --prompts p_short --chat-efforts "" --teacher $R/K2-equiv-mlx_lm-dense.json --out $R/K2-equiv-attn8-forced.json > $R/K2-equiv-attn8-forced.log 2>&1; echo "attn8 forced rc=$? $(stamp) left $(left)s"
run python equiv_forced_diff.py $R/K2-equiv-mlx_lm-dense.json $R/K2-equiv-attn8-forced.json > $R/K2-equiv-forced-diff-attn8.txt 2>&1; cat $R/K2-equiv-forced-diff-attn8.txt
run python bench_mlx_lm.py --tag mlxlm-attn8-h3 --no-lock --model $ATTN8 --prompts p_short --repeats 3 > $R/K2-bench-mlxlm-attn8-h3.log 2>&1; echo "bench attn8 rc=$? $(stamp) left $(left)s"
# 4. the stock CLI (one-shot load); fallback transcript with a lazy load if it fails
{ echo "\$ mlx_lm.generate --model $MODEL -p 'The secret to baking a good cake is' -m 200"
  run /usr/bin/time -l python -m mlx_lm generate --model "$MODEL" -p "The secret to baking a good cake is" -m 200 2>&1
  echo "[exit code $?]"; } > $R/K2-cli-generate.log 2>&1
grep -q "Peak memory" $R/K2-cli-generate.log || run python cli_fallback.py "$MODEL" > $R/K2-cli-generate-lazy.log 2>&1
echo "cli done $(stamp) left $(left)s"
rm -f $B/GPU_LOCK
sed -i '' '/^K2$/d' $B/GPU_QUEUE
echo "[$(stamp)] hold 3 released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
