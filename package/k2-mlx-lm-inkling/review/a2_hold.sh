#!/bin/zsh
# A2's bounded GPU hold (target <= 35 min) under the bench/GPU_QUEUE protocol:
# waits until A2 heads the queue and the GPU is free, takes the lock with this
# shell's PID, runs the real-model checks in priority order with a deadline
# between steps, and always releases the lock and leaves the queue (trap).
# Every child is started by this script and killed by PID only.
set -u
C=~/code/inkling-contrib
B=$C/bench
R=$C/package/k2-mlx-lm-inkling/review
W=$C/upstream/mlx-lm-a2                       # the reviewed tree (branch a2-review)
PY=~/.venvs/inkling-a2/bin/python             # mlx_lm -> $W
PYBASE=~/.venvs/inkling-base/bin/python       # mlx_vlm 0.7.11 wheel
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
ATTN8=~/models/inkling-small/mlx-2bit-attn8
BUDGET=${BUDGET:-2100}
stamp() { date -u +%FT%TZ; }
child=""
run() { "$@" & child=$!; wait $child; local rc=$?; child=""; return $rc; }
run_bounded() {  # run_bounded <seconds> cmd...
  local limit=$1; shift
  "$@" & local pid=$!; child=$pid; local t=0
  while kill -0 $pid 2>/dev/null; do
    if [ $t -ge $limit ]; then echo "[$(stamp)] timeout ${limit}s: killing pid $pid"; kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null; child=""; return 124; fi
    sleep 5; t=$((t+5))
  done
  wait $pid; local rc=$?; child=""; return $rc
}
cleanup() {
  if [ -n "$child" ]; then kill $child 2>/dev/null; sleep 2; kill -9 $child 2>/dev/null; fi
  if [ -n "${server_pid:-}" ]; then kill $server_pid 2>/dev/null; fi
  if [ "$(cut -d' ' -f1 $B/GPU_LOCK 2>/dev/null)" = "$$" ]; then rm -f $B/GPU_LOCK; echo "[$(stamp)] GPU_LOCK released"; fi
  sed -i '' '/^A2$/d' $B/GPU_QUEUE
  echo "[$(stamp)] A2 left the queue; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
}
trap 'cleanup; exit 1' TERM INT HUP
trap 'cleanup' EXIT
left() { echo $((DEADLINE - $(date +%s))); }
echo "[$(stamp)] A2 hold script start (pid $$); queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
grep -qx A2 $B/GPU_QUEUE || echo A2 >> $B/GPU_QUEUE
cd $B/K2 && run $PY gpulock.py wait A2 || exit 1
echo "$$ $(date +%FT%T) A2-review (bounded ${BUDGET}s)" > $B/GPU_LOCK
START=$(date +%s); DEADLINE=$((START + BUDGET))
echo "[$(stamp)] hold started; load=$(uptime | sed 's/.*load averages: //')"
cd $R
{ echo "inkling head: $(git -C $C/upstream/mlx-lm log -1 --format='%h %ad %s' --date=iso inkling)"
  echo "reviewed tree ($W): $(git -C $W log -1 --format='%h %ad %s' --date=iso)"
  echo "worktree status: $(git -C $W status --short | tr '\n' ' ')"; } | tee $R/hold-branch-state.txt

# 1. mlx-lm (reviewed tree) teacher forced on the mlx-vlm tokens; the p_2k outlier dumps; p_32k; its logprobs saved
run $PY a2_forced.py --impl mlx_lm --runs p_short_raw,p_2k_raw,p_8k_raw --dump "p_2k_raw:45,46,47,48" \
    --dump-dir $R/a2-dumps --save-logprobs $R/a2-logprobs-mlx_lm --p32k --out $R/a2-forced-mlx_lm.json > $R/a2-forced-mlx_lm.log 2>&1
echo "[$(stamp)] mlx_lm forced rc=$? ($(left)s left)"
# 2. mlx-vlm 0.7.11: the same dumps (and p_short as a determinism check of the reference logprobs)
run $PYBASE a2_forced.py --impl mlx_vlm --runs p_short_raw,p_2k_raw --dump "p_2k_raw:45,46,47,48" \
    --dump-dir $R/a2-dumps --out $R/a2-forced-mlx_vlm.json > $R/a2-forced-mlx_vlm.log 2>&1
echo "[$(stamp)] mlx_vlm forced rc=$? ($(left)s left)"
run $PY a2_outlier_compare.py $R/a2-dumps p_2k_raw 45 46 47 48 > $R/a2-outlier-compare.md 2>&1

# 3. user-facing paths: the stock CLI and the server (one-shot loads; bounded)
if [ $(left) -gt 600 ]; then
  { echo "\$ mlx_lm.generate --model $MODEL -p 'The secret to baking a good cake is' -m 64 --temp 0  (bounded 300s)"
    run_bounded 300 /usr/bin/time -l $PY -m mlx_lm generate --model "$MODEL" -p "The secret to baking a good cake is" -m 64 --temp 0 2>&1
    echo "[exit code $?] $(stamp)"; } > $R/a2-cli-generate.log 2>&1
  echo "[$(stamp)] cli done ($(left)s left)"
fi
if [ $(left) -gt 480 ]; then
  { echo "\$ python -m mlx_lm.server --model $MODEL --port 8181 (bounded 300s to load)"
    $PY -m mlx_lm.server --model "$MODEL" --port 8181 --log-level INFO > $R/a2-server-stdout.log 2>&1 & server_pid=$!
    echo "server pid $server_pid"
    t=0; ok=0
    while [ $t -lt 300 ]; do
      if ! kill -0 $server_pid 2>/dev/null; then echo "server exited early"; break; fi
      if curl -s -m 5 http://127.0.0.1:8181/v1/models > /dev/null 2>&1; then ok=1; break; fi
      sleep 5; t=$((t+5))
    done
    echo "ready after ~${t}s: ok=$ok"
    if [ $ok = 1 ]; then
      curl -s -m 180 -X POST http://127.0.0.1:8181/v1/completions -H 'Content-Type: application/json' \
        -d '{"prompt": "The secret to baking a good cake is", "max_tokens": 24, "temperature": 0}'; echo
      curl -s -m 180 -X POST http://127.0.0.1:8181/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages": [{"role": "user", "content": "In one sentence, who narrates Moby-Dick?"}], "max_tokens": 32, "temperature": 0}'; echo
    fi
    kill $server_pid 2>/dev/null; sleep 3; kill -9 $server_pid 2>/dev/null; server_pid=""
    echo "[server stopped] $(stamp)"; echo "--- server stdout/stderr (tail)"; tail -30 $R/a2-server-stdout.log; } > $R/a2-server-smoke.log 2>&1
  echo "[$(stamp)] server smoke done ($(left)s left)"
fi

# 4. like-for-like timing, back to back: K2's mlx-lm bench then M1's mlx-vlm bench (same prompts, 3 reps)
if [ $(left) -gt 900 ]; then
  run $PY $B/K2/bench_mlx_lm.py --tag a2-mlxlm --no-lock --prompts p_short,p_2k,p_8k --repeats 3 --out $R/a2-bench-mlxlm.jsonl > $R/a2-bench-mlxlm.log 2>&1
  echo "[$(stamp)] bench mlx-lm rc=$? ($(left)s left)"
fi
if [ $(left) -gt 720 ]; then
  run $PYBASE $B/baseline/bench_mlx.py --tag a2-mlxvlm --no-lock --prompts p_short,p_2k,p_8k --repeats 3 --max-tokens 128 --out $R/a2-bench-mlxvlm.jsonl > $R/a2-bench-mlxvlm.log 2>&1
  echo "[$(stamp)] bench mlx-vlm rc=$? ($(left)s left)"
fi

# 5. the 8-bit-attention variant, if K2 built it: p_short forced on the same tokens vs the bf16-attention logprobs, and speed
if [ -d $ATTN8 ] && [ $(left) -gt 300 ]; then
  run $PY a2_forced.py --impl mlx_lm --model $ATTN8 --runs p_short_raw --ref-logprobs $R/a2-logprobs-mlx_lm --out $R/a2-forced-attn8.json > $R/a2-forced-attn8.log 2>&1
  echo "[$(stamp)] attn8 forced rc=$? ($(left)s left)"
  if [ $(left) -gt 240 ]; then
    run $PY $B/K2/bench_mlx_lm.py --tag a2-attn8 --no-lock --model $ATTN8 --prompts p_short --repeats 3 --out $R/a2-bench-attn8.jsonl > $R/a2-bench-attn8.log 2>&1
    echo "[$(stamp)] bench attn8 rc=$? ($(left)s left)"
  fi
fi
echo "[$(stamp)] hold done after $(( $(date +%s) - START ))s"
