#!/bin/zsh
# A4 GPU hold: before (0.7.11 wheel, ~/.venvs/inkling-base) vs after (eb93183 via
# the a4-review worktree, ~/.venvs/inkling-a4). Lock protocol: take
# bench/GPU_LOCK only when absent or its PID is dead, no process above 20 GB
# RSS, and A4 is the FIRST line of bench/GPU_QUEUE. One driver process per run
# (--no-lock), killed by PID only. Budget 2100 s. Run under nohup.
set -u
HERE=${0:A:h}
A4TREE=~/code/inkling-contrib/upstream/mlx-vlm-a4
BASE=~/.venvs/inkling-base/bin/python
DEV=~/.venvs/inkling-a4/bin/python
DRIVER=$HERE/a4_equiv_bench.py
OUT=$HERE/results
LOGS=$HERE/logs
LOCK=~/code/inkling-contrib/bench/GPU_LOCK
QUEUE=~/code/inkling-contrib/bench/GPU_QUEUE
ME=A4
BUDGET=${BUDGET:-2100}
mkdir -p $LOGS $OUT
CHILD=""
HOLD_START=""

note() { echo "=== $(date +%T) $*" | tee -a $LOGS/a4_hold.log; }

cleanup() {
  if [[ -n "$CHILD" ]]; then
    kill $CHILD 2>/dev/null
    for i in {1..10}; do kill -0 $CHILD 2>/dev/null || break; sleep 1; done
    kill -9 $CHILD 2>/dev/null; wait $CHILD 2>/dev/null
    note "stopped driver pid $CHILD"
  fi
  if [[ -f $LOCK ]] && [[ "$(cut -d' ' -f1 $LOCK)" == "$$" ]]; then rm -f $LOCK; note "lock released"; fi
  if [[ -n "$HOLD_START" ]]; then
    sed -i '' "/^${ME}\$/d" $QUEUE 2>/dev/null
    note "removed $ME from GPU_QUEUE: now [$(tr '\n' ' ' < $QUEUE 2>/dev/null)]"
  fi
}
trap 'cleanup; exit' INT TERM
trap cleanup EXIT

lock_free() {
  [[ ! -f $LOCK ]] && return 0
  local pid=$(cut -d' ' -f1 $LOCK)
  [[ -n "$pid" ]] && kill -0 $pid 2>/dev/null && return 1
  return 0
}
no_big_process() { [[ -z "$(ps -axo pid=,rss= | awk '$2>20000000')" ]]; }
my_turn() { [[ "$(head -1 $QUEUE 2>/dev/null)" == "$ME" ]]; }

note "waiting for the GPU: queue [$(tr '\n' ' ' < $QUEUE)], lock: $(cat $LOCK 2>/dev/null || echo none)"
while ! { my_turn && lock_free && no_big_process; }; do sleep 30; done
echo "$$ $(date -u +%FT%TZ) A4-review (one hold, cap ${BUDGET}s)" > $LOCK
HOLD_START=$(date +%s)
note "hold acquired by pid $$ (queue head: $(head -1 $QUEUE)); budget ${BUDGET}s; load1 $(sysctl -n vm.loadavg)"
note "after tree: $(git -C $A4TREE rev-parse --short HEAD) dirty=[$(git -C $A4TREE status --short | tr '\n' ' ')] diff vs eb93183 in mlx_vlm (non-test): $(git -C $A4TREE diff eb93183 --stat -- mlx_vlm ':!mlx_vlm/tests' | tail -1)"
note "before wheel: $($BASE -W ignore -c 'import mlx_vlm;print(mlx_vlm.__file__)' 2>/dev/null)"

elapsed() { echo $(( $(date +%s) - HOLD_START )); }

run() {  # tag python estimate-seconds extra-args...
  local tag=$1 py=$2 est=$3; shift 3
  if (( $(elapsed) + est > BUDGET )); then
    note "skip $tag: elapsed $(elapsed)s + estimate ${est}s exceeds budget ${BUDGET}s"
    return 1
  fi
  note "start $tag ($py) $* load1 $(sysctl -n vm.loadavg)"
  $py $DRIVER --no-lock --tag $tag --out-dir $OUT "$@" > $LOGS/$tag.log 2>&1 &
  CHILD=$!
  wait $CHILD; local rc=$?
  CHILD=""
  note "end $tag rc=$rc (elapsed $(elapsed)s)"
  return 0
}

TF="--teacher-forced $OUT/a4-before-gen.jsonl --max-tokens 64 --prompts p_short,p_8k"
# 1. the wheel: greedy 64 tokens, min-of-3, on p_short and p_8k (teacher tokens come from here)
run a4-before-gen     $BASE 150 --prompts p_short,p_8k --repeats 3 --max-tokens 64
# 2. teacher-forced reference and the chunking noise floor of the unmodified code
run a4-tf-before-2048 $BASE 90  ${=TF} --prefill-step-size 2048
run a4-tf-before-512  $BASE 100 ${=TF} --prefill-step-size 512  --tf-reference a4-tf-before-2048
# 3. the branch: same generation, then teacher-forced against the wheel
run a4-after-gen      $DEV  150 --prompts p_short,p_8k --repeats 3 --max-tokens 64
run a4-tf-after-2048  $DEV  90  ${=TF} --prefill-step-size 2048 --tf-reference a4-tf-before-2048
# 4. 32K: correctness, peak memory, before and after (3 reps each)
run a4-after-32k      $DEV  300 --prompts p_32k --repeats 3 --max-tokens 64 --warmup-prompt p_short
run a4-before-32k     $BASE 310 --prompts p_32k --repeats 3 --max-tokens 64 --warmup-prompt p_short
# 5. kv-bits 8 on the branch: generation and teacher-forced
run a4-after-kv8      $DEV  120 --prompts p_short,p_8k --repeats 1 --max-tokens 64 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
run a4-tf-after-kv8   $DEV  90  ${=TF} --prefill-step-size 2048 --tf-reference a4-tf-before-2048 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
# 6. stability: a 2048-token generation on the branch (bounded memory, no garbage)
run a4-after-gen2k    $DEV  160 --prompts p_short --repeats 1 --max-tokens 2048
# 7. the eager loader on the branch
run a4-after-eager    $DEV  80  --eager-load --prompts p_short --repeats 1 --max-tokens 16
note "all done after $(elapsed)s"
