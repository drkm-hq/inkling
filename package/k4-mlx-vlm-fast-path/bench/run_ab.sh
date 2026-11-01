#!/bin/zsh
# Second K4 hold (queued at the END of GPU_QUEUE after the first hold ended):
# an interleaved A/B so that machine-load drift cancels out of the decode
# comparison, which is CPU-dispatch bound. Order: before, after, before, after
# on p_short + p_32k, 2 timed runs each (~3.5 min per process). Same queue
# protocol and cleanup as run_all.sh; budget 20 minutes.
set -u
HERE=${0:A:h}
CLONE=~/code/inkling-contrib/upstream/mlx-vlm
BASE=~/.venvs/inkling-base/bin/python
DEV=~/.venvs/inkling/bin/python
DRIVER=$HERE/equiv_bench.py
LOGS=$HERE/logs
LOCK=~/code/inkling-contrib/bench/GPU_LOCK
QUEUE=~/code/inkling-contrib/bench/GPU_QUEUE
ME=K4
BUDGET=${BUDGET:-2100}
mkdir -p $LOGS $HERE/results
CHILD=""
HOLD_START=""

note() { echo "=== $(date +%T) [ab] $*" | tee -a $LOGS/run_all.log; }

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
echo "$$ $(date -u +%FT%TZ) k4-mlx-vlm-fast-path (A/B hold, budget ${BUDGET}s)" > $LOCK
HOLD_START=$(date +%s)
note "hold acquired by pid $$ (queue head: $(head -1 $QUEUE)); budget ${BUDGET}s"
elapsed() { echo $(( $(date +%s) - HOLD_START )); }

run() {  # tag python estimate-seconds extra-args...
  local tag=$1 py=$2 est=$3; shift 3
  if (( $(elapsed) + est > BUDGET )); then
    note "skip $tag: elapsed $(elapsed)s + estimate ${est}s exceeds budget ${BUDGET}s"
    return 1
  fi
  note "start $tag ($py) $*"
  $py $DRIVER --no-lock --tag $tag "$@" > $LOGS/$tag.log 2>&1 &
  CHILD=$!
  wait $CHILD; local rc=$?
  CHILD=""
  note "end $tag rc=$rc (elapsed $(elapsed)s)"
  return 0
}

# 1. teacher-forced: the baseline's 128 tokens fed to every run (per-token
#    logprobs, full fp32 log-softmax matrices, KL against the reference)
run tf-before-2048 $BASE 150 --teacher-forced $HERE/results/before.jsonl --prefill-step-size 2048 --max-tokens 128
run tf-after       $DEV  150 --teacher-forced $HERE/results/before.jsonl --prefill-step-size 2048 --max-tokens 128 --tf-reference tf-before-2048
run tf-before-512  $BASE 180 --teacher-forced $HERE/results/before.jsonl --prefill-step-size 512  --max-tokens 128 --tf-reference tf-before-2048
run tf-after-kv8   $DEV  160 --teacher-forced $HERE/results/before.jsonl --prefill-step-size 2048 --max-tokens 128 --tf-reference tf-before-2048 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
# 2. interleaved decode A/B (machine-load drift cancels), then kv8 generation
run ab-before-1 $BASE 230 --prompts p_short,p_32k --repeats 2 --max-tokens 128
run ab-after-1  $DEV  220 --prompts p_short,p_32k --repeats 2 --max-tokens 128
run after-kv8   $DEV  160 --prompts p_short,p_8k --repeats 2 --max-tokens 128 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
run ab-before-2 $BASE 230 --prompts p_short,p_32k --repeats 2 --max-tokens 128
run ab-after-2  $DEV  220 --prompts p_short,p_32k --repeats 2 --max-tokens 128
note "A/B done after $(elapsed)s"
