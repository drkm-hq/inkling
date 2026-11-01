#!/bin/zsh
# K4 before/after runs in ONE GPU hold, following the lead's queue
# protocol: take ~/code/inkling-contrib/bench/GPU_LOCK only when it is free
# (absent or dead PID), no other process is above 20 GB RSS, and K4 is the
# first line of ~/code/inkling-contrib/bench/GPU_QUEUE. Every driver process
# runs with --no-lock inside the hold and exits (freeing the model) before the
# next starts. Runs are ordered by priority and skipped when the remaining
# budget (40 min) cannot fit their estimate. On exit the lock is deleted and
# K4 is removed from the queue. Run under nohup; progress in logs/run_all.log.
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
BUDGET=${BUDGET:-2400}
PROMPTS=${PROMPTS:-p_short,p_2k,p_8k,p_32k}
mkdir -p $LOGS $HERE/results
HEAD_SHA=$(git -C $CLONE rev-parse --short HEAD)
CHILD=""
HOLD_START=""

note() { echo "=== $(date +%T) $*" | tee -a $LOGS/run_all.log; }

cleanup() {
  # Stop our own model process (by PID, never by name), reap it so its memory
  # is gone, and only then release the lock.
  if [[ -n "$CHILD" ]]; then
    kill $CHILD 2>/dev/null
    for i in {1..10}; do kill -0 $CHILD 2>/dev/null || break; sleep 1; done
    kill -9 $CHILD 2>/dev/null; wait $CHILD 2>/dev/null
    note "stopped driver pid $CHILD"
  fi
  git -C $CLONE checkout -q inkling-fast-path 2>/dev/null
  if [[ -f $LOCK ]] && [[ "$(cut -d' ' -f1 $LOCK)" == "$$" ]]; then rm -f $LOCK; note "lock released"; fi
  if [[ -n "$HOLD_START" ]]; then  # keep the queue place if the hold never started
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
no_big_process() { [[ -z "$(ps -axo pid=,rss= | awk '$2>20000000')" ]]; }  # no header: pid=,rss=
my_turn() { [[ "$(head -1 $QUEUE 2>/dev/null)" == "$ME" ]]; }

note "waiting for the GPU: queue [$(tr '\n' ' ' < $QUEUE)], lock: $(cat $LOCK 2>/dev/null || echo none)"
while ! { my_turn && lock_free && no_big_process; }; do sleep 30; done
echo "$$ $(date -u +%FT%TZ) k4-mlx-vlm-fast-path (one hold, budget ${BUDGET}s)" > $LOCK
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

checkout() {  # sha: the editable install follows the working tree
  git -C $CLONE checkout -q $1 && note "checked out $(git -C $CLONE rev-parse --short HEAD)"
}

# 1. the released wheel and the whole branch, same prompts (the equivalence pair)
run before $BASE 450 --prompts $PROMPTS --repeats 3 --max-tokens 128
run after  $DEV  430 --prompts $PROMPTS --repeats 3 --max-tokens 128
# 2. the eager loader on the branch (item 1): the 82 GiB load that trips the watchdog on the wheel
run after-eager-load $DEV 70 --eager-load --prompts p_short --repeats 1 --max-tokens 32
# 3. 64K context on the branch (bounded caches); the wheel's unbounded caches plus tau
#    temporaries exceed 128 GB there (32K already peaks at 104.9 GiB), not attempted
run after-64k $DEV 330 --prompts p_64k --repeats 2 --max-tokens 128 --warmup-prompt p_short
# 4. per-item attribution on the branch's commit chain (short + 32K)
if checkout 0abf8bd; then  # loader + window caches only
  run after-window-only $DEV 220 --prompts p_short,p_32k --repeats 2 --max-tokens 128
fi
if checkout 21e690a; then  # + strided sconv, tau skip
  run after-kernels $DEV 220 --prompts p_short,p_32k --repeats 2 --max-tokens 128
fi
checkout inkling-fast-path
note "restored branch at $(git -C $CLONE rev-parse --short HEAD) (expected $HEAD_SHA)"
# 5. kv-bits on the branch (item 6): global layers quantized
run after-kv8 $DEV 150 --prompts p_short,p_8k --repeats 2 --max-tokens 128 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
note "all done after $(elapsed)s"
