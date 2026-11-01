#!/bin/zsh
# A3: one GPU hold for a3_gpu_run.py on the fixed build. Takes bench/GPU_LOCK only when A3 heads bench/GPU_QUEUE,
# the lock is absent or its holder is dead, and no process is above 20 GB RSS. Hold capped at 24 minutes.
# On exit: the driver's server is stopped by PID (the driver does it), the lock is removed, A3 leaves the queue.
set -u
HERE=$HOME/code/inkling-contrib/package/k3-llamacpp-prompt-cache/review
K3=$HOME/code/inkling-contrib/upstream/llama.cpp-k3
LOCK=$HOME/code/inkling-contrib/bench/GPU_LOCK
QUEUE=$HOME/code/inkling-contrib/bench/GPU_QUEUE
driver_pid=""
cleanup() {
  if [[ -n $driver_pid ]] && kill -0 $driver_pid 2>/dev/null; then
    kill -TERM $driver_pid 2>/dev/null; wait $driver_pid 2>/dev/null
  fi
  [[ -f $LOCK && "$(awk '{print $1}' $LOCK)" == "$$" ]] && rm -f $LOCK
  [[ -f $QUEUE ]] && sed -i '' '/^A3$/d' $QUEUE
  echo "[$(date +%T)] A3 cleanup: driver stopped, lock released, A3 removed from GPU_QUEUE"
}
trap cleanup EXIT INT TERM HUP
while true; do
  head=$(head -1 $QUEUE 2>/dev/null)
  holder=$(awk '{print $1}' $LOCK 2>/dev/null)
  big=$(ps -axo rss= | awk '$1 > 20*1048576' | wc -l | tr -d ' ')
  if [[ "$head" == "A3" ]] && { [[ ! -f $LOCK ]] || ! kill -0 "$holder" 2>/dev/null; } && [[ "$big" == "0" ]]; then
    break
  fi
  echo "[$(date +%T)] waiting: head=$head lock_holder=${holder:-none} big_procs=$big"
  sleep 20
done
echo "$$ $(date -u +%FT%TZ) A3-review (one hold, cap 1440s)" > $LOCK
echo "[$(date +%T)] lock taken by $$"
t_start=$(date +%s)
python3 $HERE/a3_gpu_run.py --bin-dir $K3/build-k3/bin --tag a3-fixed-ub512 --ub 512 --n-predict 64 --out $HERE > $HERE/a3-fixed-ub512-run.log 2>&1 &
driver_pid=$!
# watchdog: 24 minutes
( sleep 1440; kill -TERM $driver_pid 2>/dev/null ) &
wd=$!
wait $driver_pid
rc=$?
driver_pid=""
kill $wd 2>/dev/null
echo "[$(date +%T)] driver exit $rc after $(( ($(date +%s) - t_start) / 60 )) min"
tail -3 $HERE/a3-fixed-ub512-run.log
