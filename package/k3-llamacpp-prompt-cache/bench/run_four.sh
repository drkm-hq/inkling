#!/bin/zsh
# K3: the server runs (before/after at -ub 512 in full, the field's -ub 256 before-case without the uncached
# references); run under bench/baseline/with_gpu_lock.sh (see run_all.sh). GPU queue protocol: start only while
# K3 is the first line of bench/GPU_QUEUE; on exit (also when killed) stop the driver and its server by PID,
# release bench/GPU_LOCK and remove the K3 line from the queue.
set -u
HERE=$HOME/code/inkling-contrib/package/k3-llamacpp-prompt-cache/bench
K3=$HOME/code/inkling-contrib/upstream/llama.cpp-k3
LOCK=$HOME/code/inkling-contrib/bench/GPU_LOCK
QUEUE=$HOME/code/inkling-contrib/bench/GPU_QUEUE
mkdir -p $HERE/results
while [[ -f $QUEUE && "$(head -1 $QUEUE)" != "K3" ]]; do
  echo "[$(date +%T)] waiting: GPU_QUEUE head is $(head -1 $QUEUE)"; sleep 30
done
driver_pid=""
cleanup() {
  # the driver (bench_prompt_cache.py) stops the llama-server it started, by PID, on SIGTERM
  if [[ -n $driver_pid ]] && kill -0 $driver_pid 2>/dev/null; then
    kill -TERM $driver_pid 2>/dev/null; wait $driver_pid 2>/dev/null
  fi
  [[ -f $LOCK && "$(awk '{print $1}' $LOCK)" == "$PPID" ]] && rm -f $LOCK
  [[ -f $QUEUE ]] && sed -i '' '/^K3$/d' $QUEUE
  echo "[$(date +%T)] K3 cleanup: driver stopped, lock released, K3 removed from GPU_QUEUE"
}
trap cleanup EXIT INT TERM HUP
echo "[$(date +%T)] starting the runs (lock holder pid $PPID)"
t_start=$(date +%s)
for cfg in "before build-k3-baseline 512 " "after build-k3 512 " "before build-k3-baseline 256 --quick"; do
  set -- ${=cfg}
  echo "[$(date +%T)] === $1 ub=$3 ($2) ${4:-}"
  python3 $HERE/bench_prompt_cache.py --bin-dir $K3/$2/bin --tag $1-ub$3 --ub $3 --n-predict 64 --out $HERE/results ${4:-} >> $HERE/results/$1-ub$3-run.log 2>&1 &
  driver_pid=$!
  wait $driver_pid
  driver_pid=""
  tail -3 $HERE/results/$1-ub$3-run.log
  echo "[$(date +%T)] elapsed under the lock: $(( ($(date +%s) - t_start) / 60 )) min"
  sleep 3
done
echo "[$(date +%T)] all runs done"
