#!/bin/zsh
# A3: a second, short hold for the noise-floor control (a3_gpu_noise.py at -ub 256). Waits for the first A3 hold
# to finish (its cleanup removes every A3 line), then appends A3 to the queue and takes the lock under the same rules.
set -u
HERE=$HOME/code/inkling-contrib/package/k3-llamacpp-prompt-cache/review
K3=$HOME/code/inkling-contrib/upstream/llama.cpp-k3
LOCK=$HOME/code/inkling-contrib/bench/GPU_LOCK
QUEUE=$HOME/code/inkling-contrib/bench/GPU_QUEUE
while kill -0 65265 2>/dev/null; do sleep 15; done
echo "[$(date +%T)] first hold finished; queueing A3 again"
grep -qx "A3" $QUEUE 2>/dev/null || echo "A3" >> $QUEUE
driver_pid=""
cleanup() {
  if [[ -n $driver_pid ]] && kill -0 $driver_pid 2>/dev/null; then
    kill -TERM $driver_pid 2>/dev/null; wait $driver_pid 2>/dev/null
  fi
  [[ -f $LOCK && "$(awk '{print $1}' $LOCK)" == "$$" ]] && rm -f $LOCK
  [[ -f $QUEUE ]] && sed -i '' '/^A3$/d' $QUEUE
  echo "[$(date +%T)] A3 cleanup (hold 2): driver stopped, lock released, A3 removed from GPU_QUEUE"
}
trap cleanup EXIT INT TERM HUP
while true; do
  head=$(head -1 $QUEUE 2>/dev/null)
  holder=$(awk '{print $1}' $LOCK 2>/dev/null)
  big=$(ps -axo rss= | awk '$1 > 20*1048576' | wc -l | tr -d ' ')
  if [[ "$head" == "A3" ]] && { [[ ! -f $LOCK ]] || ! kill -0 "$holder" 2>/dev/null; } && [[ "$big" == "0" ]]; then
    break
  fi
  echo "[$(date +%T)] waiting (hold 2): head=$head lock_holder=${holder:-none} big_procs=$big"
  sleep 20
done
echo "$$ $(date -u +%FT%TZ) A3-review noise floor (short hold, cap 900s)" > $LOCK
echo "[$(date +%T)] lock taken by $$ (hold 2)"
t_start=$(date +%s)
python3 $HERE/a3_gpu_noise.py --bin-dir $K3/build-k3/bin --tag a3-noise-ub256 --ub 256 --n-predict 64 --out $HERE > $HERE/a3-noise-ub256-run.log 2>&1 &
driver_pid=$!
( sleep 900; kill -TERM $driver_pid 2>/dev/null ) &
wd=$!
wait $driver_pid
rc=$?
driver_pid=""
kill $wd 2>/dev/null
echo "[$(date +%T)] noise driver exit $rc after $(( ($(date +%s) - t_start) / 60 )) min"
