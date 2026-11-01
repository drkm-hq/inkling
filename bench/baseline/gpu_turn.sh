#!/bin/zsh
# Usage: gpu_turn.sh <label> <command...>
# GPU queue protocol: append M1 to bench/GPU_QUEUE if absent; wait until bench/GPU_LOCK is free (absent or dead
# PID), no other process is above 20 GB RSS, and M1 is the first line of GPU_QUEUE; then hold the lock with this
# PID while <command> runs. On exit (normal, INT or TERM): kill the process tree this wrapper started (by PID,
# descendants of its own child only; never by name), delete the lock, remove M1 from the queue.
BENCH=$HOME/code/inkling-contrib/bench; LOCK=$BENCH/GPU_LOCK; QUEUE=$BENCH/GPU_QUEUE; ME=M1
label=$1; shift
grep -qx "$ME" "$QUEUE" 2>/dev/null || echo "$ME" >> "$QUEUE"
echo "[$(date +%T)] queued as $ME for '$label'; queue: $(tr '\n' ' ' < $QUEUE)"
while :; do
  head=$(grep -v '^[[:space:]]*$' "$QUEUE" 2>/dev/null | head -1 | tr -d '[:space:]')
  lockfree=1
  if [[ -f $LOCK ]]; then pid=$(awk '{print $1}' "$LOCK"); kill -0 "$pid" 2>/dev/null && lockfree=0; fi
  big=$(ps -axo pid,rss,command | awk -v me=$$ 'NR>1 && $2>20000000 && $1!=me {print $1}' | tr '\n' ',')
  if [[ $head == $ME && $lockfree == 1 && -z $big ]]; then break; fi
  echo "[$(date +%T)] waiting: head=$head lockfree=$lockfree big=${big:-none}"; sleep 30
done
echo "$$ $(date -u +%FT%TZ) M1 $label" > "$LOCK"
echo "[$(date +%T)] M1 holds GPU_LOCK for '$label'"
CHILD=""
kill_tree() {  # kill the descendants of $1 (depth first), then $1 itself; all by PID
  local p=$1 c
  for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -TERM "$p" 2>/dev/null
}
cleanup() {
  trap - EXIT INT TERM
  if [[ -n $CHILD ]] && kill -0 "$CHILD" 2>/dev/null; then
    echo "[$(date +%T)] stopping my process tree under pid $CHILD"; kill_tree "$CHILD"; sleep 5
    for c in $(pgrep -P "$CHILD" 2>/dev/null) "$CHILD"; do kill -KILL "$c" 2>/dev/null; done
  fi
  [[ -f $LOCK && "$(awk '{print $1}' $LOCK)" == "$$" ]] && rm -f "$LOCK"
  sed -i '' "/^${ME}$/d" "$QUEUE"
  echo "[$(date +%T)] M1 released GPU_LOCK and left the queue; queue now: $(tr '\n' ' ' < $QUEUE)"
}
trap cleanup EXIT INT TERM
"$@" &
CHILD=$!
wait $CHILD
