#!/bin/zsh
# Second continuation of K2's hold 2: re-take the lock the instant the first continuation
# releases it (K2 is still the queue head), run the real-width decode profile, release.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/K2
B=~/code/inkling-contrib/bench
R=$B/results
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
until grep -q "hold 2 released" $R/K2-hold2.log; do sleep 0.2; done
if [ -f $B/GPU_LOCK ]; then echo "[$(stamp)] someone else took the lock: $(cat $B/GPU_LOCK); not profiling"; exit 0; fi
echo "$$ $(date +%FT%T) K2-chain-2c (decode profile, end of hold 2)" > $B/GPU_LOCK
grep -qx K2 $B/GPU_QUEUE || { { echo K2; cat $B/GPU_QUEUE; } > $B/GPU_QUEUE.new && mv $B/GPU_QUEUE.new $B/GPU_QUEUE; }
echo "[$(stamp)] hold 2c started (lock pid $$); queue: $(tr '\n' ' ' < $B/GPU_QUEUE)"
run python profile_decode.py > $R/K2-profile-decode.log 2>&1; echo "profile rc=$? $(stamp)"
rm -f $B/GPU_LOCK
sed -i '' '/^K2$/d' $B/GPU_QUEUE
echo "[$(stamp)] hold 2c released; queue now: $(tr '\n' ' ' < $B/GPU_QUEUE)"
