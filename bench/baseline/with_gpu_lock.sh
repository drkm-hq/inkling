#!/bin/zsh
# Usage: with_gpu_lock.sh <command...>
# Waits until no other big-model process (RSS > 20 GB) is running and bench/GPU_LOCK is free (or stale), then
# waits up to QUIET_WAIT s (default 600) for other GPU-suspect processes (test-backend-ops, llama-*, micro-benchmarks)
# to finish, holds the lock with this shell's PID while <command> runs, and logs the competing processes it saw.
LOCK=$HOME/code/inkling-contrib/bench/GPU_LOCK
QUIET_WAIT=${QUIET_WAIT:-600}
[[ -f $HOME/code/inkling-contrib/bench/baseline/QUIET_WAIT ]] && QUIET_WAIT=$(awk '{print $1}' $HOME/code/inkling-contrib/bench/baseline/QUIET_WAIT)
RE='test-backend-ops|llama-(bench|server|cli|perplexity)|fa_bench|sched_bench|verify_dense|bench_(sdpa|moe|layer|misc)|kernels_bench|layer_bench|bw_probe|kv_probe|sdpa_probe|mtp_probe|diag_loop|mlx_vlm|mlx_lm|bench_mlx|load_mlx_chunked'
while :; do
  if [[ -f $LOCK ]]; then
    pid=$(awk '{print $1}' "$LOCK")
    if kill -0 "$pid" 2>/dev/null; then echo "[$(date +%T)] waiting: GPU_LOCK held by pid $pid"; sleep 30; continue; fi
    echo "[$(date +%T)] stale GPU_LOCK (pid $pid) removed"; rm -f "$LOCK"
  fi
  big=$(ps -axo pid,rss,command | awk -v me=$$ 'NR>1 && $2>20000000 && $1!=me {print $1, int($2/1048576) "GiB", $3}')
  if [[ -n $big ]]; then echo "[$(date +%T)] waiting: big process: $big"; sleep 30; continue; fi
  echo "$$ $(date +%FT%T) $1" > "$LOCK"; echo "[$(date +%T)] GPU_LOCK acquired by $$"; break
done
trap 'rm -f "$LOCK"; echo "[$(date +%T)] GPU_LOCK released"' EXIT INT TERM
t0=$(date +%s)
while :; do
  comp=$(ps -axo pid,pcpu,etime,command | awk -v me=$$ '$1!=me' | grep -E "$RE" | grep -v -E "grep|with_gpu_lock" | cut -c1-120)
  if [[ -z $comp ]]; then break; fi
  if (( $(date +%s) - t0 > QUIET_WAIT )); then echo "[$(date +%T)] quiet-wait exhausted; proceeding alongside: $comp"; break; fi
  echo "[$(date +%T)] waiting for GPU-suspect processes: $comp"; sleep 15
done
echo "[$(date +%T)] competing at start: $(ps -axo pid,pcpu,etime,command | grep -E "$RE" | grep -v -E 'grep|with_gpu_lock' | cut -c1-100 | tr '\n' ';')"
"$@"
