#!/bin/zsh
# Phase 4 orchestrator (priority change): after the running kv8-rerun releases bench/GPU_LOCK, yield
# the GPU to the other benchmark job (wait up to 10 min for it to take the lock, then until it releases it), then run the rest of
# the chain sequentially. Every step acquires the lock itself and releases it on exit; quiet cap 120 s (QUIET_WAIT).
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
B=baseline
LOCK=$HOME/code/inkling-contrib/bench/GPU_LOCK
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mlx() { local tag=$1; shift; /usr/bin/time -l python $B/bench_mlx.py --tag $tag --max-tokens 128 --quiet-wait 600 "$@" > results/mlx-$tag.log 2>&1; log "$tag done rc=$?"; }
lock_pid() { [[ -f $LOCK ]] && awk '{print $1}' "$LOCK" || echo ""; }
lock_alive() { local p=$(lock_pid); [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
log "phase4 start; waiting for my kv8-rerun (pid 63289) to finish"
while kill -0 63289 2>/dev/null; do sleep 10; done
log "kv8-rerun finished; GPU_LOCK released; yielding to K1 for up to 10 min"
t0=$(date +%s); taken=0
while (( $(date +%s) - t0 < 600 )); do
  if lock_alive; then taken=1; log "K1 took the lock: $(cat $LOCK)"; break; fi
  sleep 5
done
if (( taken )); then
  while lock_alive; do sleep 10; done
  log "K1 released the lock (file gone or pid dead) at $(date -u +%FT%TZ)"
else
  log "no one took the lock within 10 min; continuing"
fi
mlx kv8tq --prompts p_short,p_8k --repeats 3 --kv-bits 8 --kv-quant-scheme turboquant --quantized-kv-start 0
mlx kv8h  --prompts p_short,p_8k --repeats 3 --kv-bits 8 --kv-key-bits 8 --kv-value-bits 4 --kv-group-size 64 --quantized-kv-start 0
$B/with_gpu_lock.sh /usr/bin/time -l python -m mlx_vlm.generate --model "$MODEL" --prompt "$(cat prompts/p_short.txt)" --max-tokens 128 --temperature 0 --verbose > results/mlx-cli-p_short.log 2>&1; log "cli (checkout, chunked loader) done rc=$?"
$B/with_gpu_lock.sh /usr/bin/time -l ~/.venvs/inkling-base/bin/python -m mlx_vlm.generate --model "$MODEL" --prompt "$(cat prompts/p_short.txt)" --max-tokens 128 --temperature 0 --verbose > results/mlx-cli-wheel-p_short.log 2>&1; log "cli (stock 0.7.11 wheel) done rc=$?"
mlx p64k  --prompts p_32k,p_64k --repeats 3 --prefill-step-size 512 --mem-limit-gb 118 --time-limit 1200 --warmup-tokens 32
mlx smoke --prompts p_short --repeats 3 --smoke
log "MLX phase ends; GPU_LOCK free until the llama.cpp phase takes it"
until grep -q 'BUILD_OK' results/llama-base-build.log 2>/dev/null; do log "waiting for the llama.cpp-base build"; sleep 30; done
$B/with_gpu_lock.sh $B/run_llama_bench.sh 0 0,4096,16384,32768 3 > results/llama-bench-fa0.log 2>&1; log "llama-bench fa0 done rc=$?"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 1 0,4096 3 > results/llama-bench-fa1.log 2>&1; log "llama-bench fa1 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 0 --tag fa0 --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > results/llama-server-fa0.client.log 2>&1; log "server fa0 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 1 --tag fa1 --prompts p_short,p_2k --repeats 3 --single p_8k --timeout 300 > results/llama-server-fa1.client.log 2>&1; log "server fa1 done rc=$?"
log "phase4 end"
