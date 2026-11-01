#!/bin/zsh
# Phase 3 (corrections applied): remaining MLX steps, then llama.cpp on the pristine
# upstream/llama.cpp-base build with the bounded -fa matrix. Every step acquires bench/GPU_LOCK itself
# (bench_mlx.py self-locks; others via with_gpu_lock.sh), so the lock is free between steps and other
# agents (K1) can interleave at any boundary.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
B=baseline
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/ | head -1)
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mlx() { local tag=$1; shift; /usr/bin/time -l python $B/bench_mlx.py --tag $tag --max-tokens 128 --quiet-wait 600 "$@" > results/mlx-$tag.log 2>&1; log "$tag done rc=$?"; }
log "phase3 start load=$(sysctl -n vm.loadavg)"
mlx kv8h      --prompts p_short,p_8k --repeats 3 --kv-bits 8 --kv-key-bits 8 --kv-value-bits 4 --kv-group-size 64 --quantized-kv-start 0
mlx kv8tq     --prompts p_short,p_8k --repeats 3 --kv-bits 8 --kv-quant-scheme turboquant --quantized-kv-start 0
$B/with_gpu_lock.sh /usr/bin/time -l python -m mlx_vlm.generate --model "$MODEL" --prompt "$(cat prompts/p_short.txt)" --max-tokens 128 --temperature 0 --verbose > results/mlx-cli-p_short.log 2>&1
log "cli done rc=$?"
mlx p64k      --prompts p_32k,p_64k --repeats 3 --prefill-step-size 512 --mem-limit-gb 118 --time-limit 1200 --warmup-tokens 32
log "MLX phase ends; GPU_LOCK is free until the llama.cpp phase takes it"
until grep -q 'BUILD_OK' results/llama-base-build.log 2>/dev/null; do
  if grep -q 'BUILD_FAIL' results/llama-base-build.log 2>/dev/null; then log "llama.cpp-base build FAILED"; exit 1; fi
  log "waiting for the llama.cpp-base build"; sleep 30
done
log "llama.cpp-base build ready: $(grep BUILD_OK results/llama-base-build.log)"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 0 0,4096,16384,32768 3 > results/llama-bench-fa0.log 2>&1; log "llama-bench fa0 done rc=$?"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 1 0,4096 3 > results/llama-bench-fa1.log 2>&1; log "llama-bench fa1 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 0 --tag fa0 --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > results/llama-server-fa0.client.log 2>&1; log "server fa0 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 1 --tag fa1 --prompts p_short,p_2k --repeats 3 --single p_8k --timeout 300 > results/llama-server-fa1.client.log 2>&1; log "server fa1 done rc=$?"
log "phase3 end"
