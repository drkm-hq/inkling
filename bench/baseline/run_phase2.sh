#!/bin/zsh
# Phase 2 (after the MLX kv8/CLI chain), each run gated on a quiet GPU (other jobs run test-backend-ops etc.):
# (a) MLX re-run of the short contexts with p_short repeated as a drift check, (b) MLX kv8 re-run,
# (c) MLX p_64k with a 512-token prefill chunk, (d) llama-bench and llama-server once the GGUF is complete.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
B=baseline
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until grep -q 'chain end' results/mlx-chain.log 2>/dev/null; do sleep 20; done
log "phase2 start load=$(sysctl -n vm.loadavg)"
/usr/bin/time -l python $B/bench_mlx.py --tag rerun --prompts p_short,p_2k,p_8k,p_short --repeats 3 --max-tokens 128 --quiet-wait 600 > results/mlx-rerun.log 2>&1
log "rerun done rc=$?"
/usr/bin/time -l python $B/bench_mlx.py --tag kv8-rerun --prompts p_short,p_8k --repeats 3 --max-tokens 128 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0 --quiet-wait 600 > results/mlx-kv8-rerun.log 2>&1
log "kv8-rerun done rc=$?"
/usr/bin/time -l python $B/bench_mlx.py --tag p64k --prompts p_32k,p_64k --repeats 3 --max-tokens 128 --prefill-step-size 512 --mem-limit-gb 118 --time-limit 1200 --warmup-tokens 32 --quiet-wait 600 > results/mlx-p64k.log 2>&1
log "p64k done rc=$?"
until $B/check_gguf.sh | grep -q COMPLETE; do log "waiting for GGUF: $($B/check_gguf.sh)"; sleep 60; done
log "GGUF complete: $($B/check_gguf.sh)"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 0 0,4096,16384,32768 3 > results/llama-bench-fa0.log 2>&1
log "llama-bench fa0 done rc=$?"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 1 0,4096 3 > results/llama-bench-fa1.log 2>&1
log "llama-bench fa1 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 0 --tag fa0 --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > results/llama-server-fa0.client.log 2>&1
log "server fa0 done rc=$?"
$B/with_gpu_lock.sh python $B/bench_llama_server.py --fa 1 --tag fa1 --prompts p_short,p_2k,p_8k --repeats 3 --single p_32k --timeout 1200 > results/llama-server-fa1.client.log 2>&1
log "server fa1 done rc=$?"
$B/with_gpu_lock.sh $B/run_llama_bench.sh 1 16384 1 > results/llama-bench-fa1-d16k.log 2>&1
log "llama-bench fa1 d16k done rc=$?"
log "phase2 end"
