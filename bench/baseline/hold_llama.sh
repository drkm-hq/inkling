#!/bin/zsh
# Hold 1: the llama.cpp phase on the pristine build, bounded as instructed (-fa 1 only at depths 0/4096 and only
# p_short/p_2k (+ one p_8k capped at 5 min) on the server). No inner locking: gpu_turn.sh holds the lock.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
B=baseline
log() { echo "[$(date -u +%FT%TZ)] $*"; }
on_signal() { trap - INT TERM; for c in $(pgrep -P $$ 2>/dev/null); do kill -TERM "$c" 2>/dev/null; done; log "hold interrupted; children signalled"; exit 143; }
trap on_signal INT TERM
log "llama hold start load=$(sysctl -n vm.loadavg)"
grep -q BUILD_OK results/llama-base-build.log || { log "no pristine build"; exit 1; }
$B/run_llama_bench.sh 0 0,4096,16384,32768 3 > results/llama-bench-fa0.log 2>&1; log "llama-bench fa0 done rc=$?"
$B/run_llama_bench.sh 1 0,4096 3 > results/llama-bench-fa1.log 2>&1; log "llama-bench fa1 done rc=$?"
python $B/bench_llama_server.py --fa 0 --tag fa0 --prompts p_short,p_2k,p_8k,p_32k --repeats 3 > results/llama-server-fa0.client.log 2>&1; log "server fa0 done rc=$?"
python $B/bench_llama_server.py --fa 1 --tag fa1 --prompts p_short,p_2k --repeats 3 --single p_8k --timeout 300 > results/llama-server-fa1.client.log 2>&1; log "server fa1 done rc=$?"
log "llama hold end"
