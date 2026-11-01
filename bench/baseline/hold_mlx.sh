#!/bin/zsh
# Hold 2 (02:2x local): only (a) llama-server re-runs at -c 16384 on the pristine build, -fa 0 and -fa 1,
# p_short/p_2k/p_8k (the -fa 1 p_8k as one run under a 300 s cap: each of its prefills takes ~250 s), no prefix test,
# and (b) the clean p_32k MLX run with the default 2048-token prefill chunk; kv8 variants and CLI attempts dropped.
# Hard budget: the MLX run is skipped if the servers took more than 19 min, so the hold stays under 25 min.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
B=baseline
log() { echo "[$(date -u +%FT%TZ)] $*"; }
on_signal() { trap - INT TERM; for c in $(pgrep -P $$ 2>/dev/null); do kill -TERM "$c" 2>/dev/null; done; log "hold interrupted; children signalled"; exit 143; }
trap on_signal INT TERM
T0=$(date +%s)
elapsed() { echo $(( $(date +%s) - T0 )); }
log "hold 2 start load=$(sysctl -n vm.loadavg) mem=$(top -l 1 -n 0 | grep PhysMem)"
python $B/bench_llama_server.py --fa 0 --tag fa0-c16k --ctx 16384 --ctx-fallback 0 --prompts p_short,p_2k,p_8k --repeats 3 --no-prefix-test > results/llama-server-fa0-c16k.client.log 2>&1; log "server fa0 c16k done rc=$? (elapsed $(elapsed)s)"
python $B/bench_llama_server.py --fa 1 --tag fa1-c16k --ctx 16384 --ctx-fallback 0 --prompts p_short,p_2k --repeats 3 --single p_8k --timeout 300 --no-prefix-test > results/llama-server-fa1-c16k.client.log 2>&1; log "server fa1 c16k done rc=$? (elapsed $(elapsed)s)"
if (( $(elapsed) < 1140 )); then
  /usr/bin/time -l python $B/bench_mlx.py --no-lock --tag p32k-clean --prompts p_32k --repeats 3 --max-tokens 128 --warmup-tokens 32 --quiet-wait 60 --time-limit 300 > results/mlx-p32k-clean.log 2>&1; log "p32k-clean done rc=$? (elapsed $(elapsed)s)"
else
  log "skipping p32k-clean: elapsed $(elapsed)s would push the hold past 25 min"
fi
log "hold 2 end (elapsed $(elapsed)s)"
