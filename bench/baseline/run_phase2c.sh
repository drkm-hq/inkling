#!/bin/zsh
# Same-process A/B: the smoke-test configuration (20-token prompt, default effort, 48 tokens) next to p_short.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench
sleep 240   # let phase2b's re-run finish first; the lock serialises everything after that
/usr/bin/time -l python baseline/bench_mlx.py --tag smoke --prompts p_short --repeats 3 --max-tokens 128 --smoke --quiet-wait 600 > results/mlx-smoke.log 2>&1
echo "[$(date -u +%FT%TZ)] smoke done rc=$?"
