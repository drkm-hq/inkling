#!/bin/bash
# Sequential benchmark chain (GPU runs never overlap).
# Usage: RUN=2 nohup ./run_all.sh > results/run_all_2.log 2>&1 &
source ~/.venvs/inkling/bin/activate
cd "$(dirname "$0")"
RUN=${RUN:-1}
for s in ${SCRIPTS:-bw bench_sdpa bench_misc bench_moe bench_layer}; do
  echo "=== START $s $(date +%H:%M:%S) load=$(uptime | sed 's/.*averages: //')"
  python $s.py $EXTRA > results/${s}_run${RUN}.log 2>&1
  echo "=== END $s rc=$? $(date +%H:%M:%S)"
done
echo "=== ALL DONE $(date +%H:%M:%S)"
