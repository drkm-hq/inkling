#!/bin/zsh
# Clean sequential re-run (fixed dense global_scale dtype); waits for run_all.sh to finish first.
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/R1
while [ ! -f results/ALL_DONE ]; do sleep 5; done
mkdir -p results/contaminated && mv results/layer_bench_*.json results/layer_bench_*.log results/kernels_bench.log results/diag_loop.log results/contaminated/ 2>/dev/null
set -x
python diag_loop.py 2                                                 > results/diag_loop.log 2>&1
python layer_bench.py --experts-bits 2                                > results/layer_bench_e2_abf16_hbf16.log 2>&1
python kernels_bench.py                                               > results/kernels_bench.log 2>&1
python layer_bench.py --experts-bits 4                                > results/layer_bench_e4_abf16_hbf16.log 2>&1
python layer_bench.py --experts-bits 2 --attn-bits 8 --head-bits 8 --skip-prefill > results/layer_bench_e2_a8_h8.log 2>&1
python layer_bench.py --experts-bits 4 --attn-bits 8 --head-bits 8 --skip-prefill > results/layer_bench_e4_a8_h8.log 2>&1
python layer_bench.py --experts-bits 4 --attn-bits 4 --head-bits 4 --skip-prefill > results/layer_bench_e4_a4_h4.log 2>&1
python layer_bench.py --experts-bits 2 --attn-bits 4 --head-bits 4 --skip-prefill > results/layer_bench_e2_a4_h4.log 2>&1
echo ALL_DONE2 > results/ALL_DONE2
