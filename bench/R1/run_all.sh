#!/bin/zsh
# Runs the synthetic benchmarks sequentially (never concurrently: they share the GPU).
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/R1
set -x
python layer_bench.py --experts-bits 2                                > results/layer_bench_e2_abf16_hbf16.log 2>&1
python kernels_bench.py                                               > results/kernels_bench.log 2>&1
python layer_bench.py --experts-bits 4                                > results/layer_bench_e4_abf16_hbf16.log 2>&1
python layer_bench.py --experts-bits 2 --attn-bits 8 --head-bits 8 --skip-prefill > results/layer_bench_e2_a8_h8.log 2>&1
python layer_bench.py --experts-bits 4 --attn-bits 4 --head-bits 4 --skip-prefill > results/layer_bench_e4_a4_h4.log 2>&1
python layer_bench.py --experts-bits 4 --attn-bits 8 --head-bits 8 --skip-prefill > results/layer_bench_e4_a8_h8.log 2>&1
echo ALL_DONE > results/ALL_DONE
