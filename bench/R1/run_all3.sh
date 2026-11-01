#!/bin/zsh
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/R1
while [ ! -f results/ALL_DONE2 ]; do sleep 5; done
set -x
python sdpa_probe.py                                                  > results/sdpa_probe.log 2>&1
python layer_bench.py --experts-bits 2 --skip-prefill --tag e2_abf16_hbf16_decode2 > results/layer_bench_e2_abf16_hbf16_decode2.log 2>&1
python layer_bench.py --experts-bits 4 --skip-prefill --tag e4_abf16_hbf16_decode2 > results/layer_bench_e4_abf16_hbf16_decode2.log 2>&1
python mtp_probe.py 2                                                 > results/mtp_probe.log 2>&1
python layer_bench.py --experts-bits 2 --attn-bits 8 --head-bits 8 --skip-prefill --tag e2_a8_h8_decode2 > results/layer_bench_e2_a8_h8_decode2.log 2>&1
python layer_bench.py --experts-bits 4 --attn-bits 4 --head-bits 4 --skip-prefill --tag e4_a4_h4_decode2 > results/layer_bench_e4_a4_h4_decode2.log 2>&1
echo ALL_DONE3 > results/ALL_DONE3
