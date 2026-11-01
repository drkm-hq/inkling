#!/bin/zsh
source ~/.venvs/inkling/bin/activate
cd ~/code/inkling-contrib/bench/R1
while [ ! -f results/kv_probe.json ]; do sleep 5; done
set -x
for cfg in "2 _ _" "4 _ _" "2 8 8" "4 8 8" "4 4 4"; do
  set -- ${=cfg}
  E=$1; A=$2; H=$3
  args="--experts-bits $E --skip-prefill"
  tag="e${E}_a$([ "$A" = "_" ] && echo bf16 || echo $A)_h$([ "$H" = "_" ] && echo bf16 || echo $H)_final"
  [ "$A" != "_" ] && args="$args --attn-bits $A"
  [ "$H" != "_" ] && args="$args --head-bits $H"
  python layer_bench.py ${=args} --tag $tag > results/layer_bench_$tag.log 2>&1
done
echo ALL_DONE4 > results/ALL_DONE4
