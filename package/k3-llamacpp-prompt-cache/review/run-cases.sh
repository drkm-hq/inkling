#!/bin/zsh
# A3: the adversarial cases (tools/server/tests/unit/test_a3_hybrid_cache.py on branch a3-review) on the fixed,
# the baseline and the upstream-master+patch binaries, sequentially, private port base 8800
set -u
SCRATCH=<scratch>
A3=<workdir>/upstream/llama.cpp-a3
K3=<workdir>/upstream/llama.cpp-k3
M=<workdir>/upstream/llama.cpp-a3-master
OUT=<workdir>/package/k3-llamacpp-prompt-cache/review
export LLAMA_CACHE=$SCRATCH/llama-cache
export PATH=$SCRATCH/venv-k3/bin:$PATH
export A3_OUT=$OUT/a3-cases
cd $A3/tools/server/tests
mkdir -p tmp
for which in fixed baseline master; do
  case $which in
    fixed) BIN=$K3/build-k3/bin/llama-server;;
    baseline) BIN=$K3/build-k3-baseline/bin/llama-server;;
    master) BIN=$M/build-a3m/bin/llama-server;;
  esac
  echo "=== cases, $which binary ($(date -u +%FT%TZ)) load=$(sysctl -n vm.loadavg)"
  A3_TAG=$which LLAMA_SERVER_BIN_PATH=$BIN LLAMA_SERVER_TEST_PORT_BASE=8800 pytest -v -p no:cacheprovider unit/test_a3_hybrid_cache.py > $OUT/a3-cases-$which.log 2>&1
  echo "exit: $? ; $(tail -1 $OUT/a3-cases-$which.log)"
done
echo "=== done ($(date -u +%FT%TZ))"
