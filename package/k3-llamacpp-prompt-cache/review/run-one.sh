#!/bin/zsh
# A3: one adversarial case ($1 = pytest -k expression) on the three binaries
set -u
SCRATCH=<scratch>
A3=<workdir>/upstream/llama.cpp-a3
K3=<workdir>/upstream/llama.cpp-k3
M=<workdir>/upstream/llama.cpp-a3-master
OUT=<workdir>/package/k3-llamacpp-prompt-cache/review
export LLAMA_CACHE=$SCRATCH/llama-cache PATH=$SCRATCH/venv-k3/bin:$PATH A3_OUT=$OUT/a3-cases
cd $A3/tools/server/tests
for which in fixed baseline master; do
  case $which in fixed) BIN=$K3/build-k3/bin/llama-server;; baseline) BIN=$K3/build-k3-baseline/bin/llama-server;; master) BIN=$M/build-a3m/bin/llama-server;; esac
  A3_TAG=$which LLAMA_SERVER_BIN_PATH=$BIN LLAMA_SERVER_TEST_PORT_BASE=8800 pytest -v -p no:cacheprovider unit/test_a3_hybrid_cache.py -k "$1" > $OUT/a3-cases-$which-$2.log 2>&1
  echo "$which: $(tail -1 $OUT/a3-cases-$which-$2.log)"
done
