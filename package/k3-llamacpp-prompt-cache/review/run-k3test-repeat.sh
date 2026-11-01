#!/bin/zsh
# A3: K3's test 5 times on the fixed binary (flakiness), private port base 8900
set -u
SCRATCH=<scratch>
REPO=<workdir>/upstream/llama.cpp-k3
OUT=<workdir>/package/k3-llamacpp-prompt-cache/review
export LLAMA_CACHE=$SCRATCH/llama-cache PATH=$SCRATCH/venv-k3/bin:$PATH
cd $REPO/tools/server/tests
for i in 1 2 3 4 5; do
  LLAMA_SERVER_BIN_PATH=$REPO/build-k3/bin/llama-server LLAMA_SERVER_TEST_PORT_BASE=8900 pytest -q -p no:cacheprovider unit/test_completion.py -k test_completion_cache_hybrid_memory > $OUT/a3-k3test-fixed-repeat$i.log 2>&1
  echo "repeat $i: $(tail -1 $OUT/a3-k3test-fixed-repeat$i.log)"
done
