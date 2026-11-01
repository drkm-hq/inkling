#!/bin/zsh
# A3: K3's named server test subset on the fixed and the baseline binaries (sequential, private port)
set -u
SCRATCH=<scratch>
REPO=<workdir>/upstream/llama.cpp-k3
OUT=<workdir>/package/k3-llamacpp-prompt-cache/review
export LLAMA_CACHE=$SCRATCH/llama-cache
export PATH=$SCRATCH/venv-k3/bin:$PATH
cd $REPO/tools/server/tests
mkdir -p tmp
for which in fixed baseline; do
  if [[ $which == fixed ]]; then BIN=$REPO/build-k3/bin/llama-server; else BIN=$REPO/build-k3-baseline/bin/llama-server; fi
  echo "=== subset, $which binary ($(date -u +%FT%TZ)) load=$(sysctl -n vm.loadavg)"
  LLAMA_SERVER_BIN_PATH=$BIN PORT=8521 pytest -v -p no:cacheprovider unit/test_completion.py unit/test_slot_save.py unit/test_ctx_shift.py unit/test_speculative.py unit/test_basic.py unit/test_chat_completion.py unit/test_kv_keep_only_active.py > $OUT/a3-subset-$which.log 2>&1
  echo "exit: $? ; $(tail -1 $OUT/a3-subset-$which.log)"
done
echo "=== done ($(date -u +%FT%TZ))"
