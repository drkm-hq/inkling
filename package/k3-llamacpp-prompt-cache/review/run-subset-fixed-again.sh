#!/bin/zsh
# A3: re-run of the fixed-binary subset on a private port base (the first run shared port 8080 with another run)
set -u
SCRATCH=<scratch>
REPO=<workdir>/upstream/llama.cpp-k3
OUT=<workdir>/package/k3-llamacpp-prompt-cache/review
export LLAMA_CACHE=$SCRATCH/llama-cache
export PATH=$SCRATCH/venv-k3/bin:$PATH
while kill -0 54479 2>/dev/null; do sleep 10; done
cd $REPO/tools/server/tests
echo "=== subset, fixed binary, rerun ($(date -u +%FT%TZ)) load=$(sysctl -n vm.loadavg)"
LLAMA_SERVER_BIN_PATH=$REPO/build-k3/bin/llama-server LLAMA_SERVER_TEST_PORT_BASE=8700 pytest -v -p no:cacheprovider unit/test_completion.py unit/test_slot_save.py unit/test_ctx_shift.py unit/test_speculative.py unit/test_basic.py unit/test_chat_completion.py unit/test_kv_keep_only_active.py > $OUT/a3-subset-fixed-rerun.log 2>&1
echo "exit: $? ; $(tail -1 $OUT/a3-subset-fixed-rerun.log)"
