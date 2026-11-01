#!/bin/zsh
# A4: the 83 APC-budget tests (test_cache.py / test_generate.py) fail while an
# 80-100 GB model is resident (the manager budgets from free host RAM). Wait
# for >= 40 GiB free, then run those two files on K4's branch head.
LOG=~/code/inkling-contrib/package/k4-mlx-vlm-fast-path/review/logs
cd ~/code/inkling-contrib/upstream/mlx-vlm || exit 1
source ~/.venvs/inkling/bin/activate
need=$((40 * 1024 * 1024 * 1024))
while true; do
  free=$(python -c "import mlx_vlm.apc as a; print(a._free_ram_bytes() or 0)" 2>/dev/null)
  echo "$(date +%T) free RAM $((free / 2**30)) GiB, load1 $(sysctl -n vm.loadavg)" >> $LOG/rerun_apc_tests_a4.log
  (( free > need )) && break
  sleep 60
done
SHA=$(git rev-parse --short HEAD)
echo "$(date +%T) running test_cache.py + test_generate.py at $SHA (dirty: $(git status --short | wc -l | tr -d ' '))" >> $LOG/rerun_apc_tests_a4.log
python -m pytest -q mlx_vlm/tests/test_cache.py mlx_vlm/tests/test_generate.py -p no:cacheprovider -n 2 > $LOG/pytest-cache-generate-$SHA-a4.log 2>&1
echo "$(date +%T) done rc=$? : $(grep -E '[0-9]+ (passed|failed)' $LOG/pytest-cache-generate-$SHA-a4.log | tail -1)" >> $LOG/rerun_apc_tests_a4.log
