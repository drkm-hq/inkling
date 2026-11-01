#!/bin/zsh
# The APC tests in test_cache.py/test_generate.py budget from free host RAM
# and fail while another job's 80-100 GB model is resident. Wait for free
# RAM, then re-run those two files at the branch head. CPU/tiny-GPU only.
LOG=~/code/inkling-contrib/package/k4-mlx-vlm-fast-path/bench/logs
cd ~/code/inkling-contrib/upstream/mlx-vlm || exit 1
source ~/.venvs/inkling/bin/activate
need=$((40 * 1024 * 1024 * 1024))
while true; do
  free=$(python -c "import mlx_vlm.apc as a; print(a._free_ram_bytes() or 0)" 2>/dev/null)
  echo "$(date +%T) free RAM $((free / 2**30)) GiB" >> $LOG/rerun_apc_tests.log
  (( free > need )) && break
  sleep 60
done
echo "$(date +%T) running at $(git rev-parse --short HEAD)" >> $LOG/rerun_apc_tests.log
python -m pytest -q mlx_vlm/tests/test_cache.py mlx_vlm/tests/test_generate.py -p no:cacheprovider -n 2 > $LOG/pytest-cache-generate-$(git rev-parse --short HEAD).log 2>&1
echo "$(date +%T) done rc=$? : $(grep -E '[0-9]+ (passed|failed)' $LOG/pytest-cache-generate-$(git rev-parse --short HEAD).log | tail -1)" >> $LOG/rerun_apc_tests.log
