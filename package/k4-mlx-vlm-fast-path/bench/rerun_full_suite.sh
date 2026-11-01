#!/bin/zsh
# Full mlx_vlm suite at the branch head, started only when >= 40 GiB of host
# RAM is free (the APC tests budget from free RAM and fail while another
# agent's 80-100 GB model is resident). CPU/tiny-GPU only; no GPU lock needed.
LOG=~/code/inkling-contrib/package/k4-mlx-vlm-fast-path/bench/logs
cd ~/code/inkling-contrib/upstream/mlx-vlm || exit 1
source ~/.venvs/inkling/bin/activate
need=$((40 * 1024 * 1024 * 1024))
while true; do
  free=$(python -c "import mlx_vlm.apc as a; print(a._free_ram_bytes() or 0)" 2>/dev/null)
  echo "$(date +%T) free RAM $((free / 2**30)) GiB" >> $LOG/rerun_full_suite.log
  (( free > need )) && break
  sleep 60
done
SHA=$(git rev-parse --short HEAD)
echo "$(date +%T) running full suite at $SHA" >> $LOG/rerun_full_suite.log
python -m pytest -q mlx_vlm/tests -p no:cacheprovider -n 2 > $LOG/pytest-full-$SHA.log 2>&1
echo "$(date +%T) done rc=$? : $(grep -E '[0-9]+ (passed|failed)' $LOG/pytest-full-$SHA.log | tail -1)" >> $LOG/rerun_full_suite.log
