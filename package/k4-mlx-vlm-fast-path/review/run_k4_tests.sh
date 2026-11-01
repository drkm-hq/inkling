#!/bin/zsh
# A4: K4's own tests, then the full suite, on K4's branch (shared venv, clone root).
LOG=~/code/inkling-contrib/package/k4-mlx-vlm-fast-path/review/logs
cd ~/code/inkling-contrib/upstream/mlx-vlm || exit 1
source ~/.venvs/inkling/bin/activate
SHA=$(git rev-parse --short HEAD)
echo "$(date +%T) branch head $SHA, working tree: $(git status --short | wc -l | tr -d ' ') dirty files" > $LOG/k4-tests.log
echo "$(date +%T) load1 $(sysctl -n vm.loadavg)" >> $LOG/k4-tests.log
FILES="mlx_vlm/tests/test_load_eval_chunks.py mlx_vlm/tests/test_sliding_window_cache.py mlx_vlm/tests/test_inkling_fast_path.py"
python -m pytest -q $FILES -p no:cacheprovider > $LOG/pytest-k4-new-$SHA.log 2>&1
echo "$(date +%T) k4 new tests rc=$? : $(tail -1 $LOG/pytest-k4-new-$SHA.log)" >> $LOG/k4-tests.log
MLX_ENABLE_TF32=0 python -m pytest -q $FILES -p no:cacheprovider > $LOG/pytest-k4-new-tf32off-$SHA.log 2>&1
echo "$(date +%T) k4 new tests TF32=0 rc=$? : $(tail -1 $LOG/pytest-k4-new-tf32off-$SHA.log)" >> $LOG/k4-tests.log
free=$(python -c "import mlx_vlm.apc as a; print((a._free_ram_bytes() or 0)//2**30)" 2>/dev/null)
echo "$(date +%T) full suite start, free RAM ${free} GiB, load1 $(sysctl -n vm.loadavg)" >> $LOG/k4-tests.log
python -m pytest -q mlx_vlm/tests -p no:cacheprovider -n 2 > $LOG/pytest-full-$SHA.log 2>&1
echo "$(date +%T) full suite rc=$? : $(grep -E '[0-9]+ (passed|failed)' $LOG/pytest-full-$SHA.log | tail -1)" >> $LOG/k4-tests.log
echo "$(date +%T) skips:" >> $LOG/k4-tests.log
grep -i "skip" $LOG/pytest-full-$SHA.log | head -40 >> $LOG/k4-tests.log
