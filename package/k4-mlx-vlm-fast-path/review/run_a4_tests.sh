#!/bin/zsh
# A4 adversarial tests on the a4-review worktree, default and MLX_ENABLE_TF32=0.
LOG=~/code/inkling-contrib/package/k4-mlx-vlm-fast-path/review/logs
cd ~/code/inkling-contrib/upstream/mlx-vlm-a4 || exit 1
PY=~/.venvs/inkling-a4/bin/python
SHA=$(git rev-parse --short HEAD)
echo "$(date +%T) a4 tests at $SHA load1 $(sysctl -n vm.loadavg)" > $LOG/a4-tests.log
$PY -m pytest -q mlx_vlm/tests/test_a4_window_cache.py mlx_vlm/tests/test_a4_attention_vs_138d761.py mlx_vlm/tests/test_a4_sconv_random.py mlx_vlm/tests/test_a4_logits_to_keep.py mlx_vlm/tests/test_a4_kv_bits.py mlx_vlm/tests/test_a4_loader.py -p no:cacheprovider -x -q > $LOG/pytest-a4-default.log 2>&1
echo "$(date +%T) default rc=$? : $(tail -1 $LOG/pytest-a4-default.log)" >> $LOG/a4-tests.log
MLX_ENABLE_TF32=0 $PY -m pytest -q mlx_vlm/tests/test_a4_window_cache.py mlx_vlm/tests/test_a4_attention_vs_138d761.py mlx_vlm/tests/test_a4_sconv_random.py mlx_vlm/tests/test_a4_logits_to_keep.py mlx_vlm/tests/test_a4_kv_bits.py mlx_vlm/tests/test_a4_loader.py -p no:cacheprovider -q > $LOG/pytest-a4-tf32off.log 2>&1
echo "$(date +%T) TF32=0 rc=$? : $(tail -1 $LOG/pytest-a4-tf32off.log)" >> $LOG/a4-tests.log
