#!/bin/zsh
# Copy the tests, the bench scripts and the raw logs into the package directory.
set -e
PKG=~/code/inkling-contrib/package/k2-mlx-lm-inkling
B=~/code/inkling-contrib/bench
mkdir -p $PKG/tests $PKG/bench/scripts $PKG/bench/logs
cp ~/code/inkling-contrib/upstream/mlx-lm/tests/test_inkling.py $PKG/tests/
git -C ~/code/inkling-contrib/upstream/mlx-lm diff 1f7057c -- tests/test_models.py > $PKG/tests/test_models.py.diff
cp $B/K2/*.py $B/K2/*.sh $PKG/bench/scripts/
cp $B/results/K2-*.log $B/results/K2-*.jsonl $B/results/K2-*.json $B/results/K2-*.txt $PKG/bench/logs/ 2>/dev/null || true
cp $B/results/mlx-baseline.jsonl $B/results/mlx-baseline.log $PKG/bench/logs/ 2>/dev/null || true
ls -la $PKG $PKG/bench/logs | head -60
