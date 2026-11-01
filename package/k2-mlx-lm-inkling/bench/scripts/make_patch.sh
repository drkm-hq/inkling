#!/bin/zsh
# Build the clean three-commit series (cache helper, model, tests) from the final tree of the
# `inkling` branch onto the base commit, and write it with git format-patch into the package.
set -e
cd ~/code/inkling-contrib/upstream/mlx-lm
BASE=1f7057c
PKG=~/code/inkling-contrib/package/k2-mlx-lm-inkling
git diff --quiet || { echo "working tree not clean"; exit 1; }
git branch -f inkling-pr $BASE
git checkout -q inkling-pr
git checkout inkling -- mlx_lm/models/cache.py
git commit -q -m "Add RotatingKVCache.key_distances() (and the batched variant)

The distance of every cached key from the most recent one, in the order
update_and_fetch returns them, so a model can gather a position-dependent
additive mask (a relative position bias) over a rotated cache.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git checkout inkling -- mlx_lm/models/inkling_mm_model.py
git commit -q -m "Add Inkling (inkling_mm_model): hybrid sliding/global attention with a relative bias, short convs and a sigmoid-routed MoE

Loads the mlx-community Inkling-Small checkpoints directly (model.llm.* paths)
and the HF source layout (interleaved w13 de-interleave, conv axis swap, MTP /
vision / audio weights dropped). q/k/v/r are one projection fused at load, k|v
share one short convolution with a single cached state, the relative bias bank
is stored reversed so a token's bias is a slice (global layers) or one shared
gather over the rotating cache (sliding layers), prefill builds the banded
additive mask with a small Metal kernel, and the shared experts are appended to
the routed ones so one SwitchGLU call serves both. The router runs in fp32.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git checkout inkling -- tests/test_inkling.py tests/test_models.py
git commit -q -m "Test Inkling: tiny-config runner, banded mask vs numpy, sconv state, router, sanitize layouts, key_distances, batched prompts, CPU exactness

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git checkout inkling -- mlx_lm/generate.py ACKNOWLEDGMENTS.md
git commit -q -m "Warn when kv_bits cannot apply to any of the model's caches

maybe_quantize_kv_cache stayed silent for models whose per-layer cache has no
to_quantized (CacheList models); the single-stream path now warns once.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git diff --stat inkling inkling-pr | tail -1   # must be empty: same tree
setopt null_glob; rm -f $PKG/*.patch
git format-patch $BASE..inkling-pr --stdout > $PKG/k2-mlx-lm-inkling.patch
git log --oneline $BASE..inkling-pr
git checkout -q inkling
wc -l $PKG/k2-mlx-lm-inkling.patch
