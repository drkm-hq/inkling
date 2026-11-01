"""The mlx_lm.generate transcript with a lazy load + chunked parameter eval (used when the
stock CLI's one-shot mx.eval of the 82 GB checkpoint trips the Metal watchdog)."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpulock  # noqa: E402

model_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load

t0 = time.perf_counter()
model, tokenizer = load(model_dir, lazy=True)
total, t_eval = gpulock.chunked_eval([v for _, v in tree_flatten(model.parameters())])
print(f"[lazy load + chunked eval of {total / 2**30:.1f} GiB in {time.perf_counter() - t0:.1f}s]")
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "The secret to baking a good cake is"}],
                                       add_generation_prompt=True, tokenize=False)
print(f"==========\nPrompt: {prompt!r}")
generate(model, tokenizer, prompt, max_tokens=200, verbose=True)
