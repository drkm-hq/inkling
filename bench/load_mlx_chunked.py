"""Load an MLX Inkling checkpoint with a chunked parameter eval (avoids the IOGPU
watchdog timeout that a one-shot mx.eval of ~80 GB trips), then run a short greedy
generation and print timings. Usage: python load_mlx_chunked.py <model_dir> [max_tokens]"""
import sys, time
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 48
t0 = time.time()
model, processor = load(model_path, lazy=True)
print(f"lazy load {time.time()-t0:.1f}s", flush=True)
params = tree_flatten(model.parameters())
chunk, nb, total, t1 = [], 0, 0, time.time()
for k, v in params:
    chunk.append(v); nb += v.nbytes
    if nb > 4 * 2**30:
        mx.eval(chunk); total += nb
        print(f"  evaluated {total/2**30:.1f} GiB ({time.time()-t1:.1f}s)", flush=True)
        chunk, nb = [], 0
if chunk:
    mx.eval(chunk); total += nb
print(f"full eval {time.time()-t1:.1f}s, {total/2**30:.1f} GiB, active {mx.get_active_memory()/2**30:.1f} GiB, peak {mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
prompt = "The capital of France is"
try:
    formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
except Exception as e:
    print("chat template failed:", e, flush=True); formatted = prompt
out = generate(model, processor, formatted, max_tokens=max_tokens, temperature=0.0, verbose=True)
print("TEXT:", getattr(out, "text", out))
print(f"peak memory {mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
