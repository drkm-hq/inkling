"""Produce the 8-bit-attention variant of an mlx-community Inkling checkpoint with mlx-lm: the
bf16 attention projections (fused qkvr, wo_ud), the embedding and the unembedding are
affine-quantized at 8 bits / group 64; the 2-bit experts are untouched. The config's
quantization map gains one entry per quantized module so mlx-lm reloads it as is.

  python make_attn8.py --out ~/models/inkling-small/mlx-2bit-attn8
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpulock  # noqa: E402
from gpulock import log  # noqa: E402

TARGETS = (".attn.qkvr_proj", ".attn.wo_ud", "llm.embed", "llm.unembed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        args.model = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists")
    if not args.no_lock:
        gpulock.acquire("make-attn8")

    import mlx.core as mx
    from mlx_lm.utils import load, quantize_model, save

    t0 = time.perf_counter()
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    log(f"lazy load {time.perf_counter() - t0:.1f}s")

    def predicate(path, module):
        return path.endswith(TARGETS)

    model, config = quantize_model(model, config, args.group_size, args.bits, quant_predicate=predicate)
    n = sum(1 for k in config["quantization"] if isinstance(config["quantization"][k], dict))
    log(f"quantized {n} modules at {args.bits} bits; saving to {args.out}")
    t1 = time.perf_counter()
    save(args.out, args.model, model, tokenizer, config)
    log(f"saved in {time.perf_counter() - t1:.1f}s, peak memory {mx.get_peak_memory() / 2**30:.1f} GiB")
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out, "*.safetensors")))
    log(f"weights on disk: {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
