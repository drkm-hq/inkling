#!/usr/bin/env python3
"""Decode the generated token ids of a results row offline (tokenizer only) and
report length, distinct 4-gram ratio and the longest exact repeat, to judge a
long generation for garbage. Usage: decode_tokens.py <jsonl> <tag> <prompt>"""
import glob
import json
import os
import sys

from transformers import AutoTokenizer

path, tag, prompt = sys.argv[1:4]
model_dir = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/*/")))[0]
tok = AutoTokenizer.from_pretrained(model_dir)
for line in open(path):
    r = json.loads(line)
    if r.get("kind") == "run" and r["tag"] == tag and r["prompt"] == prompt:
        ids = r["token_ids"][:-1]  # drop the closing duplicate
        text = tok.decode(ids)
        grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
        distinct = len(set(grams)) / max(1, len(grams))
        print(f"{tag} {prompt} rep{r.get('rep')}: {len(ids)} tokens, distinct 4-gram ratio {distinct:.3f}, "
              f"peak {r['peak_mem_gib_mx']:.1f} GiB, decode {r['decode_tps_wall']:.1f} tok/s, load1 {r['load1']:.0f}")
        print("--- first 600 chars ---"); print(text[:600])
        print("--- last 600 chars ---"); print(text[-600:])
