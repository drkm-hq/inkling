import json, struct, subprocess, sys, os
base = "https://huggingface.co/thinkingmachines/Inkling-Small/resolve/main/"
shards = ["mtp.safetensors"] + [f"model-{i:05d}-of-00032.safetensors" for i in range(1, 33)]
out = {}
for s in shards:
    try:
        b = subprocess.run(["curl", "-sL", "-r", "0-7", base + s], capture_output=True, timeout=60).stdout
        n = struct.unpack("<Q", b[:8])[0]
        h = subprocess.run(["curl", "-sL", "-r", f"8-{8+n-1}", base + s], capture_output=True, timeout=120).stdout
        hdr = json.loads(h.decode("utf-8"))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            out[k] = {"shape": v["shape"], "dtype": v["dtype"], "file": s}
        print(s, "header bytes", n, "tensors", len(hdr), file=sys.stderr)
    except Exception as e:
        print("FAILED", s, e, file=sys.stderr)
json.dump(out, open("Inkling-Small.tensor_shapes.json", "w"), indent=1)
print("total tensors", len(out))
