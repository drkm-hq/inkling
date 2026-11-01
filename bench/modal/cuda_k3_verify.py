"""K3 prompt-cache strict-extension case on CUDA (Modal H200): before and after the K3 server patch.

Two functions:
  download  (CPU only): unsloth/Inkling-Small-GGUF UD-IQ3_XXS (3 shards, 91 GiB) into the Volume
            `inkling-small-gguf` (persists; nothing else is written there).
  verify_k3 (H200):     build llama-server at the head of ggml-org/llama.cpp#25731 with GGML_CUDA, run the
            strict-extension case (p_8k cached, then p_8k + 200 tokens, and p_8k + its generated tokens + 200
            tokens, cache_prompt=true; read timings.prompt_n / cache_n), apply
            package/k3-llamacpp-prompt-cache/patches/0001-*.patch, rebuild llama-server, run the case again.
            Server flags: -fa 1 -ngl 99 -np 1 --no-kv-unified (plus -c 16384 -ub 512 --temp 0).

Run with MODAL_PROFILE set to your own Modal profile:

    MODAL_PROFILE=<profile> nohup modal run cuda_k3_verify.py::download > k3_download.run.log 2>&1 &
    MODAL_PROFILE=<profile> nohup modal run cuda_k3_verify.py > k3_verify.run.log 2>&1 &   # the entrypoint writes the logs
    # (`modal run cuda_k3_verify.py::verify_k3` runs the function alone and discards its return value)

Nothing is pushed anywhere; only the patch and the three prompt files are shipped to the container.
"""

import glob
import json
import os
import re
import signal
import subprocess
import time
import urllib.request

import modal

APP_NAME = "inkling-cuda-verify-k3"
PR_NUMBER = 25731
PR_HEAD = "3582c5ee05115892e4e96520be0026c3ceb9f829"
REPO_URL = "https://github.com/ggml-org/llama.cpp"
HF_REPO = "unsloth/Inkling-Small-GGUF"
HF_SUBDIR = "UD-IQ3_XXS"
VOL_NAME = "inkling-small-gguf"
VOL_PATH = "/vol"

WORK = "/work"
SRC = f"{WORK}/llama.cpp"
NPROC = 32
PORT = 18640

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "cmake", "ninja-build", "build-essential", "ca-certificates", "curl")
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

if modal.is_local():
    _PKG = os.path.expanduser("~/code/inkling-contrib/package/k3-llamacpp-prompt-cache/patches")
    _PROMPTS = os.path.expanduser("~/code/inkling-contrib/bench/prompts")
    LOCAL_OUT = os.path.expanduser("~/code/inkling-contrib/package/k1-metal-banded-fa/bench/cuda/k3")
    _patch = sorted(glob.glob(os.path.join(_PKG, "0001-*.patch")))[0]
    image = (image
             .add_local_file(_patch, remote_path="/patches/k3-0001.patch")
             .add_local_file(os.path.join(_PROMPTS, "p_8k.txt"), remote_path="/prompts/p_8k.txt")
             .add_local_file(os.path.join(_PROMPTS, "p_short.txt"), remote_path="/prompts/p_short.txt")
             .add_local_file(os.path.join(_PROMPTS, "src/moby-dick.txt"), remote_path="/prompts/src/moby-dick.txt"))

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def sh(cmd, logs, log, cwd=SRC, check=False, quiet=False):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace")
    out = []
    for line in p.stdout:
        out.append(line)
        if not quiet:
            print(line, end="", flush=True)
    rc = p.wait()
    text = "".join(out)
    print(f"-> rc={rc} ({len(out)} lines)", flush=True)
    logs[log] = logs.get(log, "") + f"$ {cmd}\n{text}\n[rc={rc}]\n"
    if check and rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd}\n{text[-4000:]}")
    return rc, text


@app.function(image=image, cpu=8, memory=16384, timeout=5400, volumes={VOL_PATH: vol})
def download() -> dict:
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(HF_REPO, allow_patterns=[f"{HF_SUBDIR}/*"], local_dir=f"{VOL_PATH}/{HF_REPO}")
    vol.commit()
    files = sorted(glob.glob(f"{path}/{HF_SUBDIR}/*.gguf"))
    sizes = {os.path.basename(f): os.path.getsize(f) for f in files}
    print(json.dumps({"path": path, "files": sizes, "s": round(time.time() - t0, 1)}, indent=2), flush=True)
    return {"path": path, "files": sizes, "s": round(time.time() - t0, 1)}


class Server:
    def __init__(self, cmd, log_path):
        self.log = open(log_path, "a")
        self.log.write(f"\n===== {time.strftime('%FT%T')} {' '.join(cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{PORT}"

    def wait_healthy(self, timeout=1500):
        t0 = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return time.perf_counter() - t0
            except Exception:
                pass
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("llama-server did not become healthy in time")
            time.sleep(3)

    def post(self, path, payload, timeout=1800):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        return self.proc.returncode


def strict_extension_case(tag, model, logs):
    """K3's case (a)/(a2) in quick form: returns the rows (prompt_n / cache_n / prompt_ms per request)."""
    cmd = [f"{SRC}/build/bin/llama-server", "-m", model, "-fa", "1", "-ngl", "99", "-np", "1", "--no-kv-unified",
           "-c", "16384", "-ub", "512", "--temp", "0", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui",
           "-to", "3600", "-lv", "4"]
    rows = []
    srv = Server(cmd, f"{WORK}/server-{tag}.log")
    try:
        load_s = srv.wait_healthy()
        rows.append({"kind": "load", "load_s": round(load_s, 1), "cmd": " ".join(cmd)})
        print(f"[{tag}] server healthy after {load_s:.1f}s", flush=True)

        def tokenize(text, add_special):
            return srv.post("/tokenize", {"content": text, "add_special": add_special})["tokens"]

        p_8k = open("/prompts/p_8k.txt").read()
        p_short = open("/prompts/p_short.txt").read()
        src = open("/prompts/src/moby-dick.txt", encoding="utf-8", errors="replace").read()
        excerpt = p_8k.split("\n\nQuestion:")[0]
        i = src.find(excerpt[-200:])
        assert i > 0, "p_8k excerpt not found in the source text"
        continuation = src[i + 200:i + 200 + 4000]

        A = tokenize(p_8k, True)
        extra = tokenize("\n" + continuation, False)[:200]
        rows.append({"kind": "prompts", "n_A": len(A), "n_extra": len(extra)})

        def completion(name, prompt, cache_prompt, n_predict=64, note=""):
            t0 = time.perf_counter()
            body = srv.post("/completion", {"prompt": prompt, "cache_prompt": cache_prompt, "n_predict": n_predict,
                                            "temperature": 0, "return_tokens": True})
            wall = time.perf_counter() - t0
            t = body.get("timings", {})
            row = {"kind": "completion", "name": name, "cache_prompt": cache_prompt, "n_prompt": len(prompt),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "wall_s": round(wall, 3), "note": note, "tokens": body.get("tokens", []),
                   "content": (body.get("content") or "")[:300]}
            rows.append(row)
            print(f"[{tag}] {name}: n_prompt={len(prompt)} prompt_n={row['prompt_n']} cache_n={row['cache_n']} "
                  f"prompt_ms={row['prompt_ms']} wall={wall:.2f}s", flush=True)
            return row

        completion("warmup_p_short", tokenize(p_short, True), False, n_predict=16)
        B1 = A + extra
        r_A_ref = completion("A_ref", A, False, note="p_8k, no cache")
        r_B1_ref = completion("B1_ref", B1, False, note="p_8k + 200 tokens, no cache")
        # (a) prime, then +200 (the slot holds p_8k plus the generated tokens: a rewind past them)
        completion("A_cached", A, True, note="p_8k, cache_prompt=true (prime)")
        r_B1 = completion("B1_cached", B1, True, note="p_8k + 200 tokens, cache_prompt=true")
        # (a2) the strict extension: p_8k + what the model generated for it + 200 tokens
        r_A2 = completion("A_cached_2", A, True, note="p_8k, cache_prompt=true (prime for a2)")
        gen = r_A2["tokens"][:-1]
        B2 = A + gen + extra
        r_B2 = completion("B2_cached", B2, True, note="p_8k + generated + 200 tokens, cache_prompt=true (strict extension)")
        r_B2_ref = completion("B2_ref", B2, False, note="same prompt, no cache")

        def first_diff(a, b):
            for k, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return k
            return None if len(a) == len(b) else min(len(a), len(b))

        rows.append({"kind": "compare", "name": "B1_cached_vs_B1_ref", "first_diff": first_diff(r_B1["tokens"], r_B1_ref["tokens"])})
        rows.append({"kind": "compare", "name": "B2_cached_vs_B2_ref", "first_diff": first_diff(r_B2["tokens"], r_B2_ref["tokens"])})
        rows.append({"kind": "compare", "name": "A_ref_vs_A_cached_2", "first_diff": first_diff(r_A_ref["tokens"], r_A2["tokens"])})
    finally:
        rc = srv.stop()
        rows.append({"kind": "server_exit", "returncode": rc})
        text = open(f"{WORK}/server-{tag}.log", errors="replace").read()
        logs[f"k3-server-{tag}.log"] = text
        reuse = [l for l in text.splitlines() if re.search(r"checkpoint|forcing full prompt|n_past|n_swa|partial sequence removal", l)]
        logs[f"k3-server-{tag}-reuse-lines.txt"] = "\n".join(reuse) + "\n"
    return rows


@app.function(image=image, gpu="H200", timeout=5400, cpu=NPROC, memory=98304, volumes={VOL_PATH: vol})
def verify_k3(arch: str = "90") -> dict:
    t0 = time.time()
    logs, meta = {}, {"arch": arch, "expected_head": PR_HEAD}
    os.makedirs(WORK, exist_ok=True)
    vol.reload()
    shards = sorted(glob.glob(f"{VOL_PATH}/{HF_REPO}/{HF_SUBDIR}/*.gguf"))
    meta["shards"] = {os.path.basename(f): os.path.getsize(f) for f in shards}
    if len(shards) != 3:
        meta["error"] = f"expected 3 shards in the volume, found {shards}"
        return {"meta": meta, "logs": logs, "rows": {}}
    # one pass over the network volume, then both server starts read the local disk
    tc = time.time()
    os.makedirs(f"{WORK}/model", exist_ok=True)
    sh(f"cp {VOL_PATH}/{HF_REPO}/{HF_SUBDIR}/*.gguf {WORK}/model/ && ls -la {WORK}/model && df -h {WORK}",
       logs, "k3-env.log", cwd=WORK, check=True)
    meta["model_copy_s"] = round(time.time() - tc, 1)
    model = sorted(glob.glob(f"{WORK}/model/*.gguf"))[0]

    sh("nvidia-smi; nvcc --version | tail -2", logs, "k3-env.log", cwd=WORK)
    sh(f"rm -rf {SRC} && git init -q {SRC} && cd {SRC} && git remote add origin {REPO_URL}", logs, "k3-clone.log", cwd=WORK, check=True)
    sh("git config user.name inkling-cuda-verify && git config user.email verify@localhost", logs, "k3-clone.log", check=True)
    sh(f"git fetch --depth 200 origin pull/{PR_NUMBER}/head:pr-{PR_NUMBER}", logs, "k3-clone.log", check=True)
    _, head = sh(f"git rev-parse pr-{PR_NUMBER}", logs, "k3-clone.log", check=True)
    if head.strip() != PR_HEAD:
        rc, _ = sh(f"git cat-file -e {PR_HEAD}^{{commit}} || git fetch --depth 200 origin {PR_HEAD}", logs, "k3-clone.log")
        if rc != 0:
            meta["error"] = f"could not obtain {PR_HEAD}"
            return {"meta": meta, "logs": logs, "rows": {}}
        sh(f"git branch -f pr-{PR_NUMBER} {PR_HEAD}", logs, "k3-clone.log", check=True)
    sh(f"git checkout -q pr-{PR_NUMBER} && git rev-parse HEAD", logs, "k3-clone.log", check=True)

    sh(f"cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "
       f"-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES={arch}", logs, "k3-configure.log", check=True)
    tb = time.time()
    sh(f"cmake --build build --target llama-server -j {NPROC}", logs, "k3-build-before.log", check=True, quiet=True)
    meta["build_before_s"] = round(time.time() - tb, 1)

    rows = {}
    rows["before"] = strict_extension_case("before", model, logs)
    meta["t_before_done_s"] = round(time.time() - t0, 1)

    sh("git am /patches/k3-0001.patch && git log --oneline -2", logs, "k3-am.log", check=True)
    tb = time.time()
    sh(f"cmake --build build --target llama-server -j {NPROC}", logs, "k3-build-after.log", check=True, quiet=True)
    meta["build_after_s"] = round(time.time() - tb, 1)
    rows["after"] = strict_extension_case("after", model, logs)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    return {"meta": meta, "logs": logs, "rows": rows}


@app.local_entrypoint()
def main(arch: str = "90"):
    res = verify_k3.remote(arch=arch)
    os.makedirs(LOCAL_OUT, exist_ok=True)
    for name, text in res["logs"].items():
        with open(os.path.join(LOCAL_OUT, name), "w") as f:
            f.write(text)
    with open(os.path.join(LOCAL_OUT, "meta.json"), "w") as f:
        json.dump(res["meta"], f, indent=2)
    for tag, rows in res["rows"].items():
        with open(os.path.join(LOCAL_OUT, f"{tag}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print("\n===== SUMMARY =====")
    print(json.dumps(res["meta"], indent=2))
    for tag, rows in res["rows"].items():
        for row in rows:
            if row["kind"] == "completion":
                print(f"{tag:6s} {row['name']:14s} n_prompt={row['n_prompt']:5d} prompt_n={row['prompt_n']} "
                      f"cache_n={row['cache_n']} prompt_ms={row['prompt_ms']}")
            elif row["kind"] == "compare":
                print(f"{tag:6s} {row['name']}: first_diff={row['first_diff']}")
    print(f"logs written to {LOCAL_OUT}")
