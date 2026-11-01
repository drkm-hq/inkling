"""CUDA verification of the K1 banded flash-attention patches on an NVIDIA GPU (Modal).

Builds llama.cpp at the head of ggml-org/llama.cpp#25731 with GGML_CUDA, then runs the op-level tests
in three stages: (a) the PR head unpatched, (b) + 0001 (Metal kernel, compiled out on Linux) + 0002
(test cases), (c) + 0003 (the q_offset proposal: ggml.h/ggml.c, CPU, CUDA scalar + MMA, model views).
The function streams everything to the Modal log and returns every log as text; the local entrypoint
writes them to package/k1-metal-banded-fa/bench/cuda/.

Run from this directory with MODAL_PROFILE set to your own Modal profile:

    MODAL_PROFILE=<profile> nohup modal run cuda_verify.py > cuda_verify.run.log 2>&1 &

Nothing is pushed anywhere; only the patches directory is shipped to the container.
"""

import json
import os
import re
import subprocess
import time

import modal

APP_NAME = "inkling-cuda-verify"
PR_NUMBER = 25731
PR_HEAD = "3582c5ee05115892e4e96520be0026c3ceb9f829"
REPO_URL = "https://github.com/ggml-org/llama.cpp"

PATCHES = [
    "0001-metal-add-GGML_OP_FLASH_ATTN_EXT_BANDED.patch",
    "0002-tests-banded-flash-attention-cases.patch",
    "0003-ggml-q_offset-for-the-banded-flash-attention-proposa.patch",
]

# files the patches touch; compiler warnings mentioning them are collected
PATCHED_FILES = [
    "ggml/include/ggml.h",
    "ggml/src/ggml.c",
    "ggml-cpu/ops.cpp",
    "fattn-banded.cu",
    "fattn-common.cuh",
    "fattn-mma-f16.cuh",
    "src/models/inkling.cpp",
    "tests/test-backend-ops.cpp",
    "tests/test-flash-attn-bias.cpp",
]

WORK = "/work"
SRC = f"{WORK}/llama.cpp"
BUILD = f"{SRC}/build"
NPROC = 32

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "cmake", "ninja-build", "build-essential", "ca-certificates", "curl")
)

if modal.is_local():
    # the runner module is imported again inside the container; keep the local paths local
    LOCAL_PATCHES = os.path.expanduser("~/code/inkling-contrib/package/k1-metal-banded-fa/patches")
    LOCAL_OUT = os.path.expanduser("~/code/inkling-contrib/package/k1-metal-banded-fa/bench/cuda")
    image = image.add_local_dir(LOCAL_PATCHES, remote_path="/patches")

app = modal.App(APP_NAME)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Runner:
    def __init__(self):
        self.t0 = time.time()
        self.logs = {}      # file name -> text
        self.events = []    # (elapsed s, message)

    def note(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.events.append(line)

    def sh(self, cmd, log=None, cwd=SRC, check=False, quiet=False, timeout=None):
        """Run a shell command, stream its output to the Modal log, keep it under `log`."""
        self.note(f"$ {cmd}")
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace")
        out = []
        try:
            for line in p.stdout:
                out.append(line)
                if not quiet:
                    print(line, end="", flush=True)
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out.append(f"\n[killed after {timeout}s]\n")
            rc = -9
        text = "".join(out)
        self.note(f"-> rc={rc} ({len(out)} lines)")
        if log is not None:
            self.logs[log] = self.logs.get(log, "") + f"$ {cmd}\n" + text + f"\n[rc={rc}]\n"
        if check and rc != 0:
            raise RuntimeError(f"command failed rc={rc}: {cmd}\n{text[-4000:]}")
        return rc, text


def parse_backend_ops(text):
    """Count OK / FAIL / not supported lines of a test-backend-ops run; keep the FAIL lines."""
    clean = ANSI.sub("", text)
    ok = fail = unsupported = 0
    fails = []
    for line in clean.splitlines():
        s = line.strip()
        if s.endswith(": OK"):
            ok += 1
        elif s.endswith(": FAIL") or s.endswith("FAIL"):
            if "(" in s:
                fail += 1
                fails.append(s)
        elif "not supported" in s:
            unsupported += 1
    m = re.search(r"(\d+)/(\d+) tests passed", clean)
    passed = (int(m.group(1)), int(m.group(2))) if m else None
    backend_ok = "Backend CUDA0: OK" in clean
    return {"ok": ok, "fail": fail, "not_supported": unsupported, "fails": fails,
            "summary": passed, "backend_ok": backend_ok}


def parse_bias(text):
    clean = ANSI.sub("", text)
    npass = len(re.findall(r"\bPASS\b", clean))
    nfail = len(re.findall(r"\bFAIL\b", clean))
    fails = [l.strip() for l in clean.splitlines() if "FAIL" in l]
    return {"pass": npass, "fail": nfail, "fails": fails}


def warnings_in(text):
    clean = ANSI.sub("", text)
    seen = []
    for line in clean.splitlines():
        if "warning" in line.lower() and any(f in line for f in PATCHED_FILES):
            s = line.strip()
            if s not in seen:
                seen.append(s)
    return seen


@app.function(image=image, gpu="H100", timeout=5400, cpu=NPROC, memory=65536)
def verify(arch: str = "90", plain_fa_baseline: bool = True, mode: str = "full") -> dict:
    """mode "full": stages (a), (b), (c). mode "final": the whole series 0001..0003 plus any 0004-*.patch in the
    patches directory applied at once, then the banded test, the bias tests, the banded perf and the plain-FA perf
    ceiling at the Inkling shapes (the follow-up run after a tolerance fix)."""
    import glob
    r = Runner()
    meta = {"arch": arch, "pr": PR_NUMBER, "expected_head": PR_HEAD, "mode": mode, "stages": {}}
    os.makedirs(WORK, exist_ok=True)

    # ---- environment ---------------------------------------------------------------------------
    _, meta["nvidia_smi"] = r.sh("nvidia-smi", log="env.log", cwd=WORK)
    _, meta["nvcc"] = r.sh("nvcc --version", log="env.log", cwd=WORK)
    r.sh("gcc --version | head -1; cmake --version | head -1; ninja --version; nproc; free -g | head -2",
         log="env.log", cwd=WORK)
    r.sh("ls -la /patches && sha256sum /patches/*.patch", log="env.log", cwd=WORK)

    # ---- source at the PR head -------------------------------------------------------------------
    r.sh(f"rm -rf {SRC} && git init -q {SRC} && cd {SRC} && git remote add origin {REPO_URL}",
         log="clone.log", cwd=WORK, check=True)
    r.sh("git config user.name inkling-cuda-verify && git config user.email verify@localhost",
         log="clone.log", check=True)
    r.sh(f"git fetch --depth 200 origin pull/{PR_NUMBER}/head:pr-{PR_NUMBER}", log="clone.log", check=True)
    _, head = r.sh(f"git rev-parse pr-{PR_NUMBER}", log="clone.log", check=True)
    head = head.strip()
    meta["fetched_pr_head"] = head
    if head != PR_HEAD:
        r.note(f"PR head moved: fetched {head}, expected {PR_HEAD}; pinning the expected commit")
        rc, _ = r.sh(f"git cat-file -e {PR_HEAD}^{{commit}}", log="clone.log")
        if rc != 0:
            rc, _ = r.sh(f"git fetch --depth 200 origin {PR_HEAD}", log="clone.log")
        if rc != 0:
            rc, _ = r.sh(f"git fetch origin pull/{PR_NUMBER}/head && git cat-file -e {PR_HEAD}^{{commit}}",
                         log="clone.log")
        if rc != 0:
            meta["error"] = f"could not obtain {PR_HEAD}"
            return {"meta": meta, "logs": r.logs, "events": r.events}
        r.sh(f"git branch -f pr-{PR_NUMBER} {PR_HEAD}", log="clone.log", check=True)
    r.sh(f"git checkout -q pr-{PR_NUMBER} && git log --oneline -1", log="clone.log", check=True)
    _, head = r.sh("git rev-parse HEAD", log="clone.log", check=True)
    meta["head"] = head.strip()
    assert meta["head"] == PR_HEAD, meta["head"]

    # ---- helpers ---------------------------------------------------------------------------------
    def configure():
        r.sh(f"cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "
             f"-DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES={arch}",
             log="configure.log", check=True)

    def build(tag):
        t = time.time()
        rc, text = r.sh(f"cmake --build build --target test-backend-ops test-flash-attn-bias llama-cli -j {NPROC}",
                        log=f"build-{tag}.log", quiet=True)
        st = meta["stages"].setdefault(tag, {})
        st["build_rc"] = rc
        st["build_s"] = round(time.time() - t, 1)
        st["warnings_patched_files"] = warnings_in(text)
        # errors are always worth seeing in the stream
        errs = [l for l in ANSI.sub("", text).splitlines() if re.search(r"\berror\b", l, re.I)]
        for l in errs[:60]:
            print("  " + l, flush=True)
        st["build_errors"] = errs[:200]
        r.note(f"build {tag}: rc={rc} in {st['build_s']}s, {len(st['warnings_patched_files'])} warning lines in patched files")
        return rc == 0

    def run_tests(tag, plain_fa, perf):
        st = meta["stages"].setdefault(tag, {})
        t = time.time()
        _, text = r.sh("./build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT_BANDED",
                       log=f"test-{tag}-banded.log", timeout=1800)
        st["banded"] = parse_backend_ops(text)
        st["banded_s"] = round(time.time() - t, 1)
        r.note(f"{tag} banded: {st['banded']['ok']} OK / {st['banded']['fail']} FAIL / "
               f"{st['banded']['not_supported']} not supported; summary {st['banded']['summary']}")

        st["bias"] = {}
        for be, suite in (("cuda", "small"), ("cuda", "medium"), ("cuda", "hard"), ("cuda", "overflow"),
                          ("cpu", "small")):
            rc, text = r.sh(f"./build/bin/test-flash-attn-bias {be} {suite}",
                            log=f"bias-{tag}-{be}-{suite}.log", timeout=900)
            st["bias"][f"{be}-{suite}"] = {"rc": rc, **parse_bias(text)}
        r.note(f"{tag} test-flash-attn-bias: " + ", ".join(
            f"{k} rc={v['rc']} pass={v['pass']} fail={v['fail']}" for k, v in st["bias"].items()))

        if perf:
            t = time.time()
            r.sh("./build/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT_BANDED",
                 log=f"perf-{tag}-banded.log", timeout=900)
            st["perf_s"] = round(time.time() - t, 1)

        if plain_fa:
            t = time.time()
            _, text = r.sh("./build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT",
                           log=f"test-{tag}-plain-fa.log", quiet=True, timeout=1800)
            st["plain_fa"] = parse_backend_ops(text)
            st["plain_fa_s"] = round(time.time() - t, 1)
            for l in st["plain_fa"]["fails"][:50]:
                print("  " + l, flush=True)
            r.note(f"{tag} plain FA: {st['plain_fa']['ok']} OK / {st['plain_fa']['fail']} FAIL / "
                   f"{st['plain_fa']['not_supported']} not supported; summary {st['plain_fa']['summary']}")

    def am(tag, patches):
        files = " ".join(f"/patches/{p}" for p in patches)
        rc, text = r.sh(f"git am {files}", log=f"am-{tag}.log")
        if rc != 0:
            r.sh("git am --show-current-patch=diff | head -50; git status --short | head -30; git am --abort",
                 log=f"am-{tag}.log")
        r.sh("git log --oneline -4", log=f"am-{tag}.log")
        meta["stages"].setdefault(tag, {})["am_rc"] = rc
        return rc == 0

    elapsed = lambda: time.time() - r.t0

    if mode == "final":
        extra = sorted(os.path.basename(p) for p in glob.glob("/patches/0004-*.patch"))
        series = PATCHES + extra
        r.note(f"===== final: {' + '.join(series)} =====")
        configure()
        if am("final", series) and build("final"):
            run_tests("final", plain_fa=plain_fa_baseline, perf=True)
            # the plain-FA ceiling at the Inkling shapes (the perf cases 0002 adds next to the banded ones)
            r.sh("./build/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\\[4,1\\]'",
                 log="perf-final-plain-fa.log", timeout=900)
        else:
            meta["error"] = "final apply/build failed"
        r.sh("git log --oneline -6; git status --short | head", log="final-git.log")
        meta["elapsed_s"] = round(elapsed(), 1)
        r.note(f"done in {meta['elapsed_s']}s")
        return {"meta": meta, "logs": r.logs, "events": r.events}

    # ---- (a) PR head unpatched -------------------------------------------------------------------
    r.note("===== stage a: PR head unpatched =====")
    configure()
    if build("a"):
        # the baseline plain-FA run is optional; keep it only while the budget allows
        run_tests("a", plain_fa=plain_fa_baseline and elapsed() < 25 * 60, perf=False)
    else:
        meta["error"] = "stage a build failed"
        return {"meta": meta, "logs": r.logs, "events": r.events}

    # ---- (b) + 0001 + 0002 -----------------------------------------------------------------------
    r.note("===== stage b: + 0001 (Metal kernel) + 0002 (test cases) =====")
    if am("b", PATCHES[:2]) and build("b"):
        run_tests("b", plain_fa=False, perf=True)
    else:
        meta["error"] = "stage b apply/build failed"
        return {"meta": meta, "logs": r.logs, "events": r.events}

    # ---- (c) + 0003 ------------------------------------------------------------------------------
    r.note("===== stage c: + 0003 (q_offset) =====")
    if am("c", PATCHES[2:]) and build("c"):
        run_tests("c", plain_fa=True, perf=True)
    else:
        meta["error"] = "stage c apply/build failed"

    r.sh("git log --oneline -5; git status --short | head", log="final-git.log")
    meta["elapsed_s"] = round(elapsed(), 1)
    r.note(f"done in {meta['elapsed_s']}s")
    return {"meta": meta, "logs": r.logs, "events": r.events}


@app.local_entrypoint()
def main(arch: str = "90", plain_fa_baseline: bool = True, mode: str = "full", out_sub: str = ""):
    res = verify.remote(arch=arch, plain_fa_baseline=plain_fa_baseline, mode=mode)
    out = os.path.join(LOCAL_OUT, out_sub) if out_sub else LOCAL_OUT
    os.makedirs(out, exist_ok=True)
    for name, text in res["logs"].items():
        with open(os.path.join(out, name), "w") as f:
            f.write(text)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(res["meta"], f, indent=2)
    with open(os.path.join(out, "events.log"), "w") as f:
        f.write("\n".join(res["events"]) + "\n")
    print("\n===== SUMMARY =====")
    print(json.dumps({k: v for k, v in res["meta"].items() if k not in ("nvidia_smi", "nvcc")}, indent=2))
    print(f"logs written to {out}")
