"""The shared single-big-model-on-the-GPU protocol (same as bench/baseline/bench_mlx.py):
no other process above 20 GB RSS, and bench/GPU_LOCK absent or held by a dead PID."""
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
LOCK = BENCH_DIR / "GPU_LOCK"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def big_processes(min_rss_kb=20_000_000):
    out = subprocess.run(["ps", "-axo", "pid,rss,command"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss = int(parts[0]), int(parts[1])
        if rss > min_rss_kb and pid != os.getpid():
            rows.append((pid, rss, parts[2][:120]))
    return rows


def release():
    try:
        if LOCK.exists() and LOCK.read_text().split()[0] == str(os.getpid()):
            LOCK.unlink()
            log("GPU_LOCK released")
    except Exception as e:  # pragma: no cover
        log(f"lock release failed: {e}")


QUEUE = BENCH_DIR / "GPU_QUEUE"


def queue_head():
    """The first waiting agent in bench/GPU_QUEUE (None when the file is empty)."""
    try:
        lines = [l.strip() for l in QUEUE.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return None
    return lines[0] if lines else None


def gpu_free():
    """(free, reasons): the lock is absent or stale and nothing big is resident."""
    busy = []
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().split()[0])
        except Exception:
            pid = -1
        if pid > 0 and pid_alive(pid):
            busy.append(f"GPU_LOCK held by pid {pid}")
        else:
            log(f"stale GPU_LOCK (pid {pid}); removing")
            LOCK.unlink(missing_ok=True)
    for pid, rss, cmd in big_processes():
        busy.append(f"pid {pid} rss {rss / 2**20:.1f} GiB: {cmd}")
    return not busy, busy


def wait_for_turn(name, wait_s=12 * 3600, poll_s=20):
    """Block until ``name`` heads bench/GPU_QUEUE and the GPU is free."""
    t0 = time.time()
    last = None
    while True:
        head = queue_head()
        free, busy = gpu_free()
        if head == name and free:
            return
        msg = f"waiting: queue head {head!r}, " + ("GPU free" if free else f"GPU busy: {busy}")
        if msg != last:
            log(msg)
            last = msg
        if time.time() - t0 > wait_s:
            raise SystemExit(f"waited {wait_s}s: {msg}")
        time.sleep(poll_s)


def acquire(tag, wait_s=6 * 3600, poll_s=30):
    t0 = time.time()
    while True:
        busy = []
        if LOCK.exists():
            try:
                pid = int(LOCK.read_text().split()[0])
            except Exception:
                pid = -1
            if pid > 0 and pid_alive(pid):
                busy.append(f"GPU_LOCK held by pid {pid}")
            else:
                log(f"stale GPU_LOCK (pid {pid}); removing")
                LOCK.unlink(missing_ok=True)
        for pid, rss, cmd in big_processes():
            busy.append(f"pid {pid} rss {rss / 2**20:.1f} GiB: {cmd}")
        if not busy:
            LOCK.write_text(f"{os.getpid()} {time.strftime('%FT%T')} {tag}\n")
            atexit.register(release)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: sys.exit(1))
            log(f"GPU_LOCK acquired by pid {os.getpid()} ({tag})")
            return
        if time.time() - t0 > wait_s:
            raise SystemExit(f"GPU busy for {wait_s}s: {busy}")
        log(f"GPU busy, waiting: {busy}")
        time.sleep(poll_s)


def chunked_eval(params, chunk_gib=4.0):
    """Evaluate (array) parameters in ~chunk_gib pieces; a one-shot eval of ~80 GB trips
    the Metal GPU watchdog on this machine."""
    import mlx.core as mx

    chunk, nb, total = [], 0, 0
    t0 = time.perf_counter()
    for v in params:
        chunk.append(v)
        nb += v.nbytes
        if nb > chunk_gib * 2**30:
            mx.eval(chunk)
            total += nb
            chunk, nb = [], 0
    if chunk:
        mx.eval(chunk)
        total += nb
    return total, time.perf_counter() - t0


if __name__ == "__main__":
    # python gpulock.py wait K2   -> exits when K2 heads the queue and the GPU is free
    if len(sys.argv) == 3 and sys.argv[1] == "wait":
        wait_for_turn(sys.argv[2])
    else:
        raise SystemExit("usage: gpulock.py wait <agent>")
