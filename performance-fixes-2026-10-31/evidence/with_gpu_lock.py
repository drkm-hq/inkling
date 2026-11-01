import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

lock_path = Path(__file__).with_name('GPU.lock')
with lock_path.open('a+') as lock:
    print(f'Waiting for GPU benchmark lock: pid={os.getpid()}', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print(f'Acquired GPU benchmark lock: pid={os.getpid()} time={time.time()}', flush=True)
    try:
        result = subprocess.run(sys.argv[1:])
    finally:
        print(f'Released GPU benchmark lock: pid={os.getpid()}', flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
sys.exit(result.returncode)
