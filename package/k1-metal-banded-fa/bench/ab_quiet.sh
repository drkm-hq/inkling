#!/bin/zsh
# Interleaved llama-bench A/B of -fa 0 (dense chain) vs -fa 1 (Metal banded kernel) for the decode / prefill claims.
# pp512 and tg128 at depths 0 / 4096 / 16384 / 32768, three interleaved rounds (fa 1, fa 0, fa 0, fa 1, fa 1, fa 0),
# min and median per (test, fa) over the rounds, the 1-minute load average recorded before every run.
# Run on a QUIET machine (no other jobs, no renderer at 100% CPU): the decode numbers move by 50% with load.
# Holds bench/GPU_LOCK for the whole run (the lock is released by the exit trap, llama-bench exits by itself).
#
# usage: ab_quiet.sh <out-dir> [reps-per-run (default 2)] [depths (default 0,4096,16384,32768)]
set -u
LOCK=<workdir>/bench/GPU_LOCK
ROOT=<workdir>/upstream/llama.cpp-inkling
MODEL=~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf
OUT=$1
REPS=${2:-2}
DEPTHS=${3:-0,4096,16384,32768}
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/ab_quiet.log"; }

# the lock: free, or held by a dead process; then no other process above 20 GB
while true; do
    if [ -e "$LOCK" ]; then
        pid=$(awk '{print $1}' "$LOCK" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then sleep 15; continue; fi
        log "lock held by dead pid $pid, removing it"; rm -f "$LOCK"
    fi
    big=$(ps -axo pid,rss,comm | awk -v me=$$ 'NR > 1 && $2 > 20*1024*1024 && $1 != me {print $3"("int($2/1048576)"G)"}' | tr '\n' ' ')
    if [ -n "$big" ]; then log "waiting for large processes: $big"; sleep 15; continue; fi
    break
done
if ! ( set -o noclobber; print -r -- "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) ab_quiet.sh (llama-bench A/B)" > "$LOCK" ) 2>/dev/null; then
    log "lost the lock race"; exit 1
fi
trap 'rm -f "$LOCK"; log "lock released"' EXIT
log "lock acquired (pid $$), build $(git -C "$ROOT" rev-parse --short HEAD), reps $REPS, depths $DEPTHS"

run() { # <fa> <round>
    local load=$(sysctl -n vm.loadavg | awk '{print $2}')
    log "round $2 -fa $1 start, load1 $load"
    "$ROOT/build/bin/llama-bench" -m "$MODEL" -ngl 99 -fa $1 -p 512 -n 128 -d "$DEPTHS" -r $REPS -o jsonl \
        > "$OUT/bench-r$2-fa$1.jsonl" 2> "$OUT/bench-r$2-fa$1.err"
    log "round $2 -fa $1 exit=$?, load1 now $(sysctl -n vm.loadavg | awk '{print $2}')"
}
# alternate the order so neither setting always runs first after a pause
run 1 1; run 0 1
run 0 2; run 1 2
run 1 3; run 0 3

python3 -I - "$OUT" <<'PY'
import glob, json, os, statistics, sys
out = sys.argv[1]
rows = {}  # (test, depth, fa) -> [t/s per run]
for path in sorted(glob.glob(os.path.join(out, "bench-r*-fa*.jsonl"))):
    fa = int(path.rsplit("fa", 1)[1].split(".")[0])
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        test = "pp512" if d["n_prompt"] > 0 else "tg128"
        rows.setdefault((test, d["n_depth"], fa), []).extend(d["samples_ts"])
print("| test | depth | -fa 1 min / median (n) | -fa 0 min / median (n) | min ratio fa1/fa0 |")
print("|---|---:|---:|---:|---:|")
for test in ("pp512", "tg128"):
    for depth in sorted({k[1] for k in rows}):
        a = rows.get((test, depth, 1), []); b = rows.get((test, depth, 0), [])
        if not a or not b:
            continue
        fa1 = f"{max(a):.1f} / {statistics.median(a):.1f} ({len(a)})"
        fa0 = f"{max(b):.1f} / {statistics.median(b):.1f} ({len(b)})"
        print(f"| {test} | {depth} | {fa1} | {fa0} | {max(a)/max(b):.2f}x |")
print()
print("note: 'min' time = max tok/s over all runs of the rounds (contention only lowers tok/s)")
PY
