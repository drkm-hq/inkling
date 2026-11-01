#!/bin/zsh
# Pull the prompt-cache decision lines out of the server logs (they carry the model's bytes, hence grep -a):
# the reuse gate's verdicts, checkpoint creation/restoration and the per-request prompt eval timings.
set -u
HERE=$HOME/code/inkling-contrib/package/k3-llamacpp-prompt-cache/bench
for log in $HERE/results/*-server.log; do
  out=${log%-server.log}-reuse-lines.txt
  grep -a -n -E "forcing full prompt re-processing|restored context checkpoint|created context checkpoint|n_past = [0-9]+, slot.prompt|cache reuse|prompt eval time|cached n_tokens|new prompt," "$log" \
    | sed -E 's/^([0-9]+):[0-9.]+ [A-Z] /\1: /' > "$out"
  echo "$(basename $out): $(wc -l < $out | tr -d ' ') lines"
done
