#!/bin/bash
# Wait until the 1-minute load average is below $MAXLOAD, then run the chain as RUN=$RUN.
MAXLOAD=${MAXLOAD:-8}
cd "$(dirname "$0")"
while :; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l" -v m="$MAXLOAD" 'BEGIN{exit !(l<m)}'; then break; fi
  echo "$(date +%H:%M:%S) load $l >= $MAXLOAD, waiting"; sleep 20
done
echo "$(date +%H:%M:%S) load $l, starting RUN=$RUN"
RUN=$RUN ./run_all.sh
