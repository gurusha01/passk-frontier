#!/bin/bash
# Re-enumerates every run each cycle (no fixed glob), and reports:
#   CRASH  - traceback / OOM in the log
#   STALL  - log file not modified in > STALL_S seconds and no final/ checkpoint
#   DONE   - final/ checkpoint appeared
# Only transitions are emitted, so a steady state is silent.
D=${PF_ROOT}
STALL_S=${STALL_S:-900}
declare -A SEEN
while true; do
  now=$(date +%s)
  for f in $D/logs/*.log; do
    [ -e "$f" ] || continue
    a=$(basename "$f" .log)
    case "$a" in eval-*|_*) continue;; esac
    st=""
    if [ -d "$D/runs/$a/final" ]; then
      st=DONE
    elif grep -qE "Traceback|CUDA out of memory|OutOfMemoryError|Killed|AssertionError" "$f" 2>/dev/null; then
      st=CRASH
    else
      m=$(stat -c %Y "$f" 2>/dev/null || echo "$now")
      age=$((now - m))
      [ "$age" -gt "$STALL_S" ] && st="STALL(${age}s)"
    fi
    [ -z "$st" ] && continue
    if [ "${SEEN[$a]}" != "$st" ]; then
      SEEN[$a]=$st
      echo "$st $a"
      [ "$st" = CRASH ] && grep -E "Error|Traceback" "$f" | tail -2 | sed 's/^/    /'
    fi
  done
  sleep 120
done
