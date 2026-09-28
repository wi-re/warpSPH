#!/usr/bin/env bash
# OPEN_PROBLEMS §8: the shared detector's isolated-row flag, on vs off, same
# code, Marrone 3.1 delta+ nx67 to t=1.9, three jittered realisations each.
# Sequential, restartable. Output: scripts/out_isolatedFlagAB/
set -u
cd "$(dirname "$0")/.."
PY=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=scripts/out_isolatedFlagAB; mkdir -p "$OUT"; STATUS=$OUT/status.txt
runCase() {
    local name=$1; shift
    if [ -f "$OUT/$name.done" ]; then echo "skip $name" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}; echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$OUT/$name.done"
}
for s in 1 2 3; do
    for flag in on off; do
        extra=""; [ $flag = off ] && extra="--noFlagIsolated"
        runCase m31_${flag}_j$s $PY -u scripts/probe_deltaSPHMarrone.py --out "$OUT/m31" --nx 67 --tLimit 1.9 \
            --scheme sun2017DeltaSPH --shifting default --jitter 1e-3 --seed $s --no-show $extra
    done
done
echo "$(date +%T) ALLDONE" >> "$STATUS"
