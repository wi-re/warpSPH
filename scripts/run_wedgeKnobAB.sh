#!/usr/bin/env bash
# OPEN_PROBLEMS §5 (2026-09-28): englishWedge dp=0.01 base-corner residual,
# 0.0178 on 2026-09-12 -> 0.0674 today. Revert one default at a time to the
# 2026-09-12 configuration, then all four together. Sequential, restartable.
set -u
cd "$(dirname "$0")/.."
PY=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=scripts/out_wedgeKnobAB; mkdir -p "$OUT"; STATUS=$OUT/status.txt
runCase() {
    local name=$1; shift
    if [ -f "$OUT/$name.done" ]; then echo "skip $name" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}; echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$OUT/$name.done"
}
W="$PY -u scripts/probe_englishWedge.py --dp 0.01 --wedge --tLimit 4.0 --out $OUT/runs"
runCase mdbc_ramped   $W --mdbcDensityScheme ramped
runCase ddt_deltaSPH  $W --densityDiffusionTerm deltaSPH
runCase rk4           $W --integrationScheme rungeKutta4
runCase wall_constant $W --wallBC constant
runCase old_combo     $W --mdbcDensityScheme ramped --densityDiffusionTerm deltaSPH --integrationScheme rungeKutta4 --wallBC constant
echo "$(date +%T) ALLDONE" >> "$STATUS"
