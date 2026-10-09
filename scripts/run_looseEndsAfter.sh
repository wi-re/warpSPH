#!/usr/bin/env bash
# OPEN_PROBLEMS.md current batch, step 3's before/after check for the shared
# detector's isolated-row flag (branch isolated-surface): every case the change
# can touch, same commands as the "before" runs.
#   before: scripts/out_looseEnds/baseline/ (englishWedge, staticBlob, squarePatch, impact)
#           scripts/out_baseline_2026-09-26/ (m31 nx67 deltaSPH / delta+, sloshingTank t=7)
# Output: scripts/out_looseEnds/<label>/ ; restartable (.done markers).
set -u
cd "$(dirname "$0")/.."
PY=/home/lu26029/miniconda3/envs/warp/bin/python
LABEL=${1:-afterIsolated}
OUT=scripts/out_looseEnds/$LABEL
mkdir -p "$OUT"
STATUS=$OUT/status.txt
WATCH="--velocityAlarmPlotInterval 1 --stallProgress 1e-3"

runCase() {
    local name=$1; shift
    if [ -f "$OUT/$name.done" ]; then echo "skip $name (done)" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$OUT/$name.done"
}

runCase englishWedge $PY -u scripts/probe_englishWedge.py --dp 0.01 --wedge --tLimit 4.0 --out "$OUT/englishWedge"
for c in staticBlob squarePatch impact; do
    runCase "$c" $PY -u -m warpSPHRun "$c" --plot --video --no-show --progress --exportRoot "$OUT/$c" $WATCH
done
runCase m31_dsph_nx67  $PY -u scripts/probe_deltaSPHMarrone.py --out "$OUT/m31" --nx 67 --tLimit 1.9 --scheme deltaSPH --shifting off --no-show
runCase m31_dplus_nx67 $PY -u scripts/probe_deltaSPHMarrone.py --out "$OUT/m31" --nx 67 --tLimit 1.9 --scheme sun2017DeltaSPH --shifting default --no-show
runCase slosh_wcsph_t7 $PY -u examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7.0 --out "$OUT/sloshing"
echo "$(date +%T) ALLDONE" >> "$STATUS"
