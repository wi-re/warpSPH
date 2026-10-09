#!/usr/bin/env bash
# OPEN_PROBLEMS.md current batch, step 1 (and the re-run after step 3): the
# cheap rechecks, sequentially, each with video + a streamed log + the run
# watch (velocity alarm, stallProgress).
#
#   §5    englishWedge dp=0.01 with the wedge, current default combo
#   §6.3  the 8 cases never re-run after the sampler mass fix, at their defaults
#
# Usage:  scripts/run_looseEndsBaseline.sh <label>     (e.g. baseline, afterDetectIsolated)
# Output: scripts/out_looseEnds/<label>/<case>/  (+ <case>.log, status.txt)
# Restartable: a case whose .done marker exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=/home/lu26029/miniconda3/envs/warp/bin/python
LABEL=${1:-baseline}
OUT=scripts/out_looseEnds/$LABEL
mkdir -p "$OUT"
STATUS=$OUT/status.txt
WATCH="--velocityAlarmPlotInterval 1 --stallProgress 1e-3"

runCase() {   # name, command...
    local name=$1; shift
    if [ -f "$OUT/$name.done" ]; then echo "skip $name (done)" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$OUT/$name.done"
}

runCase englishWedge $PY -u scripts/probe_englishWedge.py --dp 0.01 --wedge --tLimit 4.0 \
    --out "$OUT/englishWedge"

for c in kidder staticBlob squarePatch kelvinHelmholtz yee impact rayleighTaylor triplePoint; do
    runCase "$c" $PY -u -m warpSPHRun "$c" --plot --video --no-show --progress \
        --exportRoot "$OUT/$c" $WATCH
done
echo "$(date +%T) ALLDONE" >> "$STATUS"
