#!/bin/bash
# Overnight batch 2026-09-28: re-validate the default WCSPH combo after the
# fourtakas2019 sign fix (68a9a6d, OPEN_PROBLEMS.md §5 in RESOLVED_PROBLEMS.md),
# then the full example gallery as a regression net.
#
# Every targeted run pairs the (now fixed) default DDT `fourtakas2019` with the
# pre-2026-09-18 `deltaSPH` DDT, everything else at today's defaults
# (english2025 mDBC density, symplecticEuler, free-slip walls, finalize no-pen).
# The fourtakas2019 default was chosen on 2026-09-18 with the sign bug in; this
# batch is the evidence for keeping it or not.
#
#   A  Marrone 3.1 nx67  delta+ (sun2017DeltaSPH, PST default), 3 jittered seeds per DDT
#   B  Marrone 3.1 nx67  deltaSPH, PST off (Marrone's own spec), one per DDT
#   C  Marrone 3.1 nx134 delta+, one per DDT
#   D  Marrone 3.4 nx256 deltaSPH PST off + delta+ PST default, per DDT (09-26 baseline commands)
#   E  sloshingTank t=7, per DDT
#   F  full gallery (render_examples.py, every family, shipped settings), outputs
#      redirected under $BASE -- nothing under examples/ is touched
#   G  scripts/summarize_overnight_2026-09-28.py -> $BASE/SUMMARY.md
#
# Sequential (one GPU run at a time), video on, restartable: a finished run
# leaves $BASE/<name>.done and is skipped on relaunch. Started with setsid nohup
# so it survives the Claude session ending:
#   setsid nohup scratchpad/run_overnight_batch_2026-09-28.sh > /dev/null 2>&1 &
# Progress: $BASE/status.txt ; per-run logs: $BASE/<name>.log
set -u
cd /home/lu26029/dev/warpSPH
PY=/home/lu26029/miniconda3/envs/warp/bin/python
BASE=scripts/out_overnight_2026-09-28
mkdir -p "$BASE"
STATUS=$BASE/status.txt
echo "$(date '+%F %T') batch start (commit $(git rev-parse --short HEAD))" >> "$STATUS"

runCase() {   # name, command...
    local name=$1; shift
    if [ -f "$BASE/$name.done" ]; then echo "$(date +%T) skip $name (done)" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$BASE/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$BASE/$name.done"
}

M31="$PY -u scripts/probe_deltaSPHMarrone.py --tLimit 1.9 --no-show"
M34="$PY -u scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 15.6605"

for ddt in fourtakas2019 deltaSPH; do
    for s in 1 2 3; do
        runCase A_m31_dplus_nx67_${ddt}_j$s $M31 --out $BASE/m31 --nx 67 --scheme sun2017DeltaSPH \
            --shifting default --densityDiffusionTerm $ddt --jitter 1e-3 --seed $s
    done
done
for ddt in fourtakas2019 deltaSPH; do
    runCase B_m31_dsph_nx67_$ddt $M31 --out $BASE/m31 --nx 67 --scheme deltaSPH --shifting off --densityDiffusionTerm $ddt
done
for ddt in fourtakas2019 deltaSPH; do
    runCase C_m31_dplus_nx134_$ddt $M31 --out $BASE/m31 --nx 134 --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm $ddt
done
for ddt in fourtakas2019 deltaSPH; do
    runCase D_m34_dsph_$ddt  $M34 --out $BASE/m34 --scheme deltaSPH --shifting off --densityDiffusionTerm $ddt
    runCase D_m34_dplus_$ddt $M34 --out $BASE/m34 --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm $ddt
done
for ddt in fourtakas2019 deltaSPH; do
    runCase E_slosh_$ddt $PY -u examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7.0 \
        --densityDiffusionTerm $ddt --out $BASE/sloshing_$ddt
done

runCase F_gallery $PY -u scripts/render_examples.py --outRoot $BASE/gallery --publishRoot $BASE/gallery/published

runCase G_summary $PY -u scripts/summarize_overnight_2026-09-28.py $BASE
echo "$(date '+%F %T') ALLDONE" >> "$STATUS"
