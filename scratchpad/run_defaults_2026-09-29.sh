#!/bin/bash
# CEILING_STICKING_PLAN.md: the 2026-09-28 overnight default-combo set (fourtakas2019
# DDT only), re-run full length with both 2026-09-29 fixes on:
#   --timeCentredContinuity  (density as a drift field, §3/§6)
#   --noPenShift impulse     (noPen as a restitution impulse, §7.1)
# Same layout as scripts/out_overnight_2026-09-28, so
# scripts/summarize_overnight_2026-09-28.py summarises both for a like-for-like
# comparison. A (δ⁺ nx67 seeds 1-3) is linked in from scripts/out_ceiling/m31_impulse.
# Sequential, video on, restartable (<name>.done).
set -u
cd /home/lu26029/dev/warpSPH
PY=/home/lu26029/miniconda3/envs/warp/bin/python
BASE=scripts/out_ceiling/defaults_2026-09-29
mkdir -p "$BASE/m31"
STATUS=$BASE/status.txt
echo "$(date '+%F %T') batch start (commit $(git rev-parse --short HEAD), dirty tree)" >> "$STATUS"

runCase() {   # name, command...
    local name=$1; shift
    if [ -f "$BASE/$name.done" ]; then echo "$(date +%T) skip $name (done)" >> "$STATUS"; return; fi
    echo "$(date +%T) start $name" >> "$STATUS"
    "$@" 2>&1 | stdbuf -oL tr '\r' '\n' > "$BASE/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$(date +%T) end $name rc=$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] && touch "$BASE/$name.done"
}

for f in scripts/out_ceiling/m31_impulse/*_j?_tcc.npz scripts/out_ceiling/m31_impulse/*_j?_tcc_output.mp4; do
    ln -sf "$(realpath "$f")" "$BASE/m31/$(basename "$f")"
done

FIX="--timeCentredContinuity --noPenShift impulse"
M31="$PY -u scripts/probe_deltaSPHMarrone.py --tLimit 1.9 --no-show"
M34="$PY -u scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 15.6605"

runCase B_m31_dsph_nx67_fourtakas2019 $M31 --out $BASE/m31 --nx 67 --scheme deltaSPH --shifting off --densityDiffusionTerm fourtakas2019 $FIX
runCase C_m31_dplus_nx134_fourtakas2019 $M31 --out $BASE/m31 --nx 134 --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm fourtakas2019 $FIX
runCase D_m34_dsph_fourtakas2019  $M34 --out $BASE/m34 --scheme deltaSPH --shifting off --densityDiffusionTerm fourtakas2019 $FIX
runCase D_m34_dplus_fourtakas2019 $M34 --out $BASE/m34 --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm fourtakas2019 $FIX
runCase E_slosh_fourtakas2019 $PY -u examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7.0 \
    --densityDiffusionTerm fourtakas2019 --out $BASE/sloshing_fourtakas2019 $FIX

runCase G_summary $PY -u scripts/summarize_overnight_2026-09-28.py $BASE
echo "$(date '+%F %T') ALLDONE" >> "$STATUS"
