#!/bin/bash
# CEILING_STICKING_PLAN.md §4: Marrone 3.1 δ⁺ nx67 fourtakas2019, seeds 1-3, with
# timeCentredContinuity -- same arguments as the overnight 2026-09-28 A-set.
cd /home/lu26029/dev/warpSPH
PY=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=scripts/out_ceiling/m31_tcc
mkdir -p $OUT
for s in 1 2 3; do
    $PY -u scripts/probe_deltaSPHMarrone.py --tLimit 1.9 --no-show --out $OUT --nx 67 \
        --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm fourtakas2019 \
        --jitter 1e-3 --seed $s --timeCentredContinuity 2>&1 \
        | stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -v "^\s*$" > $OUT/seed$s.log
    echo "seed $s done rc=${PIPESTATUS[0]}" >> $OUT/status.txt
done
echo ALLDONE >> $OUT/status.txt
