#!/bin/bash
# CEILING_STICKING_PLAN.md §7.2: is the late P2 re-wetting under noPen 'impulse'
# systematic? Marrone 3.1 δ⁺ nx67, jitter seeds 4-6, drift fix on, noPen
# finalize vs impulse. One run at a time, video on.
cd /home/lu26029/dev/warpSPH
PY=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=scripts/out_ceiling/m31_seeds456
mkdir -p $OUT
for s in 4 5 6; do
  for m in finalize impulse; do
    $PY -u scripts/probe_deltaSPHMarrone.py --tLimit 1.9 --no-show --out $OUT --nx 67 \
        --scheme sun2017DeltaSPH --shifting default --densityDiffusionTerm fourtakas2019 \
        --jitter 1e-3 --seed $s --timeCentredContinuity --noPenShift $m 2>&1 \
        | stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -v "^\s*$" > $OUT/seed${s}_$m.log
    echo "seed $s $m done rc=${PIPESTATUS[0]}" >> $OUT/status.txt
  done
done
echo ALLDONE >> $OUT/status.txt
