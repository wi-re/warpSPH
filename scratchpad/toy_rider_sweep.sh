#!/bin/bash
# CEILING_STICKING_PLAN.md §7: rider toy, v_x sweep through the wall-lattice resonance,
# noPen 'finalize' (current default) then 'impulse' (candidate). One run at a time.
cd /home/lu26029/dev/warpSPH
PY=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=scripts/out_ceiling/toy_rider
mkdir -p $OUT
VX="2 4 5 6 6.5 7 7.5 8 9 10 12"
for m in ${MODES:-finalize impulse}; do
    $PY -u scratchpad/toy_ceilingRider.py --vx $VX --vy0 0.02 --tLimit ${TL:-0.1} --noPen $m 2>&1 \
        | stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -v "^\s*$" > $OUT/sweep_$m.log
    echo "mode $m done rc=${PIPESTATUS[0]}" >> $OUT/status.txt
done
echo ALLDONE >> $OUT/status.txt
