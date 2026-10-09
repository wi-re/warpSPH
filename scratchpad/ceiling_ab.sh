#!/bin/bash
# CEILING_STICKING_PLAN.md: resume the seed-3 checkpoint with a patch file and a tag.
#   [RUNDIR=scripts/out_ceiling/seed2_drift] scratchpad/ceiling_ab.sh <tag> <offsetStep> <nSteps> [extra ceiling_forensics.py args...]
# (pass --seed N with a RUNDIR from another seed; add --patch scratchpad/patch_timeCentredContinuity.py for the drift fix)
cd /home/lu26029/dev/warpSPH
TAG=$1; OFF=$2; N=$3; shift 3
CK=$(ls ${RUNDIR:-scripts/out_ceiling/seed3}/*_run/*/trajectory/state_${OFF}.h5)
LOG=scripts/out_ceiling/forensics/${TAG}.log
/home/lu26029/miniconda3/envs/warp/bin/python -u scratchpad/ceiling_forensics.py \
    --resumeFrom "$CK" --offset "$OFF" --nSteps "$N" --tag "$TAG" "$@" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -v "^\s*$" > "$LOG"
grep -E "patch\]|Traceback|Error|diverged=|forensics\]" "$LOG" | cut -c1-220
grep -A15 Traceback "$LOG" | head -30
