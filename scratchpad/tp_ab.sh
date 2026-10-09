#!/bin/bash
# usage: tp_ab.sh label args...   -> first step with vmax>5 (or clean end)
S=/tmp/claude-598314/-home-lu26029-dev-warpSPH/9eb6c28d-fcdd-4d3d-a7da-5581b2b4af34/scratchpad
label=$1; shift
timeout 600 /home/lu26029/miniconda3/envs/warp/bin/python -u scratchpad/tp_probe.py $S/tp "$@" 2>&1 | stdbuf -oL tr '\r' '\n' | grep -a -E "^\[probe\] dt|Traceback|Error" | awk -v L="$label" '{ if ($0 ~ /Error|Traceback/) {print L": ERR " $0; f=1; exit} match($0,/vmax=[^ ]+/); v=substr($0,RSTART+5,RLENGTH-5)+0; n++; if (v>5 && !f) {print L": blow at step " n " t? " $2 " vmax " v; f=1; exit} } END{ if(!f) print L": no blowup in " n " steps, last vmax " v }'
