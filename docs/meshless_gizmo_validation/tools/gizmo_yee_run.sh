#!/bin/bash
# GIZMO (MFM and MFV) on our exported Yee ICs
S=/tmp/claude-598314/-home-lu26029-dev-warpSPH/5ce3ee03-2802-4745-be6b-fd8b0f716ffe/scratchpad
R=/home/lu26029/dev/warpSPH/results/gizmo_compare_2026-10-12
for ic in yee_nx64_j0.0 yee_nx64_j0.0_cons yee_nx64_j0.25 yee_nx64_j0.25_cons; do
  for v in mfm mfv; do
    d=$R/$ic/$v; mkdir -p $d/output; cd $d
    sed -e "s#^InitCondFile .*#InitCondFile  $R/ics/${ic}_ics#" -e 's#^TimeMax .*#TimeMax 2#' -e 's#^TimeBetSnapshot .*#TimeBetSnapshot 1.0#' \
        -e 's#^BoxSize .*#BoxSize 10#' -e 's#^TimeBetStatistics .*#TimeBetStatistics 1.0#' $S/kh_template.params > run.params
    mpirun -np 4 --oversubscribe $S/gizmo_yee_$v/GIZMO run.params > run.out 2> run.err
    echo "$ic $v exit $? snaps: $(ls output | grep -c snapshot)" >> $R/yee_runs.log
  done
done
echo done >> $R/yee_runs.log
