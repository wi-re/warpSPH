#!/bin/bash
# delta-SPH lone-particle reset batch (MDBC_CONTACT_LINE_PLAN.md §13).
# Sequential, restartable: a run whose log already has its final
# "diverged=" line is skipped, so re-launching after an interruption resumes.
# Launch detached:  setsid nohup scripts/run_loneBatch.sh >/dev/null 2>&1 </dev/null &
# Progress:         cat scripts/out_contactLine/lone_batch.status
# Results:          scripts/out_contactLine/lone_batch_summary.md (rewritten after every run)
cd /home/lu26029/dev/warpSPH
P=/home/lu26029/miniconda3/envs/warp/bin/python
O=scripts/out_contactLine
ST=$O/lone_batch.status
J="--jitter 1e-3"
run() {
  log=$1; shift
  if [ -f $O/$log.log ] && grep -qa "diverged=" $O/$log.log; then
    echo "SKIP $log (done) $(date '+%F %T')" >> $ST; return; fi
  echo "START $log $(date '+%F %T')" >> $ST
  $P scripts/probe_contactLine.py --toy sloshing --scheme default --nx 0 --tLimit 4.2 --tag trace "$@" > $O/$log.log 2>&1
  echo "END $log rc=$? $(date '+%F %T') $(grep -a 'diverged=' $O/$log.log | tail -1 | cut -c1-120)" >> $ST
  $P scripts/summarize_loneBatch.py > /dev/null 2>&1
}
echo "BATCH START $(date '+%F %T')" >> $ST
run f_base_j1      $J --seed 1
run f_lone_j1      $J --seed 1 --loneReset
run f_base_j2      $J --seed 2
run f_lone_j2      $J --seed 2 --loneReset
run f_lonehyd_j1   $J --seed 1 --loneReset --oneSidedHydro
run f_base_j3      $J --seed 3
run f_lone_j3      $J --seed 3 --loneReset
$P scripts/summarize_loneBatch.py > /dev/null 2>&1
echo "ALLDONE $(date '+%F %T')" >> $ST
