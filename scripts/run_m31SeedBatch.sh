#!/bin/bash
# Marrone 3.1 delta-SPH (no PST) P1-plateau realisation check, H/dx = 40:
# 3 seeded IC jitters (1e-3 dx) + the unperturbed baseline npz already in the
# out dir. Restartable like run_baselineBatch.sh.
# Launch: setsid nohup scripts/run_m31SeedBatch.sh >/dev/null 2>&1 </dev/null &
cd /home/lu26029/dev/warpSPH
P=/home/lu26029/miniconda3/envs/warp/bin/python
O=scripts/out_baseline_2026-09-26
ST=$O/m31_seeds.status
run() {
  name=$1; shift
  if [ -f $O/$name.done ]; then echo "SKIP $name (done) $(date '+%F %T')" >> $ST; return; fi
  echo "START $name $(date '+%F %T')" >> $ST
  "$@" > $O/$name.log 2>&1; rc=$?
  [ $rc -eq 0 ] && touch $O/$name.done
  echo "END $name rc=$rc $(date '+%F %T') | $(grep -aE 'diverged' $O/$name.log | tail -1 | cut -c1-160)" >> $ST
}
echo "BATCH START $(date '+%F %T') $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)" >> $ST
for s in 1 2 3; do
  run m31s_dsph_nx67_j$s $P scripts/probe_deltaSPHMarrone.py --out $O/m31_seeds --nx 67 --tLimit 1.9 --scheme deltaSPH --shifting off --jitter 1e-3 --seed $s
done
run m31s_report $P scripts/probe_deltaSPHMarrone.py --out $O/m31_seeds --report
echo "ALLDONE $(date '+%F %T')" >> $ST
