#!/bin/bash
# Default-settings baseline re-check: Marrone 3.1, Marrone 3.4, sloshingTank,
# all with video, current case/scheme defaults (no override flags besides the
# two validation configs: delta-SPH no-PST and delta+-SPH (Sun 2017) PST on;
# under the default symplecticEuler integrator sun2017DeltaSPH's frozen
# diffusion is a no-op, so both legs are un-frozen).
# Sequential, restartable: a run with a .done marker is skipped on relaunch.
# Launch detached:  setsid nohup scripts/run_baselineBatch.sh >/dev/null 2>&1 </dev/null &
# Progress:         cat scripts/out_baseline_2026-09-26/batch.status
#                   tail -f scripts/out_baseline_2026-09-26/<run>.log
cd /home/lu26029/dev/warpSPH
P=/home/lu26029/miniconda3/envs/warp/bin/python
O=scripts/out_baseline_2026-09-26
ST=$O/batch.status
mkdir -p $O/m31 $O/m34 $O/sloshing
run() {
  name=$1; shift
  if [ -f $O/$name.done ]; then echo "SKIP $name (done) $(date '+%F %T')" >> $ST; return; fi
  echo "START $name $(date '+%F %T')" >> $ST
  "$@" > $O/$name.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && touch $O/$name.done
  echo "END $name rc=$rc $(date '+%F %T') | $(grep -aE 'diverged' $O/$name.log | tail -1 | cut -c1-160)" >> $ST
}
echo "BATCH START $(date '+%F %T') $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)" >> $ST
# Marrone 3.1 (H/dx = 0.6 nx = 40, Marrone Fig. 5), full record t = 1.9 s (t* 7.7)
run m31_dsph_nx67   $P scripts/probe_deltaSPHMarrone.py --out $O/m31 --nx 67 --tLimit 1.9 --scheme deltaSPH --shifting off
run m31_dplus_nx67  $P scripts/probe_deltaSPHMarrone.py --out $O/m31 --nx 67 --tLimit 1.9 --scheme sun2017DeltaSPH --shifting default
# Marrone 3.4 full protocol (H/dx = nx/8 = 32), t* = 15.66
run m34_dsph_nx256  $P scripts/probe_deltaSPHMarrone34.py --out $O/m34 --nx 256 --tStar 15.6605 --scheme deltaSPH --shifting off
run m34_dplus_nx256 $P scripts/probe_deltaSPHMarrone34.py --out $O/m34 --nx 256 --tStar 15.6605 --scheme sun2017DeltaSPH --shifting default
# sloshingTank (SPHERIC TC10), case defaults (nx=200), full 7 s record vs experiment
run slosh_wcsph_t7  $P examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7.0 --out $O/sloshing
# Marrone 3.1 at H/dx = 80 (convergence point)
run m31_dsph_nx134  $P scripts/probe_deltaSPHMarrone.py --out $O/m31 --nx 134 --tLimit 1.9 --scheme deltaSPH --shifting off
run m31_dplus_nx134 $P scripts/probe_deltaSPHMarrone.py --out $O/m31 --nx 134 --tLimit 1.9 --scheme sun2017DeltaSPH --shifting default
# scored reports over everything that finished
run m31_report $P scripts/probe_deltaSPHMarrone.py --out $O/m31 --report
run m34_report $P scripts/probe_deltaSPHMarrone34.py --out $O/m34 --report
echo "ALLDONE $(date '+%F %T')" >> $ST
