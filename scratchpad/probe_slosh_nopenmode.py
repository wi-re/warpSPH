#!/usr/bin/env python
"""sloshingTank across the three mdbcNoPenShiftMode placements."""
import sys, numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

tLimit = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
integ = sys.argv[2] if len(sys.argv) > 2 else 'rungeKutta2'
case = getCase('sloshingTank')
_prev = case.configureScheme
for mode in ('derivative', 'finalize', 'off'):
    def _cfg(ctx, _m=mode):
        _prev(ctx); ctx.schemeConfig.mdbcNoPenShiftMode = _m
    case.configureScheme = _cfg
    r = run(case, scheme='deltaSPH', tLimit=tLimit, integrationScheme=integ,
            quiet=True, store=False, progress=False, plot=False)
    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    mn = [x.get('minDensity', 1.0) for x in rows]
    mx = [x.get('maxDensity', 1.0) for x in rows]
    vm = [x.get('maxVelocity', 0.0) for x in rows]
    tEnd = rows[-1].get('t', float('nan')) if rows else float('nan')
    print(f'{integ}/{mode:11s} diverged={str(r.diverged):5s} steps={r.nSteps:6d} '
          f't={tEnd:.3f} rho=[{np.nanmin(mn):.4f}, {np.nanmax(mx):.4g}] '
          f'vmax={np.nanmax(vm):.4g}', flush=True)
