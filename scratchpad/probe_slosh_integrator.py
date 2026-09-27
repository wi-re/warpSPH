#!/usr/bin/env python
"""sloshingTank: rungeKutta2 vs symplecticEuler on the CURRENT tree, to t*=3.

The run previously cited as evidence that the matched config is stable through
both impacts (t = 4.25) was rungeKutta2 -- it was launched before the case
default changed to symplecticEuler, which was only ever verified to t = 0.3.
"""
import sys, numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

tLimit = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
case = getCase('sloshingTank')
for integ in ('rungeKutta2', 'symplecticEuler'):
    r = run(case, scheme='deltaSPH', tLimit=tLimit, integrationScheme=integ,
            quiet=True, store=False, progress=False, plot=False)
    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    mn = [x.get('minDensity', 1.0) for x in rows]
    mx = [x.get('maxDensity', 1.0) for x in rows]
    vm = [x.get('maxVelocity', 0.0) for x in rows]
    tEnd = rows[-1].get('t', float('nan')) if rows else float('nan')
    print(f'{integ:16s} diverged={str(r.diverged):5s} steps={r.nSteps:6d} '
          f't={tEnd:.3f}  rho=[{np.nanmin(mn):.4f}, {np.nanmax(mx):.4g}]  '
          f'vmax={np.nanmax(vm):.4g}', flush=True)
