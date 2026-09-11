#!/usr/bin/env python
"""Is `deltaSPH_step`'s `nopenshift / dt` why symplecticEuler diverges?

That term converts a position correction into an acceleration by dividing by the
*stage* dt, so its magnitude depends on the integrator's stage splitting -- which
a physical force must not. `symplecticEuler` calls the derivative with dt/2, so
it enters 2x stronger than under a full-step call. diffSPH disables the
equivalent outright (`mDBCPenetrationCheck(...) / dt * 0`).

  python scratchpad/probe_slosh_nopen.py <integrator> <on|off> [tLimit]
"""
import sys, numpy as np, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

integ  = sys.argv[1] if len(sys.argv) > 1 else 'symplecticEuler'
nopen  = (sys.argv[2] if len(sys.argv) > 2 else 'on') == 'on'
tLimit = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5

import warpSPH.schemes.deltaSPH as sch
if not nopen:
    sch.computeMdbcNoPenShift = lambda state, *a, **k: torch.zeros_like(state.positions)

from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

r = run(getCase('sloshingTank'), scheme='deltaSPH', tLimit=tLimit,
        integrationScheme=integ, quiet=True, store=False, progress=False, plot=False)
rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
mn = [x.get('minDensity', 1.0) for x in rows]
mx = [x.get('maxDensity', 1.0) for x in rows]
vm = [x.get('maxVelocity', 0.0) for x in rows]
tEnd = rows[-1].get('t', float('nan')) if rows else float('nan')
print(f'{integ:16s} noPenShift={"ON " if nopen else "OFF"}  '
      f'diverged={str(r.diverged):5s} steps={r.nSteps:6d} t={tEnd:.3f}  '
      f'rho=[{np.nanmin(mn):.4f}, {np.nanmax(mx):.4g}]  vmax={np.nanmax(vm):.4g}',
      flush=True)
