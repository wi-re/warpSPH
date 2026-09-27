#!/usr/bin/env python
"""sloshingTank: reflecting vs projecting free-slip, with `u_body` fixed.

Both legs run the corrected `_ghostBodyVelocity` (rigid-body-only, so u_body = 0
for this case's static walls); the only difference is whether the fluid's normal
component is reflected (published, `u_g = w_t - w_n`) or projected out (the
historical form, `u_g = w_t`). Isolates the 5.8 change from the 5.2 compounding
bug, which contaminated the earlier baseline.

  python scratchpad/probe_slosh_slipform.py <reflect|project> [tLimit]
"""
import sys, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

form   = sys.argv[1] if len(sys.argv) > 1 else 'reflect'
tLimit = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

import warpSPH.modules.mdbc.velocity as vel
if form == 'project':
    def projectingFreeSlip(currentState, config, schemeConfig, adjacency):
        qVel = vel._shepardFluidVelocity(currentState, config, adjacency)
        bodyVelocity = vel._ghostBodyVelocity(currentState, schemeConfig)
        bIndices = currentState.ghostIndices[currentState.kinds == 2]
        n_b = vel._wallNormals(currentState)
        w = qVel - bodyVelocity
        w_n = torch.einsum('nd, nd -> n', w, n_b).view(-1, 1) * n_b
        u_g = bodyVelocity + (w - w_n)              # projecting: no  - w_n
        out = currentState.velocities.clone()
        out[bIndices, :] = u_g[currentState.kinds == 2, :]
        return out
    vel.freeSlip = projectingFreeSlip

from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run
import numpy as np

r = run(getCase('sloshingTank'), scheme='deltaSPH', tLimit=tLimit,
        quiet=True, store=False, progress=False, plot=False)
rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
mn = [x.get('minDensity', 1.0) for x in rows]
mx = [x.get('maxDensity', 1.0) for x in rows]
vm = [x.get('maxVelocity', 0.0) for x in rows]
tEnd = rows[-1].get('t', float('nan')) if rows else float('nan')
print(f'freeSlip form = {form:8s}  diverged={str(r.diverged):5s} '
      f'steps={r.nSteps}  t={tEnd:.3f}  '
      f'rho=[{np.nanmin(mn):.4f}, {np.nanmax(mx):.4g}]  '
      f'vmax={np.nanmax(vm):.4g}', flush=True)
