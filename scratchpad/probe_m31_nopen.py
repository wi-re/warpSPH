#!/usr/bin/env python
"""Marrone 3.1 under a chosen integrator, with the mDBC no-penetration shift
optionally zeroed.

`deltaSPH_step` turns a position correction into an acceleration via
`nopenshift / dt`, where `dt` is the *stage* dt -- so the term's magnitude
depends on the integrator's stage splitting, which a physical force must not.
`symplecticEuler` calls the derivative with dt/2, making it 2x stronger than a
full-step call. diffSPH disables the equivalent term outright (`/ dt * 0`).

  python scratchpad/probe_m31_nopen.py <integrator> <on|off> [shifting] [tLimit]
"""
import sys, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

integrator = sys.argv[1] if len(sys.argv) > 1 else 'symplecticEuler'
nopen      = (sys.argv[2] if len(sys.argv) > 2 else 'on') == 'on'
shifting   = sys.argv[3] if len(sys.argv) > 3 else 'on'
tLimit     = float(sys.argv[4]) if len(sys.argv) > 4 else 1.24

import warpSPH.schemes.deltaSPH as sch
if not nopen:
    sch.computeMdbcNoPenShift = lambda state, *a, **k: torch.zeros_like(state.positions)

from warpSPH.cases import importAll; importAll()
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run

H, TANK_L, G = 0.6, 1.0, 9.81
U_MAX = 1.95 * (G * H) ** 0.5
params = dict(W=3.2196, fillRatio=H / TANK_L, fluidWidth=1.2 / 3.2196,
              gravityMagnitude=G, pressureProbeHeights=[0.160, 0.584, 1.000],
              pressureProbeInset=0.0, pressureProbeDiscRadius=0.045,
              referenceVelocity=U_MAX, machTarget=U_MAX / (40.0 * (G * H) ** 0.5),
              freezeDiffusionAcrossStages=None,
              shifting=(shifting == 'on'))
print(f'integrator={integrator}  noPenShift={"ON" if nopen else "OFF"}  '
      f'PST={shifting}  tLimit={tLimit}', flush=True)
r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=67, tLimit=tLimit,
        integrationScheme=integrator, quiet=True, store=False, progress=False,
        plot=False, params=params)
rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
tS = [x.get('tStar', 0.0) for x in rows]
mn = [x.get('minDensity', 1.0) for x in rows]
mx = [x.get('maxDensity', 1.0) for x in rows]
vm = [x.get('maxVelocity', 0.0) for x in rows]
print(f'  diverged={r.diverged}  final t*={tS[-1]:.3f}  '
      f'rho=[{min(mn):.4f}, {max(mx):.4g}]  vmax={max(vm):.4g}  (U_max={U_MAX:.2f})')
