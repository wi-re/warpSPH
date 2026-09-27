#!/usr/bin/env python
"""Marrone 3.1 ablation: the two candidate causes of the delta+ wall-suction
blowup under `symplecticEuler`.

  clamp     'fallback' = current tree: `rho_b >= rho0` applied only to the
                         Shepard fallback share, so a well-conditioned MLS may
                         report sub-rho0 -> p_b < 0 -> the wall ATTRACTS fluid.
                         (Deliberate, to fix sloshingTank's density ratchet.)
            'blanket'  = the pre-80eabb9 guard: clamp the blended rho_b itself,
                         i.e. DualSPHysics' m2dbc anti-attraction guard.
  nopen     'on'/'off'  the mDBC no-penetration shift, which `deltaSPH_step`
                        scales as `nopenshift / dt` with the *stage* dt -- so it
                        is 2x stronger under symplecticEuler's dt/2 calls.

  python scratchpad/probe_m31_ablate.py <integrator> <nopen on|off> <clamp fallback|blanket> [pst] [tLimit]
"""
import sys, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

integrator = sys.argv[1] if len(sys.argv) > 1 else 'symplecticEuler'
nopen      = sys.argv[2] if len(sys.argv) > 2 else 'derivative'
clampMode  = sys.argv[3] if len(sys.argv) > 3 else 'fallback'
pst        = sys.argv[4] if len(sys.argv) > 4 else 'on'
tLimit     = float(sys.argv[5]) if len(sys.argv) > 5 else 1.24

import warpSPH.schemes.deltaSPH as sch
NOPEN_MODE = nopen if isinstance(nopen, str) else ('derivative' if nopen else 'off')

if clampMode == 'blanket':
    _orig = sch.computeMdbcDensity
    def clamped(currentState, config, schemeConfig, adjacency):
        rho = _orig(currentState, config, schemeConfig, adjacency)
        rho0 = schemeConfig.fluid.restDensity
        return torch.where(currentState.kinds == 1, rho.clamp_min(rho0), rho)
    sch.computeMdbcDensity = clamped

from warpSPH.cases import importAll; importAll()
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run

H, TANK_L, G = 0.6, 1.0, 9.81
U_MAX = 1.95 * (G * H) ** 0.5
params = dict(W=3.2196, fillRatio=H / TANK_L, fluidWidth=1.2 / 3.2196,
              gravityMagnitude=G, pressureProbeHeights=[0.160, 0.584, 1.000],
              pressureProbeInset=0.0, pressureProbeDiscRadius=0.045,
              referenceVelocity=U_MAX, machTarget=U_MAX / (40.0 * (G * H) ** 0.5),
              freezeDiffusionAcrossStages=None, shifting=(pst == 'on'),
              wallBC='constant')
tag = f'{integrator}/nopen={NOPEN_MODE}/clamp={clampMode}/pst={pst}'
def _setMode(ctx):
    ctx.schemeConfig.mdbcNoPenShiftMode = NOPEN_MODE
_prev = dambreakCase.configureScheme
def _cfg(ctx):
    _prev(ctx); _setMode(ctx)
dambreakCase.configureScheme = _cfg

r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=67, tLimit=tLimit,
        integrationScheme=integrator, quiet=True, store=False, progress=False,
        plot=False, params=params)
rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
tS = [x.get('tStar', 0.0) for x in rows]
mn = [x.get('minDensity', 1.0) for x in rows]
mx = [x.get('maxDensity', 1.0) for x in rows]
vm = [x.get('maxVelocity', 0.0) for x in rows]
npen = [x.get('nPenetrating', 0) for x in rows]
print(f'{tag:58s} diverged={str(r.diverged):5s} t*={tS[-1]:6.3f} '
      f'rho=[{min(mn):.4f},{max(mx):9.3g}] vmax={max(vm):9.3g} '
      f'nPen_max={max(npen) if npen else 0}', flush=True)
