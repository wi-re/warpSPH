"""Probe (`WCSPH_SHIFTING_PLAN.md` §4 transfer test): does the Sun 2019
surface treatment (`ShiftingProjectionScheme.surfaceNormal`) let
`sloshingTank --scheme deltaSPH` survive the first violent wave impact?

`sloshingTank` runs with particle shifting **off** by default and NaNs around
`t ≈ 3.5 s` on the first slam (a free-surface particle-distribution
instability). The δ⁺ shift with the legacy hard-zero-at-surface projection
(`mat`) does not help — it isn't shifting the surface. `surfaceNormal` is the
real Eq. (20)-(21) treatment; this checks whether it clears the divergence.

Modes:
  noShift            `shiftProperties.active = False`  (the case default)
  shiftZeroed        shifting on, `projectionScheme = mat`  (legacy hard-zero)
  shiftSurfaceNormal shifting on, `projectionScheme = surfaceNormal`  (§3)

Reports, per mode: steps run, final `t`, whether it diverged, and the density
bounds + wall-sensor pressure at the last finite step.

Usage:
  python scripts/probe_sloshingTankSurfaceShift.py [--nx 60] [--tLimit 4.0]
      [--modes noShift shiftZeroed shiftSurfaceNormal]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=60)
parser.add_argument('--tLimit', type=float, default=4.0)
parser.add_argument('--modes', nargs='+',
                    default=['noShift', 'shiftZeroed', 'shiftSurfaceNormal'],
                    choices=['noShift', 'shiftZeroed', 'shiftSurfaceNormal'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.sloshingTank import sloshingTankCase as case
from warpSPH.runner import run
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme


def _configure(mode):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        sc = ctx.schemeConfig
        if not hasattr(sc, 'shiftProperties'):
            return
        if mode == 'noShift':
            sc.shiftProperties.active = False
        elif mode == 'shiftZeroed':
            sc.shiftProperties.active = True
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.mat
        elif mode == 'shiftSurfaceNormal':
            sc.shiftProperties.active = True
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal

    return wrapped


hdr = (f"{'mode':>18} {'steps':>6} {'final t':>9} {'diverged':>9} "
       f"{'minRho':>9} {'maxRho':>9} {'sensorP':>10}")
print(hdr)
print('-' * len(hdr))

for mode in args.modes:
    _orig = case.configureScheme
    case.configureScheme = _configure(mode)
    try:
        r = run(case, params={'shifting': mode != 'noShift'},
                nx=args.nx, tLimit=args.tLimit,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        case.configureScheme = _orig

    t = r.series('t')
    minRho = r.series('minDensity')
    maxRho = r.series('maxDensity')
    sensorP = r.series('sensorPressure') if 'sensorPressure' in (r.trajectory[-1] if r.trajectory else {}) else np.array([np.nan])
    finalT = float(t[np.isfinite(t)][-1]) if t.size else float('nan')
    lastFiniteRho = np.where(np.isfinite(minRho))[0]
    li = lastFiniteRho[-1] if lastFiniteRho.size else -1
    print(f"{mode:>18} {r.nSteps:>6} {finalT:>9.4f} "
          f"{('YES' if r.diverged else 'no'):>9} "
          f"{minRho[li]:>9.4f} {maxRho[li]:>9.4f} "
          f"{(sensorP[li] if li < sensorP.size else float('nan')):>10.1f}")
