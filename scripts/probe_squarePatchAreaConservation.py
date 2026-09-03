"""Probe (`WCSPH_SHIFTING_PLAN.md` step 1): how badly does the δ⁺-SPH surface
shift inflate the fluid footprint on the rotating square patch, and does the
current "switch it off at the surface" mitigation actually bound it?

The δ⁺-SPH shift `δx = −CFL·Ma·2h²·∇C` is not volume-preserving: near a free
surface the truncated kernel makes `−∇C` point outward, so surface particles
ratchet into the void and the footprint grows while mass is conserved. This
runs `squarePatch --scheme deltaSPH` under a matrix of shift treatments and
reports the drift in the area/volume metrics
(`cases.weaklyCompressible.squarePatchAreaMetrics`).

Modes (`--modes`):

  shiftOff        `shiftProperties.active = False`          -- the floor: no shift.
  surfaceZeroed   default                                   -- shift on, suppressed
                                                               near the surface
                                                               (today's behaviour).
  surfaceActive   `surfaceDetectionConfig.active = False`   -- shift on, NOT
                                                               suppressed: the raw
                                                               drift the mitigation
                                                               is fighting.
  surfaceNormal   `projectionScheme = surfaceNormal`        -- Sun 2019 §2.4
                                                               Eq. (20)-(21): the real
                                                               surface treatment (§3).
  deltaU          default + `shiftProperties.correctdrhodt` -- Sun 2019 §2d: feed
                                                               the shift into the
                                                               continuity equation.
  surfaceNormalDeltaU  surfaceNormal + correctdrhodt        -- the §4 target config.

`--shape box` is the benchmark; `--shape circle` is the null experiment -- a
circle in rigid rotation *is* an equilibrium, so any area drift there is pure
shift artifact.

Per `sun2019` §3.3, a fully free-surface-bounded body barely shows *volume*
drift (p≈0 on the surface leaves nothing to inflate), so `hullArea` /
`rmsRadius` spreading and `cornerRetention` are the sharper signals here;
the volume-drift signal proper lives in the periodic cases (`tgv`).

Usage:
  python scripts/probe_squarePatchAreaConservation.py [--nx 48 [96 ...]]
      [--shapes box circle] [--modes shiftOff surfaceZeroed surfaceActive]
      [--tLimit 0.5]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, nargs='+', default=[48])
parser.add_argument('--shapes', nargs='+', default=['box', 'circle'])
parser.add_argument('--modes', nargs='+',
                    default=['shiftOff', 'surfaceZeroed', 'surfaceActive',
                             'surfaceNormal', 'surfaceNormalDeltaU'],
                    choices=['shiftOff', 'surfaceZeroed', 'surfaceActive',
                             'surfaceNormal', 'deltaU', 'surfaceActiveDeltaU',
                             'surfaceNormalDeltaU'])
parser.add_argument('--tLimit', type=float, default=0.5)
parser.add_argument('--omega', type=float, default=4.0)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase as case
from warpSPH.runner import run

_METRICS = ['sphVolume', 'hullArea', 'rmsRadius', 'surfaceFraction', 'cornerRetention']


def _configure(mode):
    """Return a `configureScheme` wrapper that applies `mode` on top of the
    case's own scheme configuration."""
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        sc = ctx.schemeConfig
        from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme
        if mode == 'shiftOff':
            sc.shiftProperties.active = False
        elif mode == 'surfaceZeroed':
            pass
        elif mode == 'surfaceActive':
            sc.surfaceDetectionConfig.active = False
        elif mode == 'surfaceNormal':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal
        elif mode == 'deltaU':
            sc.shiftProperties.correctdrhodt = True
        elif mode == 'surfaceActiveDeltaU':
            sc.surfaceDetectionConfig.active = False
            sc.shiftProperties.correctdrhodt = True
        elif mode == 'surfaceNormalDeltaU':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal
            sc.shiftProperties.correctdrhodt = True

    return wrapped


def _rate(t, y):
    """Least-squares slope dy/dt over the run, in 'fraction of the initial
    value per unit time' -- the volume-growth rate the plan asks for."""
    t, y = np.asarray(t), np.asarray(y)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3 or y[ok][0] == 0:
        return float('nan')
    return float(np.polyfit(t[ok], y[ok], 1)[0] / abs(y[ok][0]))


# Two blocks: the drift rates (the plan's deliverable), then the raw
# start->end change per metric as a percentage of the initial value.
hdr = (f"{'shape':>7} {'nx':>4} {'mode':>19} {'steps':>6} {'div':>4}  "
       f"{'dHull/dt':>9} {'dVol/dt':>9}  "
       + " ".join(f"{'Δ'+m+'%':>17}" for m in _METRICS)
       + f"  {'minRho':>8} {'maxRho':>8}")
print(hdr)
print('-' * len(hdr))

for shape in args.shapes:
    for nx in args.nx:
        for mode in args.modes:
            _orig = case.configureScheme
            case.configureScheme = _configure(mode)
            try:
                # nSteps=None -> the runner derives the step count from
                # tLimit / dt (squarePatch has no adaptive `timestep` hook).
                r = run(case, params={'shape': shape, 'omega': args.omega},
                        nx=nx, tLimit=args.tLimit, nSteps=None,
                        store=False, plot=False, quiet=True, progress=False)
            finally:
                case.configureScheme = _orig

            t = r.series('t')
            cells = []
            for m in _METRICS:
                s = r.series(m)
                fin = s[np.isfinite(s)]
                if fin.size == 0 or fin[0] == 0:
                    cells.append(f"{'--':>17}")
                else:
                    pct = 100.0 * (fin[-1] - fin[0]) / abs(fin[0])
                    cells.append(f"{fin[0]:7.4f}{pct:+8.3f}%")
            minRho, maxRho = r.series('minDensity'), r.series('maxDensity')
            print(f"{shape:>7} {nx:>4} {mode:>19} {r.nSteps:>6} "
                  f"{('yes' if r.diverged else 'no'):>4}  "
                  f"{_rate(t, r.series('hullArea')):>9.2e} "
                  f"{_rate(t, r.series('sphVolume')):>9.2e}  "
                  + " ".join(cells)
                  + f"  {minRho[-1]:8.5f} {maxRho[-1]:8.5f}")
