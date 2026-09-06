"""Probe (`PST_ALE_PLAN.md` Part 8 step 2 / `ACSPH_PLAN.md` step 7's own
"next action"): re-measure `hydrostaticColumn` under ACSPH now that the
Michel et al. 2022 PST exists, since `pairedFraction` and `‖v‖_max` are the
two numbers step 7 was supposed to move.

`ACSPH_PLAN.md` step 5b recorded, at nx=32, 200 steps to `t≈0.94`, with
particle shifting unwired (so always a no-op) and only the `noPenetrationShift`
mDBC safeguard varying:

    | | max |v| | worst corner | verdict |
    |---|---|---|---|
    | neither          | 2.9  | x=-0.62  | fluid leaving the box |
    | noPenetrationShift=True | 0.29 | x=-0.497 | bounded, but pairedFraction 0.065 |

This script adds a third mode -- the shift now actually runs
(`ShiftingScheme.michel2022` / `ShiftingProjectionScheme.michel2022`,
`noPenetrationShift=False` per the plan's own instruction: "check whether the
shift alone carries the corners" -- and a fourth, both together, since they
are not mutually exclusive.

Usage:
  python scripts/probe_michelHydrostaticColumn.py [--nx 32] [--nSteps 200]
      [--modes neither noPenetrationShift michelShift both]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=32)
parser.add_argument('--nSteps', type=int, default=200)
parser.add_argument('--modes', nargs='+',
                    default=['neither', 'noPenetrationShift', 'michelShift', 'both'],
                    choices=['neither', 'noPenetrationShift', 'michelShift', 'both'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.runner import run
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingScheme, ShiftingProjectionScheme


def _configure(mode):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        sc = ctx.schemeConfig
        if mode in ('noPenetrationShift', 'both'):
            sc.noPenetrationShift = True
        if mode in ('michelShift', 'both'):
            sc.shiftProperties.active = True
            sc.shiftProperties.scheme = ShiftingScheme.michel2022
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.michel2022

    return wrapped


hdr = (f"{'mode':>18} {'steps':>6} {'final t':>8} {'diverged':>9} "
       f"{'vMaxPeak':>9} {'pairedFrac':>11} {'nnDistP01':>10} "
       f"{'neighCV':>8} {'voidFrac':>9} {'dispMax':>8} {'KE':>10}")
print(hdr)
print('-' * len(hdr))

for mode in args.modes:
    _orig = case.configureScheme
    case.configureScheme = _configure(mode)
    try:
        r = run(case, scheme='artificialCompressible',
                nx=args.nx, nSteps=args.nSteps,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        case.configureScheme = _orig

    tr = [row for row in r.trajectory if 'maxVelocity' in row]
    if not tr:
        print(f"{mode:>18}   (no finite step)")
        continue

    t = np.array([row['t'] for row in tr])
    vmax = np.array([row['maxVelocity'] for row in tr])
    paired = np.array([row.get('pairedFraction', np.nan) for row in tr])
    nnd = np.array([row.get('nnDistP01', np.nan) for row in tr])
    ncv = np.array([row.get('neighbourCountCV', np.nan) for row in tr])
    voidF = np.array([row.get('voidFraction', np.nan) for row in tr])
    disp = np.array([row.get('dispMax', np.nan) for row in tr])
    ke = np.array([row.get('kineticEnergy', np.nan) for row in tr])

    print(f"{mode:>18} {len(tr)-1:6d} {t[-1]:8.3f} {str(r.diverged):>9} "
          f"{vmax.max():9.3f} {paired[-1]:11.4f} {nnd[-1]:10.3f} "
          f"{ncv[-1]:8.3f} {voidF[-1]:9.4f} {disp[-1]:8.3f} {ke[-1]:10.3e}",
          flush=True)
