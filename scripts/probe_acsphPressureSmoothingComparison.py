"""Probe (`ACSPH_PLAN.md` Part 8 step 8, De Courcy et al.'s Figs. 2/16): a
head-to-head of all four pressure-smoothing operators (AC-2, AC-2L, AC-4,
AC-JST) on `hydrostaticColumn` -- the paper's own operator discriminator
(§4.1.1): AC-2 cannot hold a hydrostatic gradient and diffuses the free
surface, AC-2L/AC-4/AC-JST all should.

Reuses `hydrostaticDiagnostics`'s own figures of merit, already built for
exactly this (`ACSPH_PLAN.md` step 5b): `pressureSlopeRatio` (1.0 = exact
hydrostatic `dp/dy = -rho0 g`) and `pressureResidual` (rms departure from
the fitted line, normalised by the column's own pressure drop -- the
free-surface/noise axis, independent of the slope).

Not a literal reproduction of Fig. 2's `t=50s` -- that horizon is far beyond
what this CPU-bound prototype can run in one sitting at any reasonable
resolution (see `probe_acsphOscillatingDropletTable1.py`'s own note on per-
step cost). Runs a fixed, moderate step count instead and reports whatever
happens, divergence included -- AC-4 is already known
(`modules/artificialCompressible/pressureSmoothing.py`) to diverge run
standalone at this resolution, and this script is partly what surfaced
that, not something to hide by cutting the run short.

Usage:
  python scripts/probe_acsphPressureSmoothingComparison.py [--nx 32]
      [--nSteps 100]
      [--schemes laplacian renormalizedBiLaplacian biharmonic jst]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=32)
parser.add_argument('--nSteps', type=int, default=100)
parser.add_argument('--schemes', nargs='+',
                     default=['laplacian', 'renormalizedBiLaplacian', 'biharmonic', 'jst'],
                     choices=['laplacian', 'renormalizedBiLaplacian', 'biharmonic', 'jst'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.runner import run
from warpSPH.enumTypes import PressureSmoothingScheme

_NAMES = {
    'laplacian': 'AC-2', 'renormalizedBiLaplacian': 'AC-2L',
    'biharmonic': 'AC-4', 'jst': 'AC-JST',
}


def _configure(mode):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        ctx.schemeConfig.acParams.pressureSmoothing = mode

    return wrapped


hdr = (f"{'scheme':>8} {'steps':>6} {'div':>4} {'final t':>8} "
       f"{'‖v‖ max':>9} {'pSlopeRatio':>11} {'pResidual':>10}")
print(hdr)
print('-' * len(hdr))

for name in args.schemes:
    mode = getattr(PressureSmoothingScheme, name)
    _orig = case.configureScheme
    case.configureScheme = _configure(mode)
    try:
        r = run(case, scheme='artificialCompressible', nx=args.nx, nSteps=args.nSteps,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        case.configureScheme = _orig

    tr = [row for row in r.trajectory if 'maxVelocity' in row]
    if not tr:
        print(f"{_NAMES[name]:>8}   (no finite step)")
        continue

    vmax = np.array([row['maxVelocity'] for row in tr])
    slopeRatio = tr[-1].get('pressureSlopeRatio', float('nan'))
    residual = tr[-1].get('pressureResidual', float('nan'))
    print(f"{_NAMES[name]:>8} {len(tr) - 1:6d} {('yes' if r.diverged else 'no'):>4} "
          f"{tr[-1]['t']:8.3f} {np.nanmax(vmax):9.3g} {slopeRatio:11.4f} {residual:10.4f}",
          flush=True)
