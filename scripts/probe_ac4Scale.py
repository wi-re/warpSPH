"""Probe (ACSPH_PLAN.md §4.1.1 / authors' question): is standalone AC-4 unstable because the
code scales Eq. (35) with `h^2 = supports^2` (the kernel SUPPORT RADIUS, = xi * smoothing length)
while the paper's `h` is the smoothing length (`acParameters` uses `supports / xi` for beta and k2)?

Runs AC-4 on `hydrostaticColumn` (walled, nx 24) for a list of multipliers on the biharmonic
prefactor (1 = the code as it is; 1/xi^2 = the paper's `h^2` if its `h` is the smoothing length;
smaller = the AC-JST cap 1/32 ...), and prints boundedness / the final |v|.

Usage: python scripts/probe_ac4Scale.py [--nx 24] [--nSteps 60] [--scales 1 0.25 0.0625 0.03125]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=24)
ap.add_argument('--nSteps', type=int, default=60)
ap.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.25, 0.0625, 0.03125])
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import warpSPH.modules.artificialCompressible.pressureSmoothing as ps
from warpSPH.cases.boundaries.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.enumTypes import DensityDiffusionScheme, PressureSmoothingScheme
from warpSPH.modules.deltaSPH import computeScalarFieldDiffusion
from warpSPH.runner import run

SCALE = [1.0]


def biharmonicScaled(currentState, config, adjacency, p):
    dDelta = computeScalarFieldDiffusion(currentState, config, adjacency, DensityDiffusionScheme.densityOnly, field=p)
    h2 = SCALE[0] * currentState.supports ** 2
    return -h2 * computeScalarFieldDiffusion(currentState, config, adjacency, DensityDiffusionScheme.densityOnly, field=dDelta)


ps._biharmonic = biharmonicScaled
base = case.configureScheme


def configure(ctx):
    base(ctx)
    ctx.schemeConfig.acParams.pressureSmoothing = PressureSmoothingScheme.biharmonic


case.configureScheme = configure
print(f"{'scale':>8} {'steps':>6} {'final t':>8} {'|v| max':>10} {'p min':>10} {'p max':>10}", flush=True)
for s in args.scales:
    SCALE[0] = s
    r = run(case, scheme='artificialCompressible', nx=args.nx, nSteps=args.nSteps, store=False, plot=False,
            quiet=True, progress=False, cudaGraph=True)
    tr = [row for row in r.trajectory if 'maxVelocity' in row]
    v = np.array([row['maxVelocity'] for row in tr])
    print(f"{s:8.5f} {len(tr) - 1:6d} {tr[-1]['t']:8.4f} {np.nanmax(v):10.3g} "
          f"{tr[-1].get('minPressure', float('nan')):10.3g} {tr[-1].get('maxPressure', float('nan')):10.3g}", flush=True)
