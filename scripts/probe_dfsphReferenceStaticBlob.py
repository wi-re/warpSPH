"""Regression guard for the faithful DFSPH factor (Part 27): run `staticBlob`
under `dfsphReference` and confirm the blob stays quiescent (`|v|` ~ 0, shape
unchanged). Part 26 warned the linear optimal-step solve "regresses staticBlob
hard", so the factor change (which rescales the Jacobi step by ~1/rho) needs the
same guard. The factor change must not manufacture motion out of a fluid at
rest.

`--factor dfsph|alpha` toggles the Jacobi factor: `dfsph` is the new faithful
`computeDFSPHFactor` (Part 27, step 2); `alpha` is the pre-change IISPH
`computeAlpha` diagonal, for the A/B. Both return the same <= 0 sign convention.

Usage:
    python scripts/probe_dfsphReferenceStaticBlob.py [--nx 128] [--steps 20]
    python scripts/probe_dfsphReferenceStaticBlob.py --factor alpha --steps 20
"""
import argparse

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=128)
args.add_argument('--steps', type=int, default=30)
args.add_argument('--factor', choices=['dfsph', 'alpha'], default='dfsph',
                  help='Jacobi factor: dfsph (new, faithful) or alpha (baseline)')
args.add_argument('--gauge', action='store_true',
                  help='free-surface kappa^v gauge (Part 30, step 3): hold '
                       'kappa^v = 0 on detectFreeSurface-flagged rows in the '
                       'divergence solve (the case runs freeSurface=True)')
args.add_argument('--warmStart', action='store_true',
                  help='reference damped warm start (Part 31): seed each '
                       'solve with 0.5*min(carried, cap)/dt**k gated on the '
                       'row being compressed, instead of the full-kappa carry')
args = args.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
from warpSPH.cases import staticBlob
from warpSPH.runner import run
import warpSPH.schemes.dfsphReference as ref

_orig_factor = ref._factor


def _factor_alpha(state, config, schemeConfig, adjacency):
    """Pre-change factor: the IISPH `computeAlpha` diagonal (returned negated)."""
    from warpSPH.modules.incompressible.wp_alpha import computeAlpha
    apparentArea = state.masses / state.densities
    alpha = computeAlpha(state, config, schemeConfig, adjacency,
                         apparentVolumes=apparentArea,
                         includeBoundaryReaction=False)
    return torch.clamp(alpha, max=-1e-8)


ref._factor = _factor_alpha if args.factor == 'alpha' else _orig_factor
ref.FREE_SURFACE_GAUGE = args.gauge
ref.DAMPED_WARM_START = args.warmStart
try:
    r = run(staticBlob.staticBlobCase, nx=args.nx, nSteps=args.steps,
            scheme='dfsphReference', quiet=True, plot=False, store=False,
            progress=False, integrationScheme='semiImplicitEuler')
finally:
    ref._factor = _orig_factor
    ref.FREE_SURFACE_GAUGE = False
    ref.DAMPED_WARM_START = False
tr = r.trajectory
vmax = max(x.get('maxVelocity', 0.0) for x in tr)
dmax = max(x.get('dispMax', 0.0) for x in tr)
print(f'dfsphReference  staticBlob  factor={args.factor}  '
      f'gauge={"on" if args.gauge else "off"}  '
      f'warmStart={"damped" if args.warmStart else "full"}  nx={args.nx}  '
      f'steps={args.steps}  diverged={r.diverged}')
print(f'  run max |v|={vmax:.3g}   run max disp={dmax:.3g}')
last = tr[-1]
print(f'  final: |v|max={last.get("maxVelocity", float("nan")):.3g}  '
      f'dispMax={last.get("dispMax", float("nan")):.3g}  '
      f'centroidDrift={last.get("centroidDrift", float("nan")):.3g}  '
      f'rho[{last.get("minDensity", float("nan")):.3g},'
      f'{last.get("maxDensity", float("nan")):.3g}]  '
      f'KE={last.get("kineticEnergy", float("nan")):.3g}')
