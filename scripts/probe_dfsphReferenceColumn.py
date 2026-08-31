"""Per-step / sweep diagnostic for `dfsphReference` on `hydrostaticColumn`
(DFSPH_IMPROVEMENT_PLAN.md Part 24's harden track, step 1: the Akinci boundary
pressure force).

The reference DFSPH scheme holds the exact hydrostatic gradient for ~7 steps
at nx=32 and then wall-adjacent `kappa` runs away: the constant-density solve
exits clean on the *mean* residual (2 iterations) while a handful of
wall-adjacent particles ratchet their pressure up without bound, because the
boundary term in `A*p` -- the feedback that tells the one-sided drive a
standing `kappa` is already doing its job -- under-resolves the wall. On this
codebase's multi-layer `BOUNDED_BAND` the Akinci volume `m~_k = rho0/sum_l
W_kl` comes out numerically equal to the nominal particle volume, so the
paper's correction is inert; carrying the boundary apparent volume at ~2x
removes the runaway.

This probe wraps `_jacobiSolve` to print, per step:

- the constant-density (CD) and divergence-free (DF) solves: iterations, final
  error, full-field and wall-band (bottom 3 dx) pressure range, max |a_p|;
- the fluid velocity max and the free-surface min density (the surface
  compaction that becomes the next blocker once the wall is fixed -- step 3).

`--sweep` runs `akinciBoundaryVolumeScale in {1, 1.5, 2, 3, 4}` to `--steps`
and prints the wall-runaway / surface-compaction figures of merit for each.

Usage:
    python scripts/probe_dfsphReferenceColumn.py [--nx 32] [--steps 30]
    python scripts/probe_dfsphReferenceColumn.py --sweep [--steps 40]
"""
import argparse

import torch

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.dfsphReference as ref

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args.add_argument('--steps', type=int, default=30)
args.add_argument('--sweep', action='store_true')
args.add_argument('--scale', type=float, default=None,
                 help='force akinciBoundaryVolumeScale (default: the scheme picks 2.0)')
args.add_argument('--gauge', action='store_true',
                  help='free-surface kappa^v gauge (Part 30, step 3): hold '
                       'kappa^v = 0 on detectFreeSurface-flagged rows in the '
                       'divergence solve')
args.add_argument('--warmStart', action='store_true',
                  help='reference damped warm start (Part 31): seed each '
                       'solve with 0.5*min(carried, cap)/dt**k gated on the '
                       'row being compressed, instead of the full-kappa carry')
args.add_argument('--xspH', type=float, default=0.0,
                  help='reference XSPH fluid coefficient (Part 32): the '
                       'velocity filter that damps kernel-scale shear in '
                       'the bulk (0.0 = off)')
args.add_argument('--xspHBoundary', type=float, default=0.0,
                  help='reference XSPH boundary coefficient (Part 32): the '
                       'wall drag -- damps tangential sliding against the '
                       'static boundary (0.0 = off)')
args = args.parse_args()

_orig_js = ref._jacobiSolve
_orig_step = ref.dfsphReference_step
_orig_acc = ref.applyConsistentCoupling


def _scale_forcing_coupling(scaleOverride):
    """Wrap the context manager the solve reads the boundary volume scale
    through, so a probe value wins over the scheme's own `== 1.0 -> 2.0`
    default (which mutates the shared config each step)."""
    def cm(particles, config, schemeConfig, adjacency, mode):
        if scaleOverride is not None:
            schemeConfig.solverConfig.akinciBoundaryVolumeScale = scaleOverride
        return _orig_acc(particles, config, schemeConfig, adjacency, mode)
    return cm


def _spy_js(state, config, schemeConfig, adjacency, **kw):
    out = _orig_js(state, config, schemeConfig, adjacency, **kw)
    a_p, p, nit, err = out
    fl = kw['fluidMask']
    pos = state.positions

    def _rng(mask):
        return f'{float(p[mask].min()):+.2f},{float(p[mask].max()):+.2f}' if bool(mask.any()) else 'empty'

    ylo = pos[fl, 1].min()
    wall = fl & (pos[:, 1] < ylo + 3 * config.dx)
    tag = kw['mode'][:2].upper()
    print(f'   {tag} it={nit:2d} err={err:.2e}  n_fluid={int(fl.sum())}  '
          f'p[{_rng(fl)}] pWall[{_rng(wall)}] '
          f'|a_p|max={float(a_p.norm(dim=-1).max()):.2f}', end='')
    if kw['mode'] == 'divergence':
        vmax = float(state.velocities[fl].norm(dim=-1).max())
        rho = state.densities[fl]
        sm = kw.get('surfaceMask')
        gtag = f'  fsGauge={int((sm & fl).sum())}' if sm is not None else ''
        print(f'   |v|max={vmax:.3f}  rho[{float(rho.min()):.3f},{float(rho.max()):.3f}]{gtag}')
    else:
        print()
    return out


def _run(scale, steps, verbose):
    ref.applyConsistentCoupling = _scale_forcing_coupling(scale)
    ref._jacobiSolve = _spy_js if verbose else _orig_js
    ref.FREE_SURFACE_GAUGE = args.gauge
    ref.DAMPED_WARM_START = args.warmStart
    ref.XSPH_FLUID_EPSILON = args.xspH
    ref.XSPH_BOUNDARY_EPSILON = args.xspHBoundary
    try:
        return run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx, nSteps=steps,
                   scheme='dfsphReference', quiet=True, plot=False, store=False,
                   progress=False, integrationScheme='semiImplicitEuler')
    finally:
        ref.applyConsistentCoupling = _orig_acc
        ref._jacobiSolve = _orig_js
        ref.FREE_SURFACE_GAUGE = False
        ref.DAMPED_WARM_START = False
        ref.XSPH_FLUID_EPSILON = 0.0
        ref.XSPH_BOUNDARY_EPSILON = 0.0


if args.sweep:
    print(f'{"scale":>6} {"diverged":>9} {"|v|max_run":>11} {"disp_final":>11} '
          f'{"rhoMin_final":>12} {"slope@end":>10}')
    for sc in (1.0, 1.5, 2.0, 3.0, 4.0):
        r = _run(sc, args.steps, verbose=False)
        tr = r.trajectory
        vmax = max(x.get('maxVelocity', 0.0) for x in tr)
        print(f'{sc:>6.1f} {str(r.diverged):>9} {vmax:>11.3g} '
              f'{tr[-1].get("dispMax", float("nan")):>11.3g} '
              f'{tr[-1].get("minDensity", float("nan")):>12.3g} '
              f'{tr[-1].get("pressureSlopeRatio", float("nan")):>10.3g}')
else:
    print(f'dfsphReference  hydrostaticColumn  nx={args.nx}  '
          f'scale={"scheme default (2.0)" if args.scale is None else args.scale}  '
          f'gauge={"on" if args.gauge else "off"}  '
          f'warmStart={"damped" if args.warmStart else "full"}  '
          f'xspH={args.xspH:g}  xspHBoundary={args.xspHBoundary:g}')
    r = _run(args.scale, args.steps, verbose=True)
    print(f'diverged={r.diverged}')
