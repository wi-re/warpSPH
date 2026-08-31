"""Onset-mechanism study of the `dfsphReference` free-surface degradation on
`hydrostaticColumn` (DFSPH_IMPROVEMENT_PLAN.md active track, Part 33).

By Part 33 the late-time surface collapse is isolated: it survives with the
divergence solve removed (`SKIP_DIVERGENCE_SOLVE`, so no inf-soup and a
correct hydrostatic bulk gradient), no cold start, no calibration -- a
pure surface-layer mechanism. This probe characterises it.

It wraps `dfsphReference_step` and, every `--interval` steps, reports:

- column geometry: base y, surface y (95th pct of fluid y), height, and how
  many fluid rows sit ABOVE the initial surface level (upward ejection);
- the diluted population: counts below rho 0.5 / 0.3 / 0.2, and for the
  rho < `--diluteThresh` cohort its mean height above the surface, mean
  speed, mean v_y (>0 = moving up), mean neighbour count;
- the ORIGINAL top-layer cohort (indices fixed at t=0): fraction still
  within 1.5 dx of the surface, their mean/min rho, mean and max y;
- kinetic energy of the top 10% (by height) vs the rest, to see whether the
  surface layer is where the slosh energy concentrates.

Usage:
    python scripts/probe_dfsphReferenceColumnSurface.py                 # plain IISPH
    python scripts/probe_dfsphReferenceColumnSurface.py --calibrate     # + rest-density calib
    python scripts/probe_dfsphReferenceColumnSurface.py --noSkipDiv     # DF solve back in
    python scripts/probe_dfsphReferenceColumnSurface.py --steps 1500 --interval 50
"""
import argparse

import torch

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.dfsphReference as ref
from warpSPH.modules.util import countNeighbors

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args.add_argument('--steps', type=int, default=800)
args.add_argument('--interval', type=int, default=25)
args.add_argument('--diluteThresh', type=float, default=0.30)
args.add_argument('--calibrate', action='store_true')
args.add_argument('--warmStart', action='store_true')
args.add_argument('--noSkipDiv', action='store_true',
                 help='keep the divergence solve (default: skip it, plain IISPH)')
args = args.parse_args()

_orig_step = ref.dfsphReference_step
_ctx = {}


def _wrapped_step(system, dt, config, schemeConfig, verbose=False):
    out = _orig_step(system, dt, config, schemeConfig, verbose=verbose)
    _update, adjacency, st, _rest = out
    _ctx['adj'] = adjacency
    _ctx['cfg'] = config
    _ctx['scfg'] = schemeConfig
    _ctx['n'] = _ctx.get('n', 0) + 1

    st_ = st
    fluid = st_.kinds == 0
    y = st_.positions[:, 1]
    v = st_.velocities
    rho = st_.densities
    p = st_.pressures

    if 'cohort' not in _ctx:
        yhi0 = float(y[fluid].max())
        dx = config.dx
        _ctx['dx'] = dx
        _ctx['yhi0'] = yhi0
        _ctx['cohort'] = fluid & (y > yhi0 - 1.5 * dx)
        _ctx['ybase0'] = float(y[fluid].min())

    n = _ctx['n']
    if n % args.interval != 0 and n != 1:
        return out

    dx = _ctx['dx']
    fy = y[fluid]
    surfY = float(torch.quantile(fy, 0.95))
    baseY = float(fy.min())
    height = surfY - baseY
    nAbove0 = int((fluid & (y > _ctx['yhi0'] + 0.25 * dx)).sum())

    nb = countNeighbors(st_, _ctx['cfg'], _ctx['scfg'], adjacency)

    def frac(thr):
        return int((fluid & (rho < thr)).sum())

    dil = fluid & (rho < args.diluteThresh)
    ndil = int(dil.sum())
    prevDil = _ctx.get('prevDil')
    nPersist = int((dil & prevDil).sum()) if prevDil is not None else 0
    _ctx['prevDil'] = dil
    if ndil:
        dyAbove = float((y[dil] - surfY).mean() / dx)
        spd = float(v[dil].norm(dim=-1).mean())
        vy = float(v[dil, 1].mean())
        nbd = float(nb[dil].float().mean())
    else:
        dyAbove = spd = vy = nbd = float('nan')

    fr = rho[fluid]
    rho5 = float(torch.quantile(fr, 0.05))
    # min density among rows that are NOT in the top 1 dx skin (drop the
    # ballistic-spray particles the plain `minDensity` metric is dominated by)
    embedded = fluid & (y < surfY - 1.0 * dx)
    bulkMinRho = float(rho[embedded].min()) if bool(embedded.any()) else float('nan')

    coh = _ctx['cohort']
    ncoh = int(coh.sum())
    cohNearSurf = int((coh & (y > surfY - 1.5 * dx)).sum())
    cohRhoMean = float(rho[coh].mean())
    cohRhoMin = float(rho[coh].min())
    cohYMean = float(y[coh].mean())
    cohYMax = float(y[coh].max())

    order = torch.argsort(fy)
    ntop = max(1, fluid.sum().item() // 10)
    topIdx = fluid.nonzero(as_tuple=True)[0][order[-ntop:]]
    botIdx = fluid.nonzero(as_tuple=True)[0][order[:-ntop]]
    m = st_.masses
    keTop = float((0.5 * m[topIdx] * (v[topIdx] ** 2).sum(-1)).sum())
    keRest = float((0.5 * m[botIdx] * (v[botIdx] ** 2).sum(-1)).sum())

    pSurf = float(p[fluid & (y > surfY - 1.5 * dx)].abs().mean())

    if n == 1:
        print(f'{"step":>5} {"surfY":>7} {"hght":>6} {"nUp":>4} '
              f'{"n<.5":>5} {"n<.3":>5} {"prst":>4} {"rho5":>6} {"bMinρ":>6} '
              f'{"dilΔy":>6} {"dilVy":>7} {"dilNb":>6} '
              f'{"cohNr":>6} {"keTop":>7} {"keRst":>7}')
    print(f'{n:>5} {surfY:>7.3f} {height:>6.3f} {nAbove0:>4} '
          f'{frac(0.5):>5} {frac(0.3):>5} {nPersist:>4} {rho5:>6.3f} {bulkMinRho:>6.3f} '
          f'{dyAbove:>6.2f} {vy:>+7.3f} {nbd:>6.1f} '
          f'{cohNearSurf:>3}/{ncoh:<2} {keTop:>7.4f} {keRest:>7.4f}')
    return out


ref.dfsphReference_step = _wrapped_step
ref.SKIP_DIVERGENCE_SOLVE = not args.noSkipDiv
ref.DAMPED_WARM_START = args.warmStart
try:
    print(f'dfsphReference  hydrostaticColumn  nx={args.nx}  steps={args.steps}  '
          f'skipDiv={not args.noSkipDiv}  calibrate={args.calibrate}  '
          f'warmStart={"damped" if args.warmStart else "full"}  '
          f'diluteThresh={args.diluteThresh}')
    r = run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx, nSteps=args.steps,
            scheme='dfsphReference', quiet=True, plot=False, store=False, progress=False,
            integrationScheme='semiImplicitEuler',
            params={'calibrateRestDensity': args.calibrate})
finally:
    ref.dfsphReference_step = _orig_step
    ref.SKIP_DIVERGENCE_SOLVE = False
    ref.DAMPED_WARM_START = False
print(f'diverged={r.diverged}  nSteps={_ctx.get("n")}')
