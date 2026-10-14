"""Contraction study for the linear DFSPH Jacobi (DFSPH_IMPROVEMENT_PLAN.md
Part 28's blocker: the iteration does not contract inside its budget).

The probe wraps `_jacobiSolve` and, at each production solve call, RE-DRIVES
the identical iteration loop (same state, source, warm start, factor, and the
same `applyConsistentCoupling` context the production step is inside) with a
swept relaxation `omega` and a fixed budget, logging the one-sided metric
per iteration. The production solve then runs as usual (omega 0.5, config
budget), so every sweep run sees identical states: each omega's trajectory is
a what-if on the same inputs, and the step-2 states are the production
ratchet's.

Per (step, mode, omega) it prints the err trajectory at selected iterations,
the pressure / acceleration magnitudes after the full budget, and a coarse
classification (DECAY / STAGNANT / GROW).

Usage:
    python scripts/probe_dfsphReferenceContraction.py [--nx 32] [--steps 2]
        [--omegas 0.5,0.3,0.1,0.05] [--budget 256]
"""
import argparse

import torch

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.incompressible.dfsphReference as ref

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args.add_argument('--steps', type=int, default=2)
args.add_argument('--omegas', type=str, default='0.5,0.3,0.1,0.05',
                  help='comma-separated relaxation values to sweep')
args.add_argument('--budget', type=int, default=256,
                  help='fixed iteration budget for the what-if trajectories')
args = args.parse_args()

OMEGAS = [float(x) for x in args.omegas.split(',')]
BUDGET = args.budget
LOG_ITERS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192}
LOG_ITERS.add(BUDGET)

# The current sweep run's omega and its recorded trajectories:
# (step, mode, [err at logged iters], p_max, |a_p|_max)
SWEEP_OMEGA = [None]
TRAJ = []


def _trajectory(state, config, schemeConfig, adjacency, *, vEnter, rho0,
                warmStart, opDt, solverCfg, minIters, fluidMask, mode, omega):
    """Re-drive the production Jacobi loop (Part 28 form) with `omega` for
    BUDGET iterations and no early exit, logging the one-sided metric."""
    dt = opDt if mode == 'divergence' else opDt ** 0.5
    invDiag = 1.0 / (opDt * (-ref._factor(state, config, schemeConfig, adjacency)))
    drhodtEnter = ref._drhodt(state, config, schemeConfig, adjacency, vEnter)
    if mode == 'density':
        source = 1.0 - state.densities / rho0 - dt * drhodtEnter / rho0
    else:
        source = -drhodtEnter / rho0
    guardThreshold = 7 if config.domain.dim == 2 else 20
    nNeighbours = ref.countNeighbors(state, config, schemeConfig, adjacency)
    deficient = fluidMask & (nNeighbours < guardThreshold)
    if mode == 'divergence':
        source = torch.where(deficient, torch.zeros_like(source), source)
    p = torch.where(fluidMask, warmStart, torch.zeros_like(warmStart))
    if mode == 'divergence':
        p = torch.where(deficient, torch.zeros_like(p), p)
    errs = []
    for it in range(BUDGET):
        a_p = ref._pressureAccel(state, config, adjacency, p, fluidMask)
        ap = ref._drhodt(state, config, schemeConfig, adjacency, a_p) / rho0 * opDt
        resid = source - ap
        p = (p - omega * resid * invDiag).clamp(min=0.0)
        p = torch.where(fluidMask, p, torch.zeros_like(p))
        residuum = torch.minimum(resid, torch.zeros_like(resid))
        if mode == 'divergence':
            residuum = torch.where(deficient, torch.zeros_like(residuum), residuum)
        err = float(rho0 * (-residuum)[fluidMask].mean())
        if (it + 1) in LOG_ITERS:
            errs.append((it + 1, err))
    a_p = ref._pressureAccel(state, config, adjacency, p, fluidMask)
    return errs, p, a_p


_orig_js = ref._jacobiSolve
_CALLS = [0]


def _spy_js(state, config, schemeConfig, adjacency, **kw):
    step = _CALLS[0] // 2 + 1
    mode = kw['mode']
    if SWEEP_OMEGA[0] is not None:
        errs, p, a = _trajectory(
            state, config, schemeConfig, adjacency,
            omega=SWEEP_OMEGA[0],
            vEnter=kw['vEnter'], rho0=kw['rho0'], warmStart=kw['warmStart'],
            opDt=kw['opDt'], solverCfg=kw['solverCfg'], minIters=kw['minIters'],
            fluidMask=kw['fluidMask'], mode=mode)
        pmax = float(p[kw['fluidMask']].max())
        amax = float(a.norm(dim=-1).max())
        TRAJ.append((step, mode, errs, pmax, amax))
    _CALLS[0] += 1
    return _orig_js(state, config, schemeConfig, adjacency, **kw)


def _classify(errs):
    if not errs:
        return 'n/a'
    first = errs[0][1]
    last = errs[-1][1]
    peak = max(e for _, e in errs)
    if peak <= 0.0:
        return 'FLAT0'
    if last > 10.0 * min(e for _, e in errs):
        return 'GROW'
    if last < 0.1 * first:
        return 'DECAY'
    return 'STAGNANT'


def _run(omega, steps):
    SWEEP_OMEGA[0] = omega
    TRAJ.clear()
    _CALLS[0] = 0
    ref._jacobiSolve = _spy_js
    try:
        run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx, nSteps=steps,
            scheme='dfsphReference', quiet=True, plot=False, store=False,
            progress=False, integrationScheme='semiImplicitEuler')
    finally:
        ref._jacobiSolve = _orig_js
    for step, mode, errs, pmax, amax in TRAJ:
        tag = mode[:2].upper()
        pts = '  '.join(f'{it:>3d}:{e:.2e}' for it, e in errs)
        print(f'omega={omega:<5g} step={step} {tag}  {_classify(errs):<8s}  '
              f'p_max={pmax:.3g}  |a_p|max={amax:.3g}\n          {pts}')
    print()


print(f'contraction study  hydrostaticColumn nx={args.nx}  '
      f'budget={BUDGET}  omegas={OMEGAS}')
print('metric: one-sided rho0*mean(max(0, aij_pj - s)) over fluid; '
      'no early exit (full budget)')
print()
for omega in OMEGAS:
    _run(omega, args.steps)
