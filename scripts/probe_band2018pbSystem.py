"""Offline analysis of the `band2018pb` extended-PPE system `A p = s` over the
concatenated unknown `p = [p_f ; p_b]` (DFSPH_IMPROVEMENT_PLAN.md active track,
"Next" item 1 -- Band et al. 2018).

Captures the system at a chosen solve index from a running scheme and, on that
one representative system, reports:

  * V0 / V ranges (fluid + boundary), source stats, diagonal stats
    (median |diag|, sign) -- is the near-wall block still rank-deficient?
  * symmetry defect  |<Ax,y> - <x,Ay>| / (||Ax|| ||y||)
  * relaxed Jacobi (clamped) true residual floor over many iterations
  * MINRES / CG / BiCGStab: iterations + true ||s - A x||_2 / ||s||_2 + ||x||

Compare against `probe_omniIncompressibleCDSymmetry.py` on the same case: the
whole point of `band2018pb` is that this operator is NOT near-singular at the
wall.

    python scripts/probe_band2018pbSystem.py                       # hydrostaticColumn nx=32 solve#10
    python scripts/probe_band2018pbSystem.py --nx 64 --step 30
    python scripts/probe_band2018pbSystem.py --case randomFlowBounded --step 5
"""
import argparse

import torch

argp = argparse.ArgumentParser()
argp.add_argument('--case', default='hydrostaticColumn',
                  choices=['hydrostaticColumn', 'dambreak', 'randomFlowBounded'])
argp.add_argument('--nx', type=int, default=32)
argp.add_argument('--step', type=int, default=10, help='solve index to capture (1-based)')
argp.add_argument('--maxiter', type=int, default=400)
argp.add_argument('--rtol', type=float, default=1e-8)
args = argp.parse_args()

import warpSPH.schemes.band2018pb as bmod
from warpSPH.runner import run
from warpSPH.modules.incompressible.pressureBoundaries import (
    bandActualVolumes, bandApplyOperator, bandDiagonal, bandInjectVolumes,
    bandPressureAccel, bandRelaxation, bandRestVolumes, bandVelocityDivergence)
from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.cg import cgSolve
from warpSPH.modules.shifting.minres import minresSolve

_orig = bmod._solve
_n = {'i': 0}
_done = {'v': False}


def _rand(mask, seed):
    g = torch.Generator(device=mask.device).manual_seed(seed)
    x = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    x[mask] = torch.randn(int(mask.sum()), generator=g, device=mask.device)
    x[mask] -= x[mask].mean()
    return x


def spy(state, config, schemeConfig, adjacency, *, solveRows, fluid, rho0,
        vStar, warmStart, dt):
    _n['i'] += 1
    if _done['v'] or _n['i'] != args.step:
        return _orig(state, config, schemeConfig, adjacency, solveRows=solveRows,
                     fluid=fluid, rho0=rho0, vStar=vStar, warmStart=warmStart, dt=dt)
    _done['v'] = True
    boundary = solveRows & ~fluid

    V0 = bandRestVolumes(state, config, adjacency, rho0)
    V = bandActualVolumes(state, config, adjacency, V0, rho0)
    omega = bandRelaxation(state, V0, rho0, bmod.OMEGA_FLUID)

    with bandInjectVolumes(state, V):
        divVstar = bandVelocityDivergence(state, config, adjacency, vStar, solveRows)
        s = (1.0 - V0 / V.clamp_min(1e-12)) + dt * divVstar
        s = torch.where(solveRows, s, torch.zeros_like(s))
        diag = bandDiagonal(state, config, schemeConfig, adjacency, V, dt)
        precond = torch.where(solveRows, 1.0 / diag.clamp(max=-1e-30),
                              torch.zeros_like(diag))

        def A(pt):
            pt = torch.where(solveRows, pt, torch.zeros_like(pt))
            a_p = bandPressureAccel(state, config, adjacency, pt, V, fluid)
            return bandApplyOperator(state, config, adjacency, a_p, dt, solveRows)

        sNorm = float(s.norm())
        print(f'{args.case} nx={args.nx}  band2018pb solve #{args.step}   '
              f'Nfluid={int(fluid.sum())}  Nbnd={int(boundary.sum())}')
        print(f'  V0  fluid[{float(V0[fluid].min()):.3g},{float(V0[fluid].max()):.3g}]'
              f'  bnd[{float(V0[boundary].min()):.3g},{float(V0[boundary].max()):.3g}]')
        print(f'  V   fluid[{float(V[fluid].min()):.3g},{float(V[fluid].max()):.3g}]'
              f'  bnd[{float(V[boundary].min()):.3g},{float(V[boundary].max()):.3g}]')
        print(f'  1-V0/V  fluid[{float((1-V0/V)[fluid].min()):+.3g},'
              f'{float((1-V0/V)[fluid].max()):+.3g}]  '
              f'bnd[{float((1-V0/V)[boundary].min()):+.3g},{float((1-V0/V)[boundary].max()):+.3g}]')
        print(f'  |s|={sNorm:.4g}  s fluid mean {float(s[fluid].mean()):+.3g}  '
              f'bnd mean {float(s[boundary].mean()):+.3g}')
        print(f'  dt^2*a_ii  median|f|={float(diag[fluid].abs().median()):.3g}  '
              f'median|b|={float(diag[boundary].abs().median()):.3g}  '
              f'sign: f>0 {int((diag[fluid]>0).sum())}/{int(fluid.sum())}  '
              f'b>0 {int((diag[boundary]>0).sum())}/{int(boundary.sum())}')

        print('\n--- symmetry defect  |<Ax,y> - <x,Ay>| / (||Ax|| ||y||) ---')
        pairs = [(_rand(solveRows, 10 + k), _rand(solveRows, 500 + k)) for k in range(5)]
        ds = []
        for x, y in pairs:
            Ax, Ay = A(x), A(y)
            num = abs(float((Ax * y).sum()) - float((x * Ay).sum()))
            ds.append(num / max(float(Ax.norm() * y.norm()), 1e-30))
        print(f'  norm-rel {sum(ds) / len(ds):.3e}')

        def trueres(x):
            return float((s - A(x)).norm() / max(sNorm, 1e-30))

        def line(tag, x, nit, st=None):
            sfx = '' if st is None else f'  status={st}'
            print(f'  {tag:26s} it={nit:4d}  |r|/|s|={trueres(x):.3e}  '
                  f'|x|max={float(x.abs().max()):.4g}  |x|2={float(x.norm()):.4g}{sfx}')

        # Hydrostatic residual check: does A p_hydro ~ s for the analytic
        # column pressure?  (scale sanity for Eq. 8 + the operator.)
        y = state.positions[:, 1]
        ysurf = float(y[fluid].max())
        pHy = torch.where(solveRows, (rho0 * 9.81 * (ysurf - y)).clamp(min=0.0),
                          torch.zeros_like(y))
        ApHy = A(pHy)
        m = solveRows
        print(f'\n  p_hydro[{float(pHy[m].min()):.3g},{float(pHy[m].max()):.3g}]  '
              f'A p_hydro[{float(ApHy[m].min()):+.3g},{float(ApHy[m].max()):+.3g}]  '
              f's[{float(s[m].min()):+.3g},{float(s[m].max()):+.3g}]  '
              f'|A p_hydro - s|/|s| = {float((ApHy - s).norm() / max(sNorm,1e-30)):.3f}')

        # Free-surface source pollution: how much of |s| is the positive
        # (expansion) part the pressure solve cannot fix?
        sPos = s.clamp(min=0.0)
        print(f'  s>0 fraction of |s|: {float(sPos[m].norm() / max(sNorm,1e-30)):.3f}   '
              f'(kernel-deficient rows, 1-V0/V > 0)')

        print('\n--- relaxed Jacobi (clamped p>=0, per-sample omega) ---')
        for iters in (256, 2000):
            x = torch.zeros_like(s)
            for _ in range(iters):
                x = (x + (omega * precond) * (s - A(x))).clamp(min=0.0)
                x = torch.where(solveRows, x, torch.zeros_like(x))
            line(f'jacobi {iters}it', x, iters)

        print('\n--- Krylov (unclamped) ---')
        x, st, cv = minresSolve(A, s, torch.zeros_like(s), rtol=args.rtol,
                                maxiter=args.maxiter, precond=precond, dim=1)
        line('MINRES', x.clamp(min=0.0), len(cv), st)
        x, st, cv = cgSolve(lambda p: -A(p), -s, torch.zeros_like(s), rtol=args.rtol,
                            maxiter=args.maxiter, precond=-precond, dim=1)
        line('CG (sign-flip)', x.clamp(min=0.0), len(cv), st)
        x, st, cv = bicgstabSolve(A, s, torch.zeros_like(s), rtol=args.rtol,
                                  maxiter=args.maxiter, precond=precond, dim=1)
        line('BiCGStab', x.clamp(min=0.0), len(cv), st)

    raise SystemExit(0)


bmod._solve = spy

if args.case == 'hydrostaticColumn':
    from warpSPH.cases import hydrostaticColumn
    caseObj, extra = hydrostaticColumn.hydrostaticColumnCase, {}
elif args.case == 'randomFlowBounded':
    from warpSPH.cases import randomFlowIncompressible
    caseObj = randomFlowIncompressible.randomFlowIncompressibleCase
    extra = {'params': {'bounded': True}}
else:
    from warpSPH.cases import dambreak
    caseObj, extra = dambreak.dambreakCase, {}

run(caseObj, nx=args.nx, nSteps=args.step + 1, scheme='band2018pb',
    quiet=True, plot=False, store=False, progress=False,
    integrationScheme='semiImplicitEuler', **extra)
print(f'\nno solve #{args.step} reached ({_n["i"]} solves ran)')
