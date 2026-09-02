"""Offline analysis of `omniIncompressible`'s density-mode linear system
`A p = s` (DFSPH_IMPROVEMENT_PLAN.md Part 43).

`probe_omniIncompressibleCDSolver.py` runs the scheme end-to-end under each
`CD_SOLVER`; this one captures the constant-density operator + RHS from a
*healthy* run (`CD_SOLVER='jacobi'`, full iteration budget) at a chosen
density-solve index, then runs offline solvers / regularisations on that
representative system to see WHY the free-surface Jacobi stalls:

  * BiCGStab / GMRES raw          -- do they converge, and to what `|x|`?
  * relaxed Jacobi (clamped / unclamped) -- the omniSPH iteration itself
  * uniform Tikhonov + BiCGStab   -- `(A - eps*med|alpha|) p = s`

Part 43 result: the free-surface operator is near-singular -- Krylov drives
`|r|` down but `|x|` blows up (`~1e9` on `hydrostaticColumn` step 1); the
clamped Jacobi is bounded but leaves `|r|/|b| ~ 0.94`; a uniform Tikhonov
shift (the landed `CD_TIKHONOV`) bounds `|x|` and lets the Jacobi contract.

    python scripts/probe_omniIncompressibleCDSystem.py                       # hydrostaticColumn nx=64 solve#30
    python scripts/probe_omniIncompressibleCDSystem.py --case dambreak --nx 64 --step 30
    python scripts/probe_omniIncompressibleCDSystem.py --case randomFlowBounded --step 5
"""
import argparse

import torch

argp = argparse.ArgumentParser()
argp.add_argument('--case', default='hydrostaticColumn',
                  choices=['hydrostaticColumn', 'dambreak', 'randomFlowBounded'])
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--step', type=int, default=30,
                  help='density-solve index to capture (1-based)')
args = argp.parse_args()

import warpSPH.schemes.omniIncompressible as omod
import warpSPH.modules.shifting.bicgstab as bmod
import warpSPH.modules.shifting.gmres as gmod
from warpSPH.runner import run
from warpSPH.configurations import BoundaryPressureMode
from warpSPH.modules.incompressible.consistent import applyConsistentCoupling
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.incompressible.wallPressure import wallPressureExtrapolation

_orig = omod._solve
_grab = {}
_n = {'i': 0}


def spy(state, config, schemeConfig, adjacency, *, fluid, rho0, vEnter,
        warmStart, dt, mode, minIters, maxIters, tol):
    if mode == 'density':
        _n['i'] += 1
    if mode == 'density' and _n['i'] == args.step and not _grab:
        with applyConsistentCoupling(state, config, schemeConfig, adjacency,
                                     BoundaryPressureMode.consistent):
            apparent = state.masses / state.densities
            alpha = dt * dt * computeAlpha(
                state, config, schemeConfig, adjacency,
                apparentVolumes=apparent, includeBoundaryReaction=False)
            divEnter = omod._divergence(state, config, adjacency, vEnter)
            source = (1.0 - state.densities / rho0) + dt * divEnter
            source = torch.where(fluid, source, torch.zeros_like(source))
            alphaSafe = torch.clamp(alpha, max=-1e-6)
            precond = torch.where(fluid, 1.0 / alphaSafe, torch.zeros_like(alpha))

            def accel(pt):
                pin = wallPressureExtrapolation(
                    state, config, adjacency, pt, fluid,
                    mode=omod.WALL_PRESSURE_MODE, clampNonNeg=False)
                return omod._pressureAccel(state, config, adjacency, pin, fluid)

            def matvec(pt):
                pt = torch.where(fluid, pt, torch.zeros_like(pt))
                aP = -dt * dt * omod._divergence(state, config, adjacency, accel(pt))
                return torch.where(fluid, aP, torch.zeros_like(aP))

            _grab.update(matvec=matvec, b=source.detach().clone(),
                         precond=precond.detach().clone(),
                         fluid=fluid.detach().clone())
    return _orig(state, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
                 vEnter=vEnter, warmStart=warmStart, dt=dt, mode=mode,
                 minIters=minIters, maxIters=maxIters, tol=tol)


omod._solve = spy
omod.CD_SOLVER = 'jacobi'

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

run(caseObj, nx=args.nx, nSteps=args.step + 1, scheme='omniIncompressible',
    quiet=True, plot=False, store=False, progress=False,
    integrationScheme='semiImplicitEuler', **extra)

if not _grab:
    raise SystemExit(f'no density solve #{args.step} was reached')

A = _grab['matvec']
b = _grab['b']
precond = _grab['precond']
fluid = _grab['fluid']
r0 = b.norm()
bm = b[fluid].mean()
print(f'{args.case} nx={args.nx} HEALTHY density-solve #{args.step}   '
      f'Nfluid={int(fluid.sum())}   |b|={r0:.4g}   mean(b)={bm:.3g}   '
      f'|b-mean|/|b|={((b[fluid] - bm).norm() / r0):.4g}   '
      f'max|1-rho|~{b[fluid].abs().max():.3g}')


def report(tag, x, mv=A):
    rr = (b - mv(x)).norm() / r0
    print(f'  {tag:32s} |r|/|b|={rr:.2e}  |x|max={x.abs().max():.4g}  '
          f'|x|2={x.norm():.4g}')


for name, fn, kw in [
        ('bicgstab', bmod.bicgstabSolve,
         dict(rtol=1e-6, maxiter=400, precond=precond, dim=1)),
        ('gmres(50)', gmod.gmresSolve,
         dict(rtol=1e-6, maxiter=400, precond=precond, restart=50, dim=1))]:
    x, st, cv = fn(A, b, torch.zeros_like(b), **kw)
    report(f'{name} raw st={st} it={len(cv)}', x)

for clamp in (False, True):
    x = torch.zeros_like(b)
    for _ in range(2000):
        x = x + (0.3 * precond) * (b - A(x))
        if clamp:
            x = x.clamp(min=0.0)
    report(f'jacobi 2000it clamp={clamp}', x)

medAlpha = float((1.0 / precond[fluid].abs()).median())
for eps in (1e-2, 3e-2, 1e-1):
    def Ash(x, e=eps):
        return A(x) - e * medAlpha * x
    x, st, cv = bmod.bicgstabSolve(Ash, b, torch.zeros_like(b), rtol=1e-6,
                                   maxiter=400, precond=precond, dim=1)
    report(f'bicgstab tikhonov eps={eps:g} st={st} it={len(cv)}', x, mv=Ash)
