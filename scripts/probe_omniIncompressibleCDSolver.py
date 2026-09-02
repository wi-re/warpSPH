"""Free-surface constant-density solve: relaxed Jacobi vs non-symmetric Krylov
(DFSPH_IMPROVEMENT_PLAN.md active track, "Next, in order" item 1 -- Part 43).

Part 42 established that `omniIncompressible`'s density-mode relaxed Jacobi
*stalls* on the free-surface cases (`hydrostaticColumn`, `dambreak`): on
`hydrostaticColumn` step 30, 2000 raw Jacobi iterations remove only ~20 % of
the residual (non-contractive, not merely slow). The closed box was fixed by
`CD_SOURCE_PROJECT`; MINRES breaks down on the wall-closed operator
(`status -13`). This probe drives the same linear system `A p = s` with a
*non-symmetric* Krylov method (`omniIncompressible.CD_SOLVER`) and grades:

  * per-step density-solve iteration count + final residual (a `_solve` spy);
  * whether the run holds, and its column FOMs (slope ratio -> 1.0 is the
    exact hydrostatic gradient, embeddedMinDensity, |v|max / KE).

Usage:
    python scripts/probe_omniIncompressibleCDSolver.py                       # hydrostaticColumn nx=64, 120 steps, all 3 solvers
    python scripts/probe_omniIncompressibleCDSolver.py --nx 128 --steps 200
    python scripts/probe_omniIncompressibleCDSolver.py --case dambreak --nx 64 --steps 150
    python scripts/probe_omniIncompressibleCDSolver.py --solvers jacobi,bicgstab
    python scripts/probe_omniIncompressibleCDSolver.py --maxiter 512 --rtol 1e-4
"""
import argparse
import statistics

argp = argparse.ArgumentParser()
argp.add_argument('--case', default='hydrostaticColumn',
                  choices=['hydrostaticColumn', 'dambreak', 'randomFlowBounded'])
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', type=int, default=120)
argp.add_argument('--solvers', default='jacobi,bicgstab,gmres')
argp.add_argument('--tik', default='0.0',
                  help='comma list of CD_TIKHONOV values to sweep (default 0.0)')
argp.add_argument('--maxiter', type=int, default=256)
argp.add_argument('--rtol', type=float, default=1e-3)
argp.add_argument('--restart', type=int, default=50)
args = argp.parse_args()

import warpSPH.schemes.omniIncompressible as omod
from warpSPH.runner import run

_orig_solve = omod._solve
_LOG = {'div': [], 'rho': []}


def _solve_spy(*a, **k):
    a_p, p, nit, err = _orig_solve(*a, **k)
    mode = k.get('mode')
    if mode is None and len(a) > 8:
        mode = a[8]
    _LOG['rho' if mode == 'density' else 'div'].append((nit, float(err)))
    return a_p, p, nit, err


omod._solve = _solve_spy

if args.case == 'hydrostaticColumn':
    from warpSPH.cases import hydrostaticColumn
    caseObj = hydrostaticColumn.hydrostaticColumnCase
    extra = {}
elif args.case == 'randomFlowBounded':
    from warpSPH.cases import randomFlowIncompressible
    caseObj = randomFlowIncompressible.randomFlowIncompressibleCase
    extra = {'params': {'bounded': True}}
else:
    from warpSPH.cases import dambreak
    caseObj = dambreak.dambreakCase
    extra = {}


def grade(rows):
    n = len(rows)
    lo = 3 * n // 4

    def col(key):
        return [r.get(key, float('nan')) for r in rows[lo:]
                if r.get(key, float('nan')) == r.get(key, float('nan'))]
    slope = col('pressureSlopeRatio')
    ke = col('kineticEnergy')
    vmax = col('maxVelocity')
    emb = col('embeddedMinDensity')
    mrho = col('maxDensity')
    out = {}
    if slope:
        out['slope'] = statistics.mean(slope)
    if ke:
        out['KE'] = (min(ke), max(ke))
    if vmax:
        out['vmax'] = max(vmax)
    if emb:
        out['embMin'] = min(emb)
    if mrho:
        out['maxRho'] = max(mrho)
    return out


print(f'{args.case}  nx={args.nx}  steps={args.steps}   '
      f'maxiter={args.maxiter} rtol={args.rtol:g} restart={args.restart}')
print('=' * 78)

combos = [(s.strip(), float(t)) for t in args.tik.split(',')
          for s in args.solvers.split(',')]
for solver, tik in combos:
    omod.CD_SOLVER = solver
    omod.CD_TIKHONOV = tik
    omod.CD_KRYLOV_MAXITER = args.maxiter
    omod.CD_KRYLOV_RTOL = args.rtol
    omod.CD_KRYLOV_RESTART = args.restart
    _LOG['div'].clear()
    _LOG['rho'].clear()

    r = run(caseObj, nx=args.nx, nSteps=args.steps, scheme='omniIncompressible',
            quiet=True, plot=False, store=False, progress=False,
            integrationScheme='semiImplicitEuler', **extra)

    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    rhoIt = [it for it, _ in _LOG['rho']]
    rhoErr = [e for _, e in _LOG['rho']]
    late = slice(3 * len(rhoIt) // 4, None)

    print(f'\n--- CD_SOLVER={solver!r}  CD_TIKHONOV={tik:g} ---   '
          f'ran {len(rows)} steps   diverged={r.diverged}')
    if rhoIt:
        print(f'  density-solve iters:  mean {statistics.mean(rhoIt):.1f}   '
              f'max {max(rhoIt)}   late-mean {statistics.mean(rhoIt[late]):.1f}'
              f'   (cap {args.maxiter})')
        print(f'  density-solve resid:  mean {statistics.mean(rhoErr):.3g}   '
              f'last {rhoErr[-1]:.3g}   late-max {max(rhoErr[late]):.3g}')
    g = grade(rows)
    print('  trajectory (last quarter): ' +
          '  '.join(f'{k}={v}' if not isinstance(v, float) else f'{k}={v:.4g}'
                    for k, v in g.items()))
