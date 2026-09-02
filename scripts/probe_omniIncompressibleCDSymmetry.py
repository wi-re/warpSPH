"""Is `omniIncompressible`'s constant-density operator MINRES-solvable if the
near-wall block is made *symmetric*?  (DFSPH_IMPROVEMENT_PLAN.md active track,
"Next, in order" item 1 -- the `band2018pb` thread, cheaper interim probe.)

Part 43 ruled out the diagonal patch (`CD_TIKHONOV` only restores the Jacobi's
iteration budget) and the naive *non-symmetric* Krylov solve (BiCGStab / GMRES
break down along the near-null space).  The plan's next step is:

    symmetrise `A` ... then MINRES.  Grade: iteration count to a real |r|_2
    tolerance (not the floored omniSPH metric), and whether the run improves.

This probe captures the healthy density-mode system `A p = s` at a chosen
density-solve index (a `_solve` spy, everything done *inside*
`applyConsistentCoupling` so the boundary rows carry the paper's rho0/Akinci
state the real solve uses) and, on that one representative system:

  1. measures the SYMMETRY DEFECT  |<Ax,y> - <x,Ay>| / (||Ax|| ||y||)  of
       * A_wall   -- the current composed operator (per-iterate `wallPressure-
                     Extrapolation` Robin closure -- the non-symmetric part)
       * A_plain  -- boundary p == 0 (Bender-Westhofen-Jeske 2023 Eq. 33)
       * A_krylov -- `modules/incompressible/krylov.buildIISPHMatvec`, the
                     symmetric `shift o accel` operator the live `divergenceFree`
                     Krylov path uses, `staticBoundary` terms, dt_scale = dt**2
     and the wall-Robin perturbation ||(A_wall - A_plain) x|| / ||A_plain x||.

  2. solves that system every way -- relaxed Jacobi (the run's real behaviour),
     MINRES, CG (sign-flipped), BiCGStab -- on A_plain / A_krylov / A_wall and
     on A_plain + a uniform Tikhonov shift, and reports for each the iteration
     count and the TRUE relative residual ||s - A x||_2 / ||s||_2 and ||x||.

Closed box (`randomFlowBounded`): the pure-Neumann compatibility projection
(`CD_SOURCE_PROJECT`) is applied to `s`, and the mean-zero gauge replaces the
`p >= 0` clamp, matching the scheme.

    python scripts/probe_omniIncompressibleCDSymmetry.py                         # hydrostaticColumn nx=64 solve#30
    python scripts/probe_omniIncompressibleCDSymmetry.py --nx 128 --step 30
    python scripts/probe_omniIncompressibleCDSymmetry.py --case randomFlowBounded --step 5
    python scripts/probe_omniIncompressibleCDSymmetry.py --case dambreak --step 30
"""
import argparse

import torch

argp = argparse.ArgumentParser()
argp.add_argument('--case', default='hydrostaticColumn',
                  choices=['hydrostaticColumn', 'dambreak', 'randomFlowBounded'])
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--step', type=int, default=30,
                  help='density-solve index to capture (1-based)')
argp.add_argument('--tik', type=float, default=0.1,
                  help='CD_TIKHONOV value for the shifted variant')
argp.add_argument('--maxiter', type=int, default=400)
argp.add_argument('--rtol', type=float, default=1e-8)
args = argp.parse_args()

import warpSPH.schemes.omniIncompressible as omod
from warpSPH.runner import run
from warpSPH.configurations import BoundaryPressureMode, BoundaryOperatorTerms
from warpSPH.modules.incompressible.consistent import applyConsistentCoupling
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.incompressible.wallPressure import wallPressureExtrapolation
from warpSPH.modules.incompressible.krylov import buildIISPHMatvec
from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.cg import cgSolve
from warpSPH.modules.shifting.minres import minresSolve

_orig = omod._solve
_n = {'i': 0}
_done = {'v': False}


def _randfield(fluid, seed):
    g = torch.Generator(device=fluid.device).manual_seed(seed)
    x = torch.zeros(fluid.shape, dtype=torch.float32, device=fluid.device)
    x[fluid] = torch.randn(int(fluid.sum()), generator=g, device=fluid.device)
    x[fluid] -= x[fluid].mean()      # off the constant gauge null space
    return x


def spy(state, config, schemeConfig, adjacency, *, fluid, rho0, vEnter,
        warmStart, dt, mode, minIters, maxIters, tol):
    if mode == 'density':
        _n['i'] += 1
    if not (mode == 'density' and _n['i'] == args.step and not _done['v']):
        return _orig(state, config, schemeConfig, adjacency, fluid=fluid,
                     rho0=rho0, vEnter=vEnter, warmStart=warmStart, dt=dt,
                     mode=mode, minIters=minIters, maxIters=maxIters, tol=tol)
    _done['v'] = True

    with applyConsistentCoupling(state, config, schemeConfig, adjacency,
                                 BoundaryPressureMode.consistent):
        boundary = state.kinds == 1
        apparent = state.masses / state.densities
        alpha = dt * dt * computeAlpha(
            state, config, schemeConfig, adjacency,
            apparentVolumes=apparent, includeBoundaryReaction=False)
        alphaSafe = torch.clamp(alpha, max=-1e-6)
        precond = torch.where(fluid, 1.0 / alphaSafe, torch.zeros_like(alpha))
        medAlpha = float(alpha[fluid].abs().median())

        divEnter = omod._divergence(state, config, adjacency, vEnter)
        source = (1.0 - state.densities / rho0) + dt * divEnter
        source = torch.where(fluid, source, torch.zeros_like(source))
        bm = source[fluid].mean()
        sn = source[fluid].norm().clamp_min(1e-30)
        fracUniform = 1.0 - float((source[fluid] - bm).norm() / sn)
        project = fracUniform > omod.CD_PROJECT_THRESHOLD
        if project:
            source = source - torch.where(fluid, bm.expand_as(source),
                                          torch.zeros_like(source))
        s = source
        sNorm = float(s.norm())

        def accel(pt, wallP):
            pin = wallPressureExtrapolation(
                state, config, adjacency, pt, fluid, mode=wallP,
                clampNonNeg=False) if wallP else pt
            return omod._pressureAccel(state, config, adjacency, pin, fluid)

        def makeA(wallP, shift=0.0):
            def A(pt):
                pt = torch.where(fluid, pt, torch.zeros_like(pt))
                aP = -dt * dt * omod._divergence(state, config, adjacency,
                                                 accel(pt, wallP))
                if shift:
                    aP = aP - shift * pt
                return torch.where(fluid, aP, torch.zeros_like(aP))
            return A

        A_wall = makeA(omod.WALL_PRESSURE_MODE)
        A_plain = makeA(None)
        _mvK = buildIISPHMatvec(state, config, schemeConfig, adjacency,
                                dt * dt, boundaryTerms=BoundaryOperatorTerms.staticBoundary)

        def A_krylov(pt):
            pt = torch.where(fluid, pt, torch.zeros_like(pt))
            return torch.where(fluid, _mvK(pt), torch.zeros_like(pt))

        shift = args.tik * medAlpha
        A_shift = makeA(None, shift)

        # ---------------- report --------------------------------------------
        print(f'{args.case} nx={args.nx}  HEALTHY density-solve #{args.step}   '
              f'Nfluid={int(fluid.sum())}  Nbnd={int(boundary.sum())}')
        print(f'  |s|={sNorm:.4g}   mean(s)={float(bm):+.3g}   '
              f'frac_uniform={fracUniform:.4g}   project={project}   '
              f'max|1-rho|={float((1 - state.densities / rho0)[fluid].abs().max()):.3g}')
        print(f'  median|alpha_fluid|={medAlpha:.3g}   tikhonov shift={shift:.3g}'
              f'  (tik={args.tik:g})')

        print('\n--- symmetry defect  |<Ax,y> - <x,Ay>| / (||Ax|| ||y||) ---')
        pairs = [(_randfield(fluid, 10 + k), _randfield(fluid, 500 + k))
                 for k in range(5)]
        for tag, A in [('A_wall  (current, Robin)', A_wall),
                       ('A_plain (BWJ23 Eq.33)   ', A_plain),
                       ('A_krylov(shift o accel) ', A_krylov)]:
            ds, rel = [], []
            for x, y in pairs:
                Ax, Ay = A(x), A(y)
                num = abs(float((Ax * y).sum()) - float((x * Ay).sum()))
                ds.append(num / max(float(Ax.norm() * y.norm()), 1e-30))
                rel.append(num / max(abs(float((Ax * y).sum())), 1e-30))
            print(f'  {tag}   norm-rel {sum(ds) / len(ds):.3e}   '
                  f'inner-rel {sum(rel) / len(rel):.3e}')
        pert = [float((A_wall(x) - A_plain(x)).norm() / A_plain(x).norm().clamp_min(1e-30))
                for x, _ in pairs]
        print(f'  wall-Robin perturbation  ||(A_wall - A_plain) x|| / ||A_plain x|| '
              f'= {sum(pert) / len(pert):.3e}')

        # ---------------- solvers ------------------------------------------
        def trueres(A, x):
            return float((s - A(x)).norm() / max(sNorm, 1e-30))

        def gauge(x):
            if project:
                return x - torch.where(fluid, x[fluid].mean().expand_as(x),
                                       torch.zeros_like(x))
            return x.clamp(min=0.0)

        def jacobi(A, iters=4000):
            x = torch.zeros_like(s)
            for _ in range(iters):
                x = gauge(x + (omod.OMEGA * precond) * (s - A(x)))
            return x, iters

        def run_minres(A):
            x, st, cv = minresSolve(A, s, torch.zeros_like(s), rtol=args.rtol,
                                    maxiter=args.maxiter, precond=precond, dim=1)
            return x, len(cv), st

        def run_cg(A):
            # negative-(semi)definite -> flip to SPD
            x, st, cv = cgSolve(lambda p: -A(p), -s, torch.zeros_like(s),
                                rtol=args.rtol, maxiter=args.maxiter,
                                precond=-precond, dim=1)
            return x, len(cv), st

        def run_bicgstab(A):
            x, st, cv = bicgstabSolve(A, s, torch.zeros_like(s), rtol=args.rtol,
                                      maxiter=args.maxiter, precond=precond, dim=1)
            return x, len(cv), st

        def line(tag, A, x, nit, st=None):
            sfx = '' if st is None else f'  status={st}'
            print(f'  {tag:34s} it={nit:4d}  |r|/|s|={trueres(A, x):.3e}  '
                  f'|x|max={float(x.abs().max()):.4g}  |x|2={float(x.norm()):.4g}{sfx}')

        print('\n--- relaxed Jacobi (omega=0.3, run\'s real path: '
              f'{"mean-zero" if project else "p>=0 clamp"}) ---')
        for tag, A in [('A_wall  2000it', A_wall), ('A_plain 2000it', A_plain),
                       ('A_shift 2000it', A_shift)]:
            x, nit = jacobi(A, 2000)
            line(tag, A, x, nit)

        print('\n--- MINRES (symmetric; needs A ~ A^T) ---')
        for tag, A in [('A_plain ', A_plain), ('A_krylov', A_krylov),
                       ('A_shift ', A_shift), ('A_wall  ', A_wall)]:
            x, nit, st = run_minres(A)
            line(f'MINRES {tag}', A, gauge(x) if not project else x, nit, st)

        print('\n--- CG (sign-flipped to SPD) ---')
        for tag, A in [('A_plain ', A_plain), ('A_krylov', A_krylov),
                       ('A_shift ', A_shift)]:
            x, nit, st = run_cg(A)
            line(f'CG {tag}', A, gauge(x) if not project else x, nit, st)

        print('\n--- BiCGStab (non-symmetric; Part 43 baseline) ---')
        for tag, A in [('A_plain ', A_plain), ('A_shift ', A_shift),
                       ('A_wall  ', A_wall)]:
            x, nit, st = run_bicgstab(A)
            line(f'BiCGStab {tag}', A, gauge(x) if not project else x, nit, st)

    raise SystemExit(0)


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

print(f'\nno density solve #{args.step} was reached '
      f'({_n["i"]} density solves ran)')
