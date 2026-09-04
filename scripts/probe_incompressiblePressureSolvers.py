"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 8, 2026-08-27): does swapping the
relaxed-Jacobi iteration for a Krylov method make the *constant-density*
(PS/shifting) solve actually converge in a running simulation?

`docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md` already characterised the operator and the
methods, but only as **single solves on a seeded state**: symmetric to fp32,
negative-semi-definite with a gauge mode, `kappa(M^-1 A) ~ 1.1e8`, and per-method
residuals at 200-1200 iterations (MINRES best at 9.7e-4 @200, CG strong,
BiCGStab stagnating then diverging, `krylovFp64` worth ~10x). What has never
been run is any of them **end-to-end inside a case**, which is what
`DFSPH_IMPROVEMENT_PLAN.md`'s "Next session" item 3 means by "its intra-solve
iterate has still never been instrumented".

The question this asks is specifically the one Part 8's results section left
open: the relaxed-Jacobi path runs its full 64 iterations every step forever
and never meets its tolerance. Is that the *solver* being slow, or the
*stopping criterion* being unreachable? A Krylov method on the same operator
and the same source separates them -- if MINRES also pegs at the cap, the
criterion is the problem; if it terminates, the iteration was.

**Gauge caveat, which this probe cannot avoid.** `solveIncompressible` sends
the Krylov path through `solvePressureKrylov(..., gauge='nonnegative')`, a
*post-hoc* clamp, while the relaxed-Jacobi path applies `ShiftPressureGauge`
per iteration (default `minShift`). So `--solvers relaxedJacobi minres` varies
solver *and* gauge together. `--gauge clamp` runs the Jacobi control under the
historical per-iteration clamp instead, which is the closer match; both
controls are reported by default for that reason.

Note also [I] (Ihmsen et al. 2014) Sec. 3.2's warning, which applies directly:
CG "clamping in between the iterations leads to invalid states", and they
"observed instabilities in case of any change in the final pressure field,
such as clamping of negative pressure values" -- i.e. even the post-hoc clamp
is not free. [BK] Sec. 5 names dropping the clamp as the enabler for CG. Hence
`--noClamp`, which passes `gauge=None`.

Usage:
  python scripts/probe_incompressiblePressureSolvers.py --nx 128 --nsteps 600 --cflFactor 0.1
  python scripts/probe_incompressiblePressureSolvers.py --solvers minres cg --fp64 --maxIters 200
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--nsteps', type=int, default=600)
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="Bender & Koschier's `dt <= 0.4 d/|v_max|`, in their "
                         "units -- since Part 12 `cflFactor` multiplies the "
                         "particle diameter. Numbers recorded before Part 12 "
                         "say 0.1, the same timestep under the old "
                         "support-radius convention. See Part 8 item 3.")
parser.add_argument('--solvers', nargs='*',
                    default=['relaxedJacobi', 'relaxedJacobi-clamp', 'minres', 'cg', 'bicgStab'],
                    help="'relaxedJacobi' uses the shipped ShiftPressureGauge "
                         "default; 'relaxedJacobi-clamp' forces the historical "
                         "per-iteration clamp, which is the gauge-matched "
                         "control for the Krylov rows.")
parser.add_argument('--maxIters', type=int, default=None,
                    help="override pressureSolver.maxIterations (default 64)")
parser.add_argument('--fp64', action='store_true',
                    help="set krylovFp64 (Krylov recurrence in float64, matvec "
                         "stays fp32); docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md measured "
                         "~10x better residual for BiCGStab/CG")
parser.add_argument('--noClamp', action='store_true',
                    help="pass gauge=None to the Krylov path instead of "
                         "'nonnegative' ([BK] Sec. 5's precondition for CG)")
parser.add_argument('--case', default='kolmogorovIncompressible')
parser.add_argument('--extra', nargs='*', default=[])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math
import statistics
import torch

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import ShiftPressureGauge
from warpSPH.configurations.moduleConfigurations.solver import PressureSolverType

import warpSPH.systems.incompressible as sysmod
import warpSPH.modules.incompressible.incompressible as incmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

_realSolve = sysmod.solveIncompressible
_realKrylov = incmod.solvePressureKrylov

results = {}

for name in args.solvers:
    solverName = 'relaxedJacobi' if name.startswith('relaxedJacobi') else name
    forceClamp = name.endswith('-clamp')

    log = []

    def _instrumented(particles, config, schemeConfig, adjacency, dvdt, dt,
                      verbose=False, _log=log):
        a_p, pressure, errors, pressures = _realSolve(
            particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
        fluid = particles.kinds == 0
        _log.append(dict(
            nIter=len(errors),
            finalErr=float(errors[-1]) if errors else float('nan'),
            pMean=float(pressure[fluid].mean().item()),
            pMax=float(pressure[fluid].max().item()),
            finite=bool(torch.isfinite(pressure).all()),
        ))
        return a_p, pressure, errors, pressures

    if args.noClamp:
        def _krylovNoClamp(*a, gauge='center', **kw):
            return _realKrylov(*a, gauge=None, **kw)
        incmod.solvePressureKrylov = _krylovNoClamp

    _origConfigure = case.configureScheme

    def _configure(ctx, _orig=_origConfigure, _solver=solverName, _clamp=forceClamp):
        _orig(ctx)
        ps = ctx.schemeConfig.solverConfig.pressureSolver
        ps.solverType = PressureSolverType[_solver]
        if args.maxIters is not None:
            ps.maxIterations = args.maxIters
        if args.fp64:
            ps.krylovFp64 = True
        if _clamp:
            ctx.schemeConfig.solverConfig.shiftPressureGauge = ShiftPressureGauge.nonNegativeClamp

    case.configureScheme = _configure
    sysmod.solveIncompressible = _instrumented
    try:
        argv = ['--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
                '--cflFactor', str(args.cflFactor), '--quiet', '--no-store', '--no-plot']
        r = caseMain(case, argv=argv + args.extra)
    except Exception as exc:  # a solver breakdown should not lose the other rows
        r = None
        print(f"  [{name}] raised: {type(exc).__name__}: {exc}")
    finally:
        case.configureScheme = _origConfigure
        sysmod.solveIncompressible = _realSolve
        incmod.solvePressureKrylov = _realKrylov

    results[name] = (r, log)

cap = args.maxIters if args.maxIters is not None else 64
print(f"\n=== {args.case} nx={args.nx} nSteps={args.nsteps} cflFactor={args.cflFactor} "
      f"maxIters={cap} fp64={args.fp64} noClamp={args.noClamp} ===")
print(f"{'solver':>22} {'steps':>6} {'finite':>7} {'nIter mn':>9} {'@cap%':>6} "
      f"{'finalErr':>10} {'|pMean|':>10} {'rhoErr':>10} {'rhoMax':>10} {'wall s':>8}")

for name, (r, log) in results.items():
    if r is None or not log:
        print(f"{name:>22} {'--':>6} {'FAILED':>7}")
        continue
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    finite = [e for e in log if e['finite']]
    half = log[len(log) // 2:] or log
    atCap = 100.0 * sum(1 for e in half if e['nIter'] >= cap) / max(1, len(half))
    band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0))
            for row in tr if 'maxDensity' in row]
    tail = band[len(band) // 2:] or band
    print(f"{name:>22} {len(tr):6d} {str(len(finite) == len(log)):>7} "
          f"{statistics.mean(e['nIter'] for e in half):9.1f} {atCap:6.0f} "
          f"{statistics.median(e['finalErr'] for e in half):10.3e} "
          f"{abs(statistics.mean(e['pMean'] for e in half)):10.3e} "
          f"{(statistics.mean(tail) if tail else float('nan')):10.3e} "
          f"{(max(band) if band else float('nan')):10.3e} {r.wallTime:8.1f}")
    if len(finite) < len(log):
        print(f"{'':>22} (non-finite pressure from solve {len(finite)} of {len(log)})")
