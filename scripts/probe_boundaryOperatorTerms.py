"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 9, 2026-08-28): the published
solvers do not let a *static* boundary particle contribute every term of the
pressure operator. This codebase does. Measure what that is worth.

Two claims, one physical statement (`BoundaryOperatorTerms`):

  - `computeAlpha`'s **second sum** (`sum_j V_j^2/m_j |gradW_ij|^2`) is
    `dp_i/dx_j`: how much `rho_i` changes because neighbour `j` accelerated
    under `i`'s pressure. [BK] 3.2: "since `F^p_{j<-i} = 0` if particle j is
    not dynamic, the equation for `kappa^v_i` must be adapted accordingly for
    static boundary particles." SPlisHSPlasH's `TimeStepDFSPH::
    computeDFSPHFactor` implements it literally -- its boundary loop
    accumulates into `grad_p_i` (the first sum) and never into
    `sum_grad_p_k` (the second).
  - the **divergence's `a_j` term**: `dx_p_i = sum_j V_j (a_i - a_j).gradW_ij`
    counts the neighbour's own pressure displacement, which a static particle
    does not have. SPlisHSPlasH's `TimeStepIISPH::pressureSolveIteration`
    boundary loop keeps only `i`'s displacement (`sum += V_j * dij_pj_i.gradW`).

`schemes/divergenceFree.py` zeroes `dxdt`/`dvdt` for every `kind != 0` row, so both
apply here. Two modes:

  --mode diag  (cheap, one warmed-up state)
      Per wall-depth bin: `alpha` with and without the boundary reaction, and
      the *true* diagonal of each of the two operators, extracted exactly by
      applying the operator to unit vectors. Answers (a) how much of the
      diagonal is the disputed term, and (b) which alpha/operator pairing is
      actually self-consistent -- a diagonal that does not match its operator
      is a silently rescaled relaxation factor (see `JacobiRelaxationMode`).

  --mode ab    (expensive, one full run per mode)
      End-to-end A/B of `BoundaryOperatorTerms` on the same case, reported
      like `probe_shiftPressureGauge.py`.

Usage:
  python scripts/probe_boundaryOperatorTerms.py --mode diag
  python scripts/probe_boundaryOperatorTerms.py --mode ab --nsteps 900
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='diag', choices=['diag', 'ab', 'spectrum', 'dfTrace'])
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'],
                    help="extra argv forwarded to the case")
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="[BK]'s published constant, in their units -- since "
                         "Part 12 `cflFactor` multiplies the particle diameter. "
                         "Numbers recorded before Part 12 say `cflFactor=0.1`, "
                         "which was the same timestep under the old "
                         "support-radius convention")
# --mode diag
parser.add_argument('--warmup', type=int, default=120,
                    help="steps before probing, so the configuration is developed")
parser.add_argument('--nprobe', type=int, default=12,
                    help="rows probed per depth bin; 2 matvecs each")
parser.add_argument('--power', type=int, default=400,
                    help="--mode spectrum: power iterations")
# --mode ab
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--dfMaxIters', type=int, default=0,
                    help="--mode ab: override the divergence-free solver's iteration "
                         "cap (0 = leave the case default, 32). Tests whether a "
                         "slower-contracting operator is merely under-iterated")
parser.add_argument('--tLimit', type=float, default=1000.0)
parser.add_argument('--modes', nargs='*',
                    default=['full', 'staticBoundary', 'diagonalOnly', 'operatorOnly'])
parser.add_argument('--solvers', default='both',
                    choices=['both', 'divergenceFree', 'incompressible'],
                    help="which of the step's two solves the mode applies to; the "
                         "other one is forced back to `full` for the duration of its "
                         "call. `both` is what the config setting does on its own")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

import torch

from warpSPH.runner.cli import caseMain
from warpSPHCore import SupportScheme
from warpSPH.configurations import BoundaryOperatorTerms
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.cases.weaklyCompressible import domainBoundarySdf

import warpSPH.systems.incompressible as sysmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

BINS = [(-1e9, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]


def binLabel(lo, hi):
    if lo < -1e8:
        return "(<0 inside)"
    return f"[{lo:g},inf)" if hi > 1e8 else f"[{lo:g},{hi:g})"


def captureState():
    """Run the case for `--warmup` steps and grab the exact state, adjacency and
    `dt` the last solve ran on."""
    cap = {}
    _real = sysmod.solveIncompressible

    def _capture(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
        cap.update(particles=particles, config=config, schemeConfig=schemeConfig,
                   adjacency=adjacency, dt=dt)
        return _real(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)

    sysmod.solveIncompressible = _capture
    try:
        r = caseMain(case, argv=['--nx', str(args.nx), '--nSteps', str(args.warmup),
                                 '--tLimit', '1000.0', '--cflFactor', str(args.cflFactor),
                                 '--quiet', '--no-store', '--no-plot'] + args.extra)
    finally:
        sysmod.solveIncompressible = _real
    return cap, r


def runDiag():
    cap, r = captureState()
    particles, config = cap['particles'], cap['config']
    schemeConfig, adjacency, dt = cap['schemeConfig'], cap['adjacency'], cap['dt']

    apparentArea = particles.masses / particles.densities
    alphaFull = dt ** 2 * computeAlpha(
        currentState=particles, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, apparentVolumes=apparentArea, includeBoundaryReaction=True)
    alphaStatic = dt ** 2 * computeAlpha(
        currentState=particles, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, apparentVolumes=apparentArea, includeBoundaryReaction=False)

    fluidMask = (particles.kinds == 0).unsqueeze(-1)

    def applyOperator(p, boundaryMoves):
        a = computePressureAccelIISPH(state=particles, pressureValues=p, config=config,
                                      supportScheme=SupportScheme.Scatter, adjacency=adjacency)
        if not boundaryMoves:
            a = torch.where(fluidMask, a, torch.zeros_like(a))
        return dt ** 2 * computePressureShiftIISPH(
            state=particles, config=config, pressureAccels=a,
            supportScheme=SupportScheme.Scatter, adjacency=adjacency)

    ctx = r.ctx
    dxs = ctx.config.dx
    sdf = domainBoundarySdf(ctx)
    d, _ = sdf(particles.positions.detach().clone().requires_grad_(True))
    depth = d.detach() / dxs
    fluid = particles.kinds == 0

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"after {args.warmup} steps, cflFactor={args.cflFactor} ===")
    print(f"dt={dt:.6g}  h={float(particles.supports.median()) / dxs:.2f} spacings  "
          f"{int(fluid.sum())} fluid / {int((~fluid).sum())} static particles")

    print("\n(1) how much of the IISPH diagonal is the disputed reaction term")
    print(f"{'depth':>12} {'n':>7} {'alpha(full)':>13} {'alpha(static)':>14} "
          f"{'static/full':>12}")
    for lo, hi in BINS:
        m = fluid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0:
            continue
        af, as_ = float(alphaFull[m].mean()), float(alphaStatic[m].mean())
        print(f"{binLabel(lo, hi):>12} {n:7d} {af:13.5e} {as_:14.5e} {as_ / af:12.4f}")

    print("\n(2) each operator's TRUE diagonal against each alpha "
          "(one unit-vector matvec per row, per operator)")
    print(f"{'depth':>12} {'n':>5} {'dFull/aFull':>12} {'dStat/aStat':>12} "
          f"{'dStat/aFull':>12} {'dFull/aStat':>12}")
    g = torch.Generator(device=depth.device)
    g.manual_seed(0)
    for lo, hi in BINS:
        m = fluid & (depth >= lo) & (depth < hi)
        idxs = torch.nonzero(m).flatten()
        if len(idxs) == 0:
            continue
        pick = idxs[torch.linspace(0, len(idxs) - 1, min(args.nprobe, len(idxs))).long()]
        dF, dS, aF, aS = [], [], [], []
        for idx in pick.tolist():
            e = torch.zeros_like(particles.densities)
            e[idx] = 1.0
            dF.append(applyOperator(e, True)[idx].item())
            dS.append(applyOperator(e, False)[idx].item())
            aF.append(alphaFull[idx].item())
            aS.append(alphaStatic[idx].item())
        t = lambda v: torch.tensor(v, dtype=torch.float64)
        dF, dS, aF, aS = t(dF), t(dS), t(aF), t(aS)
        print(f"{binLabel(lo, hi):>12} {len(pick):5d} {float((dF / aF).mean()):12.4f} "
              f"{float((dS / aS).mean()):12.4f} {float((dS / aF).mean()):12.4f} "
              f"{float((dF / aS).mean()):12.4f}")

    print("\nColumn 1 and 2 are the two self-consistent pairings (`full`, "
          "`staticBoundary`);\ncolumns 3 and 4 are the mismatched diagnostics "
          "(`diagonalOnly`, `operatorOnly`):\na ratio c there means the Jacobi "
          "step is really running at omega/c.")


def scopeToOneSolver():
    """Force `boundaryOperatorTerms` back to `full` around the *other* solver's
    call, so a run isolates one of the step's two solves. Both solvers read the
    setting off `schemeConfig` when they are called, so wrapping the call is
    enough -- no second config field needed."""
    import warpSPH.schemes.divergenceFree as dfsphmod

    if args.solvers == 'divergenceFree':
        target, holder, attr = sysmod.solveIncompressible, sysmod, 'solveIncompressible'
    else:
        target, holder, attr = dfsphmod.solveDivergenceFree, dfsphmod, 'solveDivergenceFree'

    def forcedFull(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
        saved = schemeConfig.solverConfig.boundaryOperatorTerms
        schemeConfig.solverConfig.boundaryOperatorTerms = BoundaryOperatorTerms.full
        try:
            return target(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
        finally:
            schemeConfig.solverConfig.boundaryOperatorTerms = saved

    setattr(holder, attr, forcedFull)
    return lambda: setattr(holder, attr, target)


def runSpectrum():
    """Does dropping the boundary reaction push the relaxed-Jacobi iteration
    out of its stability window?

    Both solvers iterate `p <- p + omega * D^-1 r`, which converges iff
    `omega < 2 / rho(D^-1 A)` (`D^-1 A` is similar to a symmetric PSD matrix --
    see `JacobiRelaxationMode`'s docstring). `rho` is scale-free: `A` and `D`
    both carry the `dt` factor, so the *same* number governs both solvers, and
    both ship `relaxationFactor = 0.3`. Power iteration on the fluid subproblem
    gives it directly.
    """
    cap, r = captureState()
    particles, config = cap['particles'], cap['config']
    schemeConfig, adjacency, dt = cap['schemeConfig'], cap['adjacency'], cap['dt']

    fluid = particles.kinds == 0
    fluidMask = fluid.unsqueeze(-1)
    apparentArea = particles.masses / particles.densities

    ctx = r.ctx
    sdf = domainBoundarySdf(ctx)
    d, _ = sdf(particles.positions.detach().clone().requires_grad_(True))
    depth = d.detach() / ctx.config.dx

    omega = schemeConfig.solverConfig.divergenceFreeSolver.relaxationFactor
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} after "
          f"{args.warmup} steps, cflFactor={args.cflFactor} ===")
    print(f"omega = {omega} (both solvers), {args.power} power iterations\n")
    print(f"{'operator':>16} {'rho(D^-1 A)':>12} {'window 2/rho':>13} {'omega/window':>13} "
          f"{'eigvec at wall':>15}")

    for label, boundaryMoves in (('full', True), ('staticBoundary', False)):
        alphas = dt ** 2 * computeAlpha(
            currentState=particles, config=config, schemeConfig=schemeConfig,
            adjacency=adjacency, apparentVolumes=apparentArea,
            includeBoundaryReaction=boundaryMoves)
        alphas = torch.clamp(alphas, max=-1e-6)

        def precondOp(p):
            a = computePressureAccelIISPH(state=particles, pressureValues=p, config=config,
                                          supportScheme=SupportScheme.Scatter,
                                          adjacency=adjacency)
            if not boundaryMoves:
                a = torch.where(fluidMask, a, torch.zeros_like(a))
            out = dt ** 2 * computePressureShiftIISPH(
                state=particles, config=config, pressureAccels=a,
                supportScheme=SupportScheme.Scatter, adjacency=adjacency)
            return torch.where(fluid, out / alphas, torch.zeros_like(out))

        g = torch.Generator(device=particles.densities.device)
        g.manual_seed(0)
        x = torch.rand(particles.densities.shape, generator=g,
                       device=particles.densities.device, dtype=particles.densities.dtype)
        x = torch.where(fluid, x - x[fluid].mean(), torch.zeros_like(x))
        x = x / x.norm()
        rho = float('nan')
        for _ in range(args.power):
            y = precondOp(x)
            n = float(y.norm())
            if n == 0.0:
                break
            rho, x = n, y / n

        w = x.abs() ** 2
        atWall = float(w[fluid & (depth < 2)].sum() / w[fluid].sum())
        frac = float((fluid & (depth < 2)).sum()) / float(fluid.sum())
        print(f"{label:>16} {rho:12.4f} {2.0 / rho:13.4f} {omega / (2.0 / rho):13.4f} "
              f"{atWall:14.1%} (rows: {frac:.1%})")

    print("\nomega/window > 1 means the fixed-omega Jacobi iteration is divergent on")
    print("that operator: each sweep amplifies the dominant mode instead of damping it.")
    print("'eigvec at wall' is the share of the dominant eigenvector's energy within 2")
    print("spacings of the wall, against the share of rows there -- a number far above")
    print("the row share means the unstable mode is a wall mode.")


def runDFTrace():
    """Where does `staticBoundary` in the *divergence-free* solve go wrong:
    inside a solve, or across steps?

    Two candidates, and they are distinguishable. If the Jacobi iteration is
    unstable on the new operator, the residual grows *within* a single solve
    (last error > first error). If instead the solve converges fine each step
    but the correction it produces is too large at the wall, the per-solve
    residual stays healthy while the applied acceleration grows *across* steps.
    `--mode spectrum` already rules the first one out on paper; this checks it
    on the running case and measures the second.
    """
    import warpSPH.schemes.divergenceFree as dfsphmod

    for label, mode in (('full', BoundaryOperatorTerms.full),
                        ('staticBoundary (DF only)', BoundaryOperatorTerms.staticBoundary)):
        trace = []
        _origCfg = case.configureScheme
        _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible

        def _wrapped(ctx, _orig=_origCfg, mode=mode):
            _orig(ctx)
            ctx.schemeConfig.solverConfig.boundaryOperatorTerms = mode

        def traceDF(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
            out = _df(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
            aP, _p, errs, _ps_ = out
            fluid = particles.kinds == 0
            mag = aP.norm(dim=-1)
            trace.append((errs[0] if errs else float('nan'),
                          errs[-1] if errs else float('nan'),
                          float(mag[fluid].max()), float(mag[fluid].mean())))
            return out

        def forcedFullPS(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
            saved = schemeConfig.solverConfig.boundaryOperatorTerms
            schemeConfig.solverConfig.boundaryOperatorTerms = BoundaryOperatorTerms.full
            try:
                return _ps(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
            finally:
                schemeConfig.solverConfig.boundaryOperatorTerms = saved

        case.configureScheme = _wrapped
        dfsphmod.solveDivergenceFree = traceDF
        sysmod.solveIncompressible = forcedFullPS
        try:
            caseMain(case, argv=['--nx', str(args.nx), '--nSteps', str(args.warmup),
                                 '--tLimit', '1000.0', '--cflFactor', str(args.cflFactor),
                                 '--quiet', '--no-store', '--no-plot'] + args.extra)
        finally:
            case.configureScheme = _origCfg
            dfsphmod.solveDivergenceFree = _df
            sysmod.solveIncompressible = _ps

        print(f"\n=== divergence-free solve, {label} "
              f"({args.case} {' '.join(args.extra)} nx={args.nx}, {len(trace)} steps) ===")
        print(f"{'step':>6} {'err first':>11} {'err last':>11} {'last/first':>11} "
              f"{'max|a_p|':>11} {'mean|a_p|':>11}")
        step = max(1, len(trace) // 12)
        for i in range(0, len(trace), step):
            f0, f1, mx, mn = trace[i]
            print(f"{i:6d} {f0:11.4e} {f1:11.4e} {f1 / f0 if f0 else float('nan'):11.4f} "
                  f"{mx:11.4e} {mn:11.4e}")
        grew = sum(1 for f0, f1, _, _ in trace if f1 > f0)
        print(f"solves whose residual *grew* over its iterations: {grew}/{len(trace)}")


def runAB():
    results = {}
    restore = (lambda: None) if args.solvers == 'both' else scopeToOneSolver()
    for name in args.modes:
        mode = BoundaryOperatorTerms[name]
        _orig = case.configureScheme

        def _wrapped(ctx, _orig=_orig, mode=mode):
            _orig(ctx)
            ctx.schemeConfig.solverConfig.boundaryOperatorTerms = mode
            if args.dfMaxIters:
                ctx.schemeConfig.solverConfig.divergenceFreeSolver.maxIterations = args.dfMaxIters

        case.configureScheme = _wrapped
        try:
            results[name] = caseMain(case, argv=[
                '--nx', str(args.nx), '--nSteps', str(args.nsteps),
                '--tLimit', str(args.tLimit), '--cflFactor', str(args.cflFactor),
                '--quiet', '--no-store', '--no-plot',
            ] + args.extra)
        finally:
            case.configureScheme = _orig
    restore()

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"nSteps={args.nsteps} cflFactor={args.cflFactor} "
          f"solvers={args.solvers} ===")
    print(f"{'boundaryOperatorTerms':>22} {'steps':>7} {'diverged':>9} {'minRho':>9} "
          f"{'maxRho':>9} {'|rho-1| (2nd half)':>19} {'t_final':>9} {'wall s':>8}")
    for name, r in results.items():
        tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
        tail = tr[len(tr) // 2:] or tr
        band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0))
                for row in tail]
        print(f"{name:>22} {len(tr):7d} {str(r.diverged):>9} "
              f"{min(row['minDensity'] for row in tr):9.5f} "
              f"{max(row['maxDensity'] for row in tr):9.5f} "
              f"{sum(band) / max(1, len(band)):19.4e} "
              f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):9.4f} "
              f"{r.wallTime:8.1f}")
        if len(tr) < len(r.trajectory):
            print(f"{'':>22} (non-finite from step {len(tr)} of {len(r.trajectory)})")


if args.mode == 'diag':
    runDiag()
elif args.mode == 'spectrum':
    runSpectrum()
elif args.mode == 'dfTrace':
    runDFTrace()
else:
    runAB()
