"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 5, 2026-08-27): the bounded
`randomFlowIncompressible` case reaches NaN on its own at nx=128, t=5.54,
under the untouched default. Part 2 only ever validated it to t~1.5, so this
is past the horizon anyone had looked at rather than a regression -- and it
is *not* the shifting-gauge mechanism Part 4 fixed, which by construction
declines to act on a wall-bounded solve.

This watches the blowup happen instead of only observing that it happened.
Every step it records the case's health (density extrema, peak speed, dt)
and, crucially, *where* the worst particle is: `wallDepth` is the same
`domainBoundarySdf` distance `probe_dfsphWallDensityProfile.py` bins by, so
"the error is wall-localized" vs "the error is in the bulk" is answerable
per step rather than only at the end. Both pressure solves are wrapped
(read-only) to log their iteration counts and pressure magnitudes, so a
runaway in either is visible before it reaches the density.

Reports the last `--tail` steps before divergence in full, since that is
where the mechanism lives.

Usage: `python scripts/probe_boundedIncompressibleBlowup.py [--nx 128]
[--tlimit 8.0] [--mode mdbcDensity] [--tail 25]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='randomFlowIncompressible',
                    choices=['randomFlowIncompressible', 'randomFlow'],
                    help="`randomFlow` is the weakly-compressible (deltaSPH) "
                         "sibling on the same bounded geometry -- the control "
                         "for 'is the wall penetration a DFSPH trait?'")
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--tlimit', type=float, default=8.0)
parser.add_argument('--maxSteps', type=int, default=2000)
parser.add_argument('--mode', type=str, default='mdbcDensity')
parser.add_argument('--tail', type=int, default=25)
parser.add_argument('--every', type=int, default=20)
parser.add_argument('--mlsBeforeSolve', action='store_true',
                    help="also run `computeMdbcPressure` *before* "
                         "`solveDivergenceFree`, not only after it. As shipped, "
                         "the MLS projection is one full step stale by the time "
                         "the solve reads it: it is computed from the previous "
                         "step's fluid pressure at the previous step's positions")
parser.add_argument('--mlsRelaxation', type=float, default=None,
                    help="override `mdbcPressureRelaxation` (default 0.3)")
parser.add_argument('--mdbcFinalize', action='store_true',
                    help="apply the mDBC density extrapolation in `finalize` "
                         "before the shifting solve. `divergenceFree_step` gives boundary "
                         "particles an extrapolated density for the "
                         "divergence-free solve, but `finalize` recomputes plain "
                         "summation densities and never re-applies mDBC, so the "
                         "shifting solve sees boundary rows whose density is "
                         "systematically low (truncated outward support)")
parser.add_argument('--dfMaxIters', type=int, default=None)
parser.add_argument('--dfRelaxMode', type=str, default=None, choices=['fixed', 'optimal'])
parser.add_argument('--psMaxIters', type=int, default=None)
parser.add_argument('--shiftApplication', type=str, default=None,
                    choices=['positionShift', 'positionAndVelocity', 'inStepVelocity'],
                    help="how `finalize` applies the constant-density solve; "
                         "`positionAndVelocity` is what makes this case stable "
                         "at the default CFL (see ShiftApplication's docstring)")
parser.add_argument('--wallFraction', type=float, default=None,
                    help="override `shiftVelocityWallFraction` (default 0.1); "
                         "smaller widens the band the correction acts over")
parser.add_argument('--fixedDt', type=float, default=None,
                    help="pin dt instead of using the case's adaptive timestep "
                         "hook -- required for any A/B whose variants change the "
                         "velocity field, since dt is CFL-derived from vMax and "
                         "would otherwise differ between the arms")
parser.add_argument('--cflFactor', type=float, default=None,
                    help="override the case's CFL factor -- DFSPH runs an ~80x "
                         "larger dt than the deltaSPH sibling on this geometry, "
                         "so this separates 'the wall treatment is wrong' from "
                         "'the wall treatment cannot cope with that dt'")
parser.add_argument('--snapshots', type=int, default=6,
                    help="how many steps back from the end the first per-particle "
                         "wall-depth profile is taken from")
parser.add_argument('--noPenShift', type=str, default=None, choices=[None, 'on', 'off'])
parser.add_argument('--shiftCap', type=float, default=None,
                    help="clip the implicit shift `dx = dt**2 * a_p` to this "
                         "many particle spacings per step (nothing bounds it "
                         "in the solver today)")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

import importlib
_caseMod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(_caseMod, f'{args.case}Case')
from warpSPH.cases.weaklyCompressible import domainBoundarySdf
from warpSPH.runner.runner import buildContext
from warpSPH.runner.caseSpec import CaseSpec

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod
from warpSPH.modules.mdbc import computeMdbcPressure as _mdbcPressure
from warpSPH.configurations import BoundaryPressureMode as _BPMode

solveLog = {}


def _wrap(module, name, tag):
    orig = getattr(module, name)

    def wrapped(*a, **kw):
        if tag == 'DF' and args.mlsBeforeSolve:
            particles = kw.get('particles', a[0] if a else None)
            config = kw.get('config')
            schemeConfig = kw.get('schemeConfig')
            adjacency = kw.get('adjacency')
            if (particles is not None and particles.pressures is not None
                    and schemeConfig.solverConfig.boundaryPressureMode
                    is _BPMode.mdbcMlsPressure):
                particles.pressures = _mdbcPressure(particles, config, schemeConfig, adjacency)
        out = orig(*a, **kw)
        a_p, pressure, errors, _pressures = out
        if tag == 'PS' and args.shiftCap is not None:
            dt = kw.get('dt', None)
            if dt is not None:
                limit = args.shiftCap * _DX[0] / (dt ** 2)
                norm = a_p.norm(dim=-1, keepdim=True)
                a_p = a_p * (limit / norm.clamp(min=1e-30)).clamp(max=1.0)
                out = (a_p,) + tuple(out[1:])
        stats = dict(
            nIter=len(errors),
            pMax=pressure.abs().max().item(),
            aMax=a_p.norm(dim=-1).max().item(),
        )
        if tag == 'PS':
            # `finalize` applies this solve's output as a *position* shift,
            # `dx = dt**2 * a_p`. A particle-shifting displacement is only
            # meaningful as a fraction of the particle spacing, and nothing in
            # this solver bounds it -- so measure it in those units, and note
            # where in the domain the largest one lands.
            particles = kw.get('particles', a[0] if a else None)
            dt = kw.get('dt', None)
            if particles is not None and dt is not None:
                shift = (dt ** 2) * a_p.norm(dim=-1)
                worst = int(shift.argmax())
                stats['shiftDx'] = shift.max().item() / _DX[0]
                stats['shiftDepth'] = _depthOf(particles, worst) / _DX[0]
        solveLog[tag] = stats
        return out

    setattr(module, name, wrapped)
    return orig


_DX = [1.0]  # filled in once `ctx` exists; the wrappers close over it


def _depthOf(particles, index):
    p = particles.positions.detach()[index:index + 1].clone().requires_grad_(True)
    d, _ = _SDF[0](p)
    return d.detach().item()


_SDF = [None]

if args.mdbcFinalize:
    from warpSPH.modules.mdbc import computeMdbcDensity as _mdbcDensity
    from warpSPH.configurations import BoundaryPressureMode as _BPM
    _origDensities = sysmod.computeDensities

    def _densitiesWithMdbc(state, config, schemeConfig, adjacency):
        rho = _origDensities(state, config, schemeConfig, adjacency)
        if schemeConfig.solverConfig.boundaryPressureMode != _BPM.plain:
            state.densities = rho
            rho = _mdbcDensity(state, config, schemeConfig, adjacency)
        return rho

    sysmod.computeDensities = _densitiesWithMdbc

_wrap(sysmod, 'solveIncompressible', 'PS')
for mod in (dfsphmod, sysmod):
    if hasattr(mod, 'solveDivergenceFree'):
        _wrap(mod, 'solveDivergenceFree', 'DF')

spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
spec = spec.merged(**case.defaults)
spec = spec.merged(nx=args.nx, tLimit=args.tlimit, store=False, plot=False, quiet=True,
                   progress=False,
                   params=dict({'bounded': True},
                               **({'boundaryPressureMode': args.mode}
                                  if 'boundaryPressureMode' in case.params else {})))

if args.cflFactor is not None:
    spec = spec.merged(cflFactor=args.cflFactor)

if args.fixedDt is not None:
    spec = spec.merged(dt=args.fixedDt, adaptiveDt=False)

ctx = buildContext(case, spec)
case.configureScheme(ctx)
if args.noPenShift is not None:
    ctx.schemeConfig.solverConfig.mdbcNoPenetrationShift = (args.noPenShift == 'on')
ps = ctx.schemeConfig.solverConfig.pressureSolver
df = ctx.schemeConfig.solverConfig.divergenceFreeSolver
if args.dfMaxIters is not None:
    df.maxIterations = args.dfMaxIters
if args.psMaxIters is not None:
    ps.maxIterations = args.psMaxIters
if args.mlsRelaxation is not None:
    ctx.schemeConfig.solverConfig.mdbcPressureRelaxation = args.mlsRelaxation
if args.shiftApplication is not None:
    from warpSPH.configurations import ShiftApplication
    ctx.schemeConfig.solverConfig.shiftApplication = ShiftApplication[args.shiftApplication]
if args.wallFraction is not None:
    ctx.schemeConfig.solverConfig.shiftVelocityWallFraction = args.wallFraction
if args.dfRelaxMode is not None:
    from warpSPH.configurations import JacobiRelaxationMode
    df.relaxationMode = JacobiRelaxationMode[args.dfRelaxMode]

system = case.buildSystem(ctx)
case.initialConditions(ctx, system)
runningState = system.initializeNewState()

if args.fixedDt is not None:
    # Set on the config directly: the case's own `targetDt` param wins over
    # `CaseSpec.dt`, so merging into the spec silently does nothing here.
    ctx.config.dt = args.fixedDt

sdf = domainBoundarySdf(ctx)
dx = ctx.config.dx
_SDF[0] = sdf
_DX[0] = dx


def depths(state):
    p = state.positions.detach().clone().requires_grad_(True)
    d, _ = sdf(p)
    return d.detach()


rows = []
# A ring buffer of the last few steps' per-particle state: the step that goes
# non-finite is preceded by one that is finite but already exploded (vMax ~1e5),
# so the informative profile is a couple of steps further back.
from collections import deque
snapshots = deque(maxlen=args.snapshots + 1)
for i in range(args.maxSteps):
    dtUsed = ctx.config.dt
    stepResult = ctx.integrator.function(
        state=runningState, f=ctx.stepFunction, dt=ctx.config.dt,
        config=ctx.config, verbose=False, schemeConfig=ctx.schemeConfig)
    runningState = stepResult.state
    if case.timestep is not None and args.fixedDt is None:
        ctx.config.dt = case.timestep(ctx, runningState)

    s = runningState.state
    fluid = s.kinds == 0
    rho = s.densities[fluid]
    v = s.velocities[fluid].norm(dim=-1)
    # Kinetic energy, because `vMax` is one particle and the question these
    # modes differ on is whether the *flow* is being damped (Part 18).
    ke = float(0.5 * (s.masses[fluid] * v ** 2).sum())
    d = depths(s)[fluid]
    err = (rho - 1.0).abs()
    finite = bool(torch.isfinite(rho).all() and torch.isfinite(v).all())
    row = dict(step=i + 1, t=float(runningState.t), dt=dtUsed, finite=finite)
    if finite:
        worst = int(err.argmax())
        fastest = int(v.argmax())
        row.update(
            rhoMin=rho.min().item(), rhoMax=rho.max().item(),
            errMean=err.mean().item(),
            worstDepth=d[worst].item() / dx, worstErr=err[worst].item(),
            vMax=v[fastest].item(), fastDepth=d[fastest].item() / dx, ke=ke,
            nOutside=int((d < 0).sum().item()),
            # `errMean` split at 2 particle spacings from the wall: is the
            # error wall-localized or bulk?
            errNear=err[d < 2 * dx].mean().item() if bool((d < 2 * dx).any()) else float('nan'),
            errFar=err[d >= 2 * dx].mean().item() if bool((d >= 2 * dx).any()) else float('nan'),
        )
    for tag, st in solveLog.items():
        for k, val in st.items():
            row[f'{tag}.{k}'] = val
    rows.append(row)
    if finite:
        snapshots.append((i + 1, rho.detach().clone(), v.detach().clone(), d.clone()))
    if not finite:
        print(f'DIVERGED at step {i + 1}, t={row["t"]:.4f}')
        break
    if row['t'] >= args.tlimit:
        break

for snapIdx in ([0, -1] if len(snapshots) > 1 else [0]):
    stepNo, rho_, v_, d_ = snapshots[snapIdx]
    dxv = dx
    print(f"\n--- wall-depth profile at step {stepNo} "
          f"(depth in particle spacings) ---")
    print(f"{'depth band':>16} {'n':>7} {'mean rho':>10} {'min rho':>9} {'max rho':>9} "
          f"{'mean|rho-1|':>12} {'mean |v|':>9}")
    edges = [-1e9, -6, -4, -2, -1, 0, 1, 2, 4, 8, 16, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_ >= lo * dxv) & (d_ < hi * dxv)
        n = int(m.sum().item())
        if n == 0:
            continue
        label = f"[{lo:g},{hi:g})" if abs(lo) < 1e8 else f"(<{hi:g})"
        if abs(hi) > 1e8:
            label = f"[{lo:g},inf)"
        print(f"{label:>16} {n:7d} {rho_[m].mean().item():10.4f} {rho_[m].min().item():9.4f} "
              f"{rho_[m].max().item():9.4f} {(rho_[m] - 1).abs().mean().item():12.4e} "
              f"{v_[m].mean().item():9.4f}")

cols = ['step', 't', 'dt', 'rhoMin', 'rhoMax', 'errNear', 'errFar',
        'vMax', 'ke', 'nOutside', 'PS.shiftDx', 'PS.shiftDepth', 'PS.pMax',
        'DF.pMax', 'DF.aMax']
cols = [c for c in cols if any(c in r for r in rows)]
hdr = ''.join(f'{c:>11}' for c in cols)


def render(r):
    out = ''
    for c in cols:
        val = r.get(c, float('nan'))
        out += f'{val:11d}' if isinstance(val, int) else f'{val:11.4g}'
    return out


print(f'\n=== bounded randomFlowIncompressible nx={args.nx} mode={args.mode} '
      f'noPenShift={args.noPenShift} : {len(rows)} steps ===')
print(hdr)
show = sorted(set(list(range(0, len(rows), args.every))
                  + list(range(max(0, len(rows) - args.tail), len(rows)))))
for i in show:
    print(render(rows[i]))
