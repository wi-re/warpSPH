"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 11, 2026-08-28): Bender, Westhofen
& Jeske 2023, "Consistent SPH Rigid-Fluid Coupling" (VMV), against the modes
this codebase already had.

The paper derives DFSPH from a density *constraint* defined for fluid
particles only. Because a static boundary particle cannot move, `dC_i/dx_k = 0`,
and three things follow: the diagonal keeps the boundary in the squared-sum
term and drops it from the sum-of-squares (their Eq. 32); the Laplacian's
boundary term carries only `a^p_i` (Eq. 34); and the pressure acceleration
contains **no boundary pressure value at all** (Eq. 33). The first two are
`BoundaryOperatorTerms.staticBoundary`, which this codebase already had. The
third is what `BoundaryPressureMode.consistent` adds, together with the
paper's boundary *state*: `rho_k = rho0`, "static fluid particles", instead of
this codebase's mDBC-extrapolated boundary density.

The rows below are chosen so each isolates one step of that:

  mdbcDensity + full            the shipped baseline
  mdbcDensity + staticBoundary  Part 9: the paper's operator terms only
  consistent                    + rho0 boundary density, + p_b pinned at 0
  consistent + akinciVolume     + the paper's m~_k = rho0 / sum_l W_kl

`BoundaryPressureMode.mdbcMlsPressure` -- Band et al.'s MLS extrapolation, the
method this paper is written against -- used to have a fifth row here. Part 2
first measured it as this codebase's best mode; that call was later retracted
(DFSPH_FINDINGS.md Sec. 3 item 1 -- worst configuration measured over a real
900-step run) and the mode itself removed in the pre-merge cleanup pass
(09-04): `wallPressureExtrapolation` supersedes it without the feedback-loop
instability its one-step lag caused. See DFSPH_IMPROVEMENT_PLAN.md.

Usage:
  python scripts/probe_consistentCoupling.py --nx 128 --nsteps 900
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="0.4 is Bender & Koschier's published constant, and since Part 12 `cflFactor` multiplies the particle diameter, so the number here is theirs. Numbers recorded in DFSPH_IMPROVEMENT_PLAN.md before Part 12 say `cflFactor=0.1`, which was the same timestep under the old support-radius convention")
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--rows', nargs='*', default=None,
                    help="subset of row labels to run")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

from warpSPH.runner.cli import caseMain
import torch

from warpSPH.configurations import BoundaryOperatorTerms, BoundaryPressureMode
from warpSPH.modules.incompressible.consistent import akinciBoundaryMass

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

def applyAkinciMassToState():
    """The paper's `m~_k` as a *property* of the boundary particles rather than
    a solve-time substitution.

    `BoundaryPressureMode.consistent`'s own `akinciBoundaryVolume` swaps the
    masses inside the pressure solve only, so the fluid's density summation
    (the paper's Eq. 14) never sees them. Setting them on the state at build
    time is the faithful reading: boundary particles are static, so `m~_k` is
    computed once and is thereafter just their mass -- which then also raises
    the near-wall fluid density, as Eq. 14 says it should.
    """
    _orig = case.buildSystem

    def _wrapped(ctx, _orig=_orig):
        system = _orig(ctx)
        state = system.state
        boundary = state.kinds == 1
        if bool(boundary.any()):
            psi = akinciBoundaryMass(state, ctx.config, None, ctx.param('rho0'))
            ratio = (psi[boundary] / state.masses[boundary])
            print(f"[akinci mass] m~/m_nominal: mean={float(ratio.mean()):.4f} "
                  f"min={float(ratio.min()):.4f} max={float(ratio.max()):.4f}")
            state.masses = torch.where(boundary, psi, state.masses)
        return system

    case.buildSystem = _wrapped
    return lambda: setattr(case, 'buildSystem', _orig)


# label -> (boundaryPressureMode, boundaryOperatorTerms, akinciBoundaryVolume)
ROWS = [
    ('mdbcDensity+full',      BoundaryPressureMode.mdbcDensity,     BoundaryOperatorTerms.full,           False),
    ('mdbcDensity+static',    BoundaryPressureMode.mdbcDensity,     BoundaryOperatorTerms.staticBoundary, False),
    ('consistent',            BoundaryPressureMode.consistent,      BoundaryOperatorTerms.full,           False),
    ('consistent+akinciVol',  BoundaryPressureMode.consistent,      BoundaryOperatorTerms.full,           True),
    # `massInState` rows put `m~_k` on the particles instead of substituting it
    # inside the solve, so the density summation sees it too (Eq. 14).
    ('consistent+massInState', BoundaryPressureMode.consistent,     BoundaryOperatorTerms.full,           'state'),
    ('mdbcDensity+massInState', BoundaryPressureMode.mdbcDensity,   BoundaryOperatorTerms.full,           'state'),
]
if args.rows:
    ROWS = [r for r in ROWS if r[0] in args.rows]

results = []
for label, bpm, terms, akinci in ROWS:
    _origCfg = case.configureScheme
    _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible
    res = {'df': [], 'ps': []}

    restoreBuild = applyAkinciMassToState() if akinci == 'state' else (lambda: None)

    def _wrapped(ctx, _orig=_origCfg, bpm=bpm, terms=terms, akinci=akinci):
        _orig(ctx)
        sc = ctx.schemeConfig.solverConfig
        sc.boundaryPressureMode = bpm
        sc.boundaryOperatorTerms = terms
        sc.akinciBoundaryVolume = (akinci is True)

    def watchDF(*a, _f=_df, **k):
        out = _f(*a, **k)
        res['df'].append(out[2][-1] if out[2] else float('nan'))
        return out

    def watchPS(*a, _f=_ps, **k):
        out = _f(*a, **k)
        res['ps'].append(out[2][-1] if out[2] else float('nan'))
        return out

    case.configureScheme = _wrapped
    dfsphmod.solveDivergenceFree = watchDF
    sysmod.solveIncompressible = watchPS
    try:
        r = caseMain(case, argv=[
            '--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
            '--cflFactor', str(args.cflFactor), '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _origCfg
        dfsphmod.solveDivergenceFree = _df
        sysmod.solveIncompressible = _ps
        restoreBuild()
    results.append((label, r, res))


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float('nan')


print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} nSteps={args.nsteps} "
      f"cflFactor={args.cflFactor} ===")
print(f"{'configuration':>22} {'steps':>6} {'div':>5} {'minRho':>9} {'maxRho':>9} "
      f"{'|rho-1| 2nd half':>17} {'t_final':>8} {'DF resid':>10} {'PS resid':>10} {'wall s':>8}")
for label, r, res in results:
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    if not tr:
        print(f"{label:>22} {0:6d} {'True':>5}   (no finite step -- diverged immediately)")
        continue
    tail = tr[len(tr) // 2:] or tr
    band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0)) for row in tail]
    half = lambda xs: xs[len(xs) // 2:] or xs
    print(f"{label:>22} {len(tr):6d} {str(r.diverged):>5} "
          f"{min(row['minDensity'] for row in tr):9.5f} "
          f"{max(row['maxDensity'] for row in tr):9.5f} "
          f"{sum(band) / max(1, len(band)):17.4e} "
          f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):8.4f} "
          f"{mean(half(res['df'])):10.4e} {mean(half(res['ps'])):10.4e} {r.wallTime:8.1f}")
