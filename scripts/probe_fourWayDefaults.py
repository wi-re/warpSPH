"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 13, 2026-08-28): the changes that
are each better at the published CFL and each worse at 3x it, run together.

Parts 8-11 each landed one opt-in switch and measured it alone against the
shipped baseline. All of them came out the same shape -- a large win at Bender
& Koschier's published CFL, and an *earlier* death at the 3x-oversized timestep
this codebase shipped -- and the explanation offered each time was the same:
each change makes the solve less damped at the wall (a smaller `|alpha|` there
means a larger Jacobi step), which is not survivable at 1.2 particle spacings
of displacement per step.

If that explanation is right, they are not independent knobs to be traded off
against each other; they are one coherent configuration that the timestep was
blocking. That is a factorial question, so:

  cfl       0.4 (published, in Part 12's particle-diameter units)
            1.2 (the pre-Part-12 default of 0.3 support radii -- the same
                 timestep that default always meant)
  gauge     `minShift` forced through on a bounded solve, against the shipped
            fallback to `nonNegativeClamp` (`forceShiftPressureGauge`)
  boundary  four levels, see below

**The boundary axis is one factor with four levels, not two crossed ones.**
`BoundaryPressureMode.consistent` *forces* `BoundaryOperatorTerms.
staticBoundary` -- not through the config, but inside both solvers
(`incompressible.py:126`, `divergenceFree.py:262`), on the grounds that
[BWJ23]'s Eqs. 32 and 34 *are* the static-boundary operator, so the two
settings must not be allowed to disagree. Crossing them would therefore
produce two pairs of bit-identical rows. The four levels that actually differ:

  shipped     mdbcDensity + full            the shipped baseline
  static      mdbcDensity + staticBoundary  Part 9: the operator terms alone
  consistent  consistent                    + rho_k = rho0 boundary state
  akinci      consistent + akinciVolume     + m~_k inside the solve (Part 11)

Usage:
  python scripts/probe_fourWayDefaults.py --nx 128 --nsteps 900
  python scripts/probe_fourWayDefaults.py --cfls 0.4      # published half only
"""
from __future__ import annotations

import argparse

BOUNDARY_LEVELS = {
    # label -> (boundaryPressureMode, boundaryOperatorTerms, akinciBoundaryVolume)
    'shipped':    ('mdbcDensity', 'full',           False),
    'static':     ('mdbcDensity', 'staticBoundary', False),
    'consistent': ('consistent',  'full',           False),  # terms forced by the mode
    'akinci':     ('consistent',  'full',           True),
}

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--cfls', nargs='*', type=float, default=[0.4, 1.2],
                    help="0.4 is [BK]'s published constant in Part 12's particle-diameter units; 1.2 is the pre-Part-12 default of 0.3 support radii")
parser.add_argument('--gauges', nargs='*', default=['minShift', 'shipped'])
parser.add_argument('--boundary', nargs='*', default=list(BOUNDARY_LEVELS),
                    choices=list(BOUNDARY_LEVELS))
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import BoundaryOperatorTerms, BoundaryPressureMode

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float('nan')


rows = [(c, g, b) for c in args.cfls for g in args.gauges for b in args.boundary]

print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
      f"nSteps={args.nsteps} -- {len(rows)} configurations ===", flush=True)
hdr = (f"{'cfl':>5} {'gauge':>9} {'boundary':>11} {'steps':>6} {'div':>5} "
       f"{'minRho':>8} {'maxRho':>8} {'|rho-1| 2nd half':>17} {'t_final':>8} "
       f"{'DF resid':>10} {'PS resid':>10} {'wall s':>7}")
print(hdr, flush=True)
print('-' * len(hdr), flush=True)

results = []
for cfl, gauge, boundary in rows:
    bpm, terms, akinci = BOUNDARY_LEVELS[boundary]
    _origCfg = case.configureScheme
    _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible
    res = {'df': [], 'ps': []}

    def _wrapped(ctx, _orig=_origCfg, gauge=gauge, bpm=bpm, terms=terms, akinci=akinci):
        _orig(ctx)
        sc = ctx.schemeConfig.solverConfig
        sc.boundaryPressureMode = getattr(BoundaryPressureMode, bpm)
        sc.boundaryOperatorTerms = getattr(BoundaryOperatorTerms, terms)
        sc.akinciBoundaryVolume = akinci
        # `shiftPressureGauge` already defaults to `minShift`; what decides a
        # bounded solve is whether the guard is allowed to downgrade it.
        sc.forceShiftPressureGauge = (gauge == 'minShift')

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
            '--cflFactor', str(cfl), '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _origCfg
        dfsphmod.solveDivergenceFree = _df
        sysmod.solveIncompressible = _ps

    # Print as each row finishes: 16 production runs is long enough that a
    # partial table beats a complete one a killed process never prints.
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    if not tr:
        print(f"{cfl:5.2f} {gauge:>9} {boundary:>11} {0:6d} {'True':>5}"
              f"   (no finite step)", flush=True)
        results.append((cfl, gauge, boundary, r, None))
        continue
    tail = tr[len(tr) // 2:] or tr
    band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0)) for row in tail]
    err = sum(band) / max(1, len(band))
    half = lambda xs: xs[len(xs) // 2:] or xs
    print(f"{cfl:5.2f} {gauge:>9} {boundary:>11} {len(tr):6d} {str(r.diverged):>5} "
          f"{min(row['minDensity'] for row in tr):8.5f} "
          f"{max(row['maxDensity'] for row in tr):8.5f} "
          f"{err:17.4e} {tr[-1].get('t', float('nan')):8.4f} "
          f"{mean(half(res['df'])):10.4e} {mean(half(res['ps'])):10.4e} "
          f"{r.wallTime:7.1f}", flush=True)
    results.append((cfl, gauge, boundary, r, err))

print("\n=== summary by cfl half ===", flush=True)
for cfl in args.cfls:
    sub = [r for r in results if r[0] == cfl]
    ok = [r for r in sub if r[4] is not None and math.isfinite(r[4]) and not r[3].diverged]
    print(f"\ncfl={cfl}: {len(ok)}/{len(sub)} reached the step budget without diverging",
          flush=True)
    for r in sorted(sub, key=lambda r: (r[4] is None, r[4])):
        flag = 'DIVERGED' if (r[4] is None or r[3].diverged) else '        '
        e = 'n/a' if r[4] is None else f"{r[4]:.4e}"
        print(f"  {flag}  {r[1]:>9} + {r[2]:<11} |rho-1| = {e:>12}", flush=True)
