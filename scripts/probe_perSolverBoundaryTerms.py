"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 14, 2026-08-29): which solver should
get `BoundaryOperatorTerms.staticBoundary` once the gauge is fixed.

Part 13's factorial measured `minShift` + `staticBoundary` at 4.48e-3, 40x the
shipped default -- but it measured it through the *single* bundle-level knob,
which sets both solves at once. Two open items disagree about what to land:

  - §4 item 2 wants `pressureSolver = staticBoundary` with
    `divergenceFreeSolver = full`, on the grounds that Part 9 scoped the win to
    the constant-density/shifting solve (2.88e-2 alone against 3.00e-2 for
    both).
  - §10 item 4 objects that a per-solver split is exactly the
    mismatched-operator configuration behind the unexplained contraction
    collapse (§4 item 3), and asks that the landed default be the `both`
    configuration unless the half-state is explained first.

Part 9 measured both halves *under the clamp gauge*, and Part 13's headline is
that a boundary ranking measured under the clamp does not survive the gauge fix
(`akinci` went from best-measured to a NaN at step 137). So the split has to be
re-measured under `minShift`. That is what this does: the two solvers'
`boundaryOperatorTerms` crossed, at the published CFL, with the gauge forced
through on a bounded solve.

Three of the five rows are configurations Part 13 already published, and they
are here as reproduction checks -- the same discipline that validated Part 13's
own harness:

  shipped      1.7782e-1  t=4.6895   (the shipped default, clamp gauge)
  minShift/full/full       1.4318e-1  t=6.458
  minShift/static/static   4.4810e-3  t=6.150

Usage:
  python scripts/probe_perSolverBoundaryTerms.py                 # all five rows
  python scripts/probe_perSolverBoundaryTerms.py --rows ps-only  # one row
"""
from __future__ import annotations

import argparse

# label -> (forceGauge, pressureSolver terms, divergenceFreeSolver terms)
ROWS = {
    'shipped':   (False, 'full',           'full'),            # reproduction: 1.7782e-1
    'gauge':     (True,  'full',           'full'),            # reproduction: 1.4318e-1
    'both':      (True,  'staticBoundary', 'staticBoundary'),  # reproduction: 4.4810e-3
    'ps-only':   (True,  'staticBoundary', 'full'),            # new: §4 item 2's candidate
    'df-only':   (True,  'full',           'staticBoundary'),  # new: the half-state, under the gauge
}

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--cfl', type=float, default=0.4,
                    help="[BK]'s published constant, in Part 12's particle-diameter units")
parser.add_argument('--rows', nargs='*', default=list(ROWS), choices=list(ROWS))
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import BoundaryOperatorTerms, ShiftPressureGauge

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float('nan')


print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} nSteps={args.nsteps} "
      f"cflFactor={args.cfl} -- {len(args.rows)} configurations ===", flush=True)
hdr = (f"{'row':>9} {'gauge':>9} {'PS terms':>15} {'DF terms':>15} {'steps':>6} {'div':>5} "
       f"{'minRho':>8} {'maxRho':>8} {'|rho-1| 2nd half':>17} {'t_final':>8} "
       f"{'DF resid':>10} {'PS resid':>10} {'wall s':>7}")
print(hdr, flush=True)
print('-' * len(hdr), flush=True)

results = []
for label in args.rows:
    forceGauge, psTerms, dfTerms = ROWS[label]
    _origCfg = case.configureScheme
    _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible
    res = {'df': [], 'ps': []}

    def _wrapped(ctx, _orig=_origCfg, forceGauge=forceGauge, psTerms=psTerms, dfTerms=dfTerms):
        _orig(ctx)
        sc = ctx.schemeConfig.solverConfig
        # Leave the bundle-level knob at None so the per-solver settings are
        # what is actually being measured; setting it would override both.
        sc.boundaryOperatorTerms = None
        sc.pressureSolver.boundaryOperatorTerms = getattr(BoundaryOperatorTerms, psTerms)
        sc.divergenceFreeSolver.boundaryOperatorTerms = getattr(BoundaryOperatorTerms, dfTerms)
        # Set the gauge that is actually wanted rather than relying on the
        # guard to downgrade it: the whole point of this measurement is to
        # decide whether the guard should keep downgrading on a bounded solve,
        # so the rows have to stay meaningful on both sides of that change.
        sc.shiftPressureGauge = (ShiftPressureGauge.minShift if forceGauge
                                 else ShiftPressureGauge.nonNegativeClamp)
        sc.forceShiftPressureGauge = forceGauge

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
            '--cflFactor', str(args.cfl), '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _origCfg
        dfsphmod.solveDivergenceFree = _df
        sysmod.solveIncompressible = _ps

    gaugeLabel = 'minShift' if forceGauge else 'shipped'
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    if not tr:
        print(f"{label:>9} {gaugeLabel:>9} {psTerms:>15} {dfTerms:>15} {0:6d} {'True':>5}"
              f"   (no finite step)", flush=True)
        results.append((label, r, None))
        continue
    tail = tr[len(tr) // 2:] or tr
    band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0)) for row in tail]
    err = sum(band) / max(1, len(band))
    half = lambda xs: xs[len(xs) // 2:] or xs
    print(f"{label:>9} {gaugeLabel:>9} {psTerms:>15} {dfTerms:>15} {len(tr):6d} {str(r.diverged):>5} "
          f"{min(row['minDensity'] for row in tr):8.5f} "
          f"{max(row['maxDensity'] for row in tr):8.5f} "
          f"{err:17.4e} {tr[-1].get('t', float('nan')):8.4f} "
          f"{mean(half(res['df'])):10.4e} {mean(half(res['ps'])):10.4e} "
          f"{r.wallTime:7.1f}", flush=True)
    results.append((label, r, err))

print("\n=== ranked ===", flush=True)
for label, r, err in sorted(results, key=lambda x: (x[2] is None, x[2])):
    flag = 'DIVERGED' if (err is None or r.diverged) else '        '
    e = 'n/a' if err is None else f"{err:.4e}"
    print(f"  {flag}  {label:>9}  |rho-1| = {e:>12}", flush=True)

EXPECTED = {'shipped': 1.7782e-1, 'gauge': 1.4318e-1, 'both': 4.4810e-3}
print("\n=== reproduction check against Part 13 ===", flush=True)
for label, r, err in results:
    if label not in EXPECTED:
        continue
    want = EXPECTED[label]
    ok = err is not None and math.isfinite(err) and abs(err - want) <= 5e-5 * max(want, 1e-3)
    got = 'n/a' if err is None else f"{err:.4e}"
    print(f"  {'OK  ' if ok else 'DIFF'}  {label:>9}  got {got:>12}  want {want:.4e}", flush=True)
