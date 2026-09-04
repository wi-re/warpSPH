"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 15, 2026-08-29): the stopping
criterion, which §1.7 calls the last unexplained thing.

The claim on file is that `solveIncompressible` runs its full 64 iterations
every step forever, under every gauge, every `dt`, every solver. It still does,
at the Part 14 defaults. Two concrete defects were named: the absolute test
cannot be met when the source carries a structural mean the operator cannot
remove (§1.1), and the statistic floors each under-dense particle's
contribution at `-tolerance` so it cannot cancel an over-dense one, which
neither published criterion does ([BK] Alg. 3, [I] §5.1).

  --mode trace   What the three statistics do *along the same iterate path*.
      Early exit is disabled (`minIterations = maxIterations`, `rtol = 0`), so
      every run is the same simulation bit for bit and the criterion only
      changes what is *reported*. That makes the three directly comparable:
      same states, same iterates, three readings. Reports each statistic's
      value at the first and last iteration of a solve, how far it moves over
      the whole solve, and how many solves it would have ended at a given
      threshold.

  --mode budget  If the residual stalls, are the iterations past the stall
      worth anything? Sweeps each solver's `maxIterations` and reports what the
      density error and the wall time do. This is the question that decides
      whether the tolerance is mis-set or merely decorative.

  --mode ab      Whether any of them is worth switching to, end to end:
      accuracy, iteration count and wall time on a real run. A criterion that
      terminates early is only an improvement if the density error does not
      pay for it.

Usage:
  python scripts/probe_stoppingCriterion.py --mode trace --nsteps 200
  python scripts/probe_stoppingCriterion.py --mode ab --nsteps 900
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='trace', choices=['trace', 'budget', 'ab'])
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--nsteps', type=int, default=200)
parser.add_argument('--criteria', nargs='*',
                    default=['flooredOneSided', 'oneSided', 'meanAbsolute'])
parser.add_argument('--budgets', nargs='*', default=None,
                    help="--mode budget: PS:DF iteration-cap pairs, e.g. 64:32 128:16")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math
import statistics

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import JacobiConvergenceCriterion

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

SOLVERS = ('pressureSolver', 'divergenceFreeSolver')
SHIPPED_TOL = {'pressureSolver': 5e-4, 'divergenceFreeSolver': 2.5e-3}


def run(configure):
    """Run the case with `configure` applied on top of its own scheme setup,
    returning the run result and the per-solve `errors` lists of both solvers."""
    _origCfg = case.configureScheme
    _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible
    rec = {'divergenceFreeSolver': [], 'pressureSolver': []}
    press = {'divergenceFreeSolver': [], 'pressureSolver': []}

    def _wrapped(ctx, _orig=_origCfg):
        _orig(ctx)
        configure(ctx.schemeConfig.solverConfig)

    def watch(fn, key):
        def inner(*a, **k):
            out = fn(*a, **k)
            rec[key].append(out[2])
            press[key].append(out[3])
            return out
        return inner

    case.configureScheme = _wrapped
    dfsphmod.solveDivergenceFree = watch(_df, 'divergenceFreeSolver')
    sysmod.solveIncompressible = watch(_ps, 'pressureSolver')
    try:
        r = caseMain(case, argv=[
            '--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
            '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _origCfg
        dfsphmod.solveDivergenceFree = _df
        sysmod.solveIncompressible = _ps
    return r, rec, press


def band(r):
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    if not tr:
        return float('nan'), 0, float('nan')
    tail = tr[len(tr) // 2:] or tr
    e = sum(max(abs(x['maxDensity'] - 1.0), abs(x['minDensity'] - 1.0)) for x in tail) / len(tail)
    return e, len(tr), tr[-1].get('t', float('nan'))


if args.mode == 'trace':
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"nSteps={args.nsteps} -- statistics along a fixed iterate path ===")
    print("(minIterations = maxIterations and rtol = 0, so every run below is "
          "the same\n simulation and only the *reported* statistic differs)\n")
    traces = {}
    pressTraces = {}
    hdr = (f"{'solver':>20} {'criterion':>16} {'first iter':>11} {'last iter':>11} "
           f"{'reduction':>10} {'shipped tol':>12} {'would end':>10}")
    print(hdr)
    print('-' * len(hdr))
    for crit in args.criteria:
        def cfg(sc, crit=crit):
            for name in SOLVERS:
                s = getattr(sc, name)
                s.convergenceCriterion = getattr(JacobiConvergenceCriterion, crit)
                s.minIterations = s.maxIterations   # disable early exit
                s.rtol = 0.0                        # disable the relative disjunct
        r, rec, press = run(cfg)
        traces[crit] = rec
        pressTraces[crit] = press
        for name in SOLVERS:
            solves = [e for e in rec[name] if e]
            first = [e[0] for e in solves]
            last = [e[-1] for e in solves]
            tol = SHIPPED_TOL[name]
            ends = sum(1 for e in solves if any(v < tol for v in e))
            red = statistics.mean(first) / statistics.mean(last) if statistics.mean(last) else float('inf')
            print(f"{name:>20} {crit:>16} {statistics.mean(first):11.4e} "
                  f"{statistics.mean(last):11.4e} {red:9.2f}x {tol:12.1e} "
                  f"{ends:>5}/{len(solves):<4}")
    print("\n'would end' counts solves in which the statistic dropped below the "
          "solver's\nshipped tolerance at any iteration -- i.e. how often that "
          "criterion would\nhave terminated the solve early.")

    # The shape matters as much as the endpoints: a statistic that falls to its
    # floor in three iterations and then flatlines says something very
    # different from one that decays steadily for sixty-four.
    marks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    for name in SOLVERS:
        print(f"\n  mean statistic at iteration k, {name}:")
        head = '    ' + ' '.join(f'{k:>9}' for k in marks)
        print(f"{'criterion':>16}" + head)
        for crit in args.criteria:
            solves = [e for e in traces[crit][name] if e]
            cells = []
            for k in marks:
                vals = [e[k - 1] for e in solves if len(e) >= k]
                cells.append(f'{statistics.mean(vals):9.3e}' if vals else f'{"-":>9}')
            print(f"{crit:>16}    " + ' '.join(cells))

    # If the residual does not move, what *is* the loop doing? Each sweep adds
    # `omega * r / alpha` to the pressure, so a residual that stays put means a
    # pressure that grows linearly in the iteration count -- the loop would be
    # a gain knob on the shift magnitude, not a solve. `maxIterations` would
    # then be a physical parameter wearing a numerical parameter's name.
    print("\n  mean pressure RANGE (max - min) at iteration k -- gauge-invariant, "
          "so it is\n  comparable across the two solvers' different gauges:")
    crit0 = args.criteria[0]
    for name in SOLVERS:
        solves = [p for p in pressTraces[crit0][name] if p]
        cells = []
        for k in marks:
            vals = [p[k - 1][1] - p[k - 1][0] for p in solves if len(p) >= k]
            cells.append(f'{statistics.mean(vals):9.3e}' if vals else f'{"-":>9}')
        print(f"{name:>16}    " + ' '.join(cells))

elif args.mode == 'budget':
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"nSteps={args.nsteps} -- iteration budget ===")
    hdr = (f"{'PS cap':>7} {'DF cap':>7} {'PS iters':>9} {'DF iters':>9} "
           f"{'band 2nd half':>14} {'t_final':>8} {'steps':>6} {'wall s':>7}")
    print(hdr)
    print('-' * len(hdr))
    # One axis at a time: hold one solver at its shipped cap and cut the other.
    budgets = [(64, 32), (32, 32), (16, 32), (8, 32), (4, 32),
               (64, 16), (64, 8), (64, 4),
               (16, 8), (8, 8)]
    if args.budgets:
        budgets = [tuple(int(v) for v in b.split(':')) for b in args.budgets]
    for psCap, dfCap in budgets:
        def cfg(sc, psCap=psCap, dfCap=dfCap):
            sc.pressureSolver.maxIterations = psCap
            sc.pressureSolver.minIterations = min(sc.pressureSolver.minIterations, psCap)
            sc.divergenceFreeSolver.maxIterations = dfCap
            sc.divergenceFreeSolver.minIterations = min(sc.divergenceFreeSolver.minIterations, dfCap)
        r, rec, _ = run(cfg)
        e, steps, tf = band(r)
        psIters = statistics.mean([len(x) for x in rec['pressureSolver']]) if rec['pressureSolver'] else 0
        dfIters = statistics.mean([len(x) for x in rec['divergenceFreeSolver']]) if rec['divergenceFreeSolver'] else 0
        print(f"{psCap:7d} {dfCap:7d} {psIters:9.1f} {dfIters:9.1f} "
              f"{e:14.4e} {tf:8.4f} {steps:6d} {r.wallTime:7.1f}", flush=True)

else:
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"nSteps={args.nsteps} -- criteria end to end ===")
    hdr = (f"{'PS criterion':>16} {'DF criterion':>16} {'PS iters':>9} {'DF iters':>9} "
           f"{'band 2nd half':>14} {'t_final':>8} {'steps':>6} {'wall s':>7}")
    print(hdr)
    print('-' * len(hdr))
    # Tolerances stay at the shipped values throughout. That is the point: the
    # question is what happens if this codebase adopts the published criterion
    # as published, and a criterion swap at a fixed tolerance is exactly what
    # "adopt [BK] Alg. 3" means. The interesting row is `oneSided` on the
    # divergence-free solve, where cancellation puts the statistic under
    # tolerance at the first iteration.
    rows = [('flooredOneSided', 'meanAbsolute'),    # shipped
            ('flooredOneSided', 'oneSided'),        # published criterion, DF
            ('oneSided',        'meanAbsolute'),    # published criterion, PS
            ('oneSided',        'oneSided')]        # both
    for psCrit, dfCrit in rows:
        def cfg(sc, psCrit=psCrit, dfCrit=dfCrit):
            sc.pressureSolver.convergenceCriterion = getattr(JacobiConvergenceCriterion, psCrit)
            sc.divergenceFreeSolver.convergenceCriterion = getattr(JacobiConvergenceCriterion, dfCrit)
        r, rec, _ = run(cfg)
        e, steps, tf = band(r)
        psIters = statistics.mean([len(x) for x in rec['pressureSolver']]) if rec['pressureSolver'] else 0
        dfIters = statistics.mean([len(x) for x in rec['divergenceFreeSolver']]) if rec['divergenceFreeSolver'] else 0
        print(f"{psCrit:>16} {dfCrit:>16} {psIters:9.1f} {dfIters:9.1f} "
              f"{e:14.4e} {tf:8.4f} {steps:6d} {r.wallTime:7.1f}", flush=True)
