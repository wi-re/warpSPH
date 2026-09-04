"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 10, 2026-08-28): does the WCSPH
convention -- integrate `drho/dt = -rho div v` instead of re-summing the SPH
density every step -- work for this scheme, and for which of its two solves?

`DensityEvolution` makes the choice real (Part 3 found the old `integrateRho`
flag inert: `finalize` re-summed unconditionally, twice per step in total).
Three settings, and the interesting question is not speed but *drift*: an
integrated density knows about `div v` and nothing else, so it cannot see
particle-distribution drift at fixed divergence -- which is exactly the error
the constant-density/shifting solve exists to remove -- and it never accounts
for the shift's own displacement either.

So every number here is reported twice: against the **carried** density (what
the solvers and the case's own diagnostics see) and against the **true**
summation density recomputed from the positions at the same instant. Under
`summation` the two are identical by construction, which is the probe's own
self-check; under `continuity` their difference is the accumulated lie.

Usage:
  python scripts/probe_densityEvolution.py --nx 128 --nsteps 900
  python scripts/probe_densityEvolution.py --extra          # periodic variant
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
parser.add_argument('--every', type=int, default=10,
                    help="steps between true-density recomputations")
parser.add_argument('--shiftApplication', default=None,
                    help="override ShiftApplication. `inStepVelocity` drops the "
                         "position shift entirely, which is the cleanest test of "
                         "whether the shift is what the carried density cannot see")
parser.add_argument('--modes', nargs='*',
                    default=['summation', 'continuity', 'hybrid'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

import torch

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import DensityEvolution, ShiftApplication
from warpSPH.modules.density import computeDensities

import warpSPH.systems.incompressible as sysmod
import warpSPH.schemes.divergenceFree as dfsphmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')


def finite(xs):
    return [x for x in xs if math.isfinite(x)]


def mean(xs):
    xs = finite(xs)
    return sum(xs) / len(xs) if xs else float('nan')


rows = []
for name in args.modes:
    mode = DensityEvolution[name]
    _origCfg = case.configureScheme
    _ps = sysmod.solveIncompressible
    _df = dfsphmod.solveDivergenceFree
    stats = {'carried': [], 'true': [], 'drift': [], 'driftMax': [], 'dfRes': []}
    step = {'n': 0}

    def _wrapped(ctx, _orig=_origCfg, mode=mode):
        _orig(ctx)
        ctx.schemeConfig.solverConfig.densityEvolution = mode
        if args.shiftApplication:
            ctx.schemeConfig.solverConfig.shiftApplication = ShiftApplication[args.shiftApplication]

    def watchPS(*a, **k):
        return _ps(*a, **k)

    def watchDF(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
        # Measured at the *divergence-free* solve, i.e. at the top of the step:
        # that is where `particles.densities` is the field the mode decided to
        # carry (`hybrid` re-sums later, inside `finalize`, so measuring there
        # would show it a drift of zero by construction and miss the point).
        step['n'] += 1
        if step['n'] % args.every == 0:
            fluid = particles.kinds == 0
            carried = particles.densities
            truth = computeDensities(particles, config, schemeConfig, adjacency)
            stats['carried'].append(float((carried[fluid] - 1.0).abs().mean()))
            stats['true'].append(float((truth[fluid] - 1.0).abs().mean()))
            d = (carried[fluid] - truth[fluid]).abs()
            stats['drift'].append(float(d.mean()))
            stats['driftMax'].append(float(d.max()))
        out = _df(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
        stats['dfRes'].append(out[2][-1] if out[2] else float('nan'))
        return out

    case.configureScheme = _wrapped
    sysmod.solveIncompressible = watchPS
    dfsphmod.solveDivergenceFree = watchDF
    try:
        r = caseMain(case, argv=[
            '--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
            '--cflFactor', str(args.cflFactor), '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _origCfg
        sysmod.solveIncompressible = _ps
        dfsphmod.solveDivergenceFree = _df
    rows.append((name, r, stats))

print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} nSteps={args.nsteps} "
      f"cflFactor={args.cflFactor} shiftApplication={args.shiftApplication or 'default'} ===")
print("'carried' is what the solvers and the case diagnostics see; 'true' is a fresh")
print("summation on the same positions. Second-half means, sampled every "
      f"{args.every} steps.\n")
print(f"{'densityEvolution':>17} {'steps':>6} {'div':>5} {'|carried-1|':>12} "
      f"{'|true-1|':>10} {'drift mean':>11} {'drift max':>10} {'DF resid':>10} "
      f"{'t_final':>8} {'wall s':>8} {'KE last/first':>13}")
for name, r, st in rows:
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    half = lambda xs: xs[len(xs) // 2:] or xs
    print(f"{name:>17} {len(tr):6d} {str(r.diverged):>5} "
          f"{mean(half(st['carried'])):12.4e} {mean(half(st['true'])):10.4e} "
          f"{mean(half(st['drift'])):11.4e} {mean(half(st['driftMax'])):10.4e} "
          f"{mean(half(st['dfRes'])):10.4e} "
          f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):8.4f} "
          f"{r.wallTime:8.1f} "
          f"{(tr[-1]['kineticEnergy'] / tr[0]['kineticEnergy'] if tr and tr[0].get('kineticEnergy') else float('nan')):13.4f}")
print("\nUnder `summation` carried == true by construction (the step opens with the")
print("summation this compares against): a nonzero drift there would mean this probe")
print("is measuring the wrong state.")
