"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 17, 2026-08-29): the three
`ShiftApplication` modes, re-measured at the *current* defaults.

The mode question has been open since Part 5, and the evidence it rests on has
rotted underneath it:

- **`tgv`'s decay rates** (0.55x for `positionShift`, 3.2x/3.4x for the two
  velocity modes) still stand as measured -- `tgv` has no `kind != 0` particles
  and `minShift` was already its default, so Part 14 was a strict no-op there.
- **The bounded-case table in §6 does not.** It is labelled "legacy CFL, so
  `positionShift` is at its death state", i.e. it was taken at 1.2 particle
  spacings of travel per step -- the timestep Part 13 then showed has **no
  viable configuration at all**, the shipped default included. It also predates
  the gauge and boundary-operator defaults Part 14 landed. Every wall-behaviour
  claim for the velocity modes (t=8.0 against a NaN, 9x lower near-wall error,
  63 against 4506 penetrating particles) comes from that table.
- **`shearWave`** (Part 16) is new and says the three modes are
  indistinguishable on dissipation when there is no pressure gradient.

Four rankings in that document have already inverted when something underneath
them changed, and three things changed under this one at once. So it gets
re-run rather than re-argued:

  tgv       artificial viscosity against an analytic decay rate, the axis the
            velocity modes are supposed to lose on
  bounded   `randomFlowIncompressible --bounded` at the published CFL and the
            Part 14 defaults, the axis they are supposed to win on
  shear     `shearWave`, which separates dissipation from volume error
            (run by `probe_shearWave.py --mode shift`; included here with
            `--cases shear` for a single table)

Usage:
  python scripts/probe_shiftApplication.py                    # tgv + bounded
  python scripts/probe_shiftApplication.py --cases bounded
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cases', nargs='*', default=['tgv', 'bounded'],
                    choices=['tgv', 'bounded', 'shear'])
parser.add_argument('--modes', nargs='*',
                    default=['positionShift', 'positionAndVelocity', 'inStepVelocity'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--tgvSteps', type=int, default=200)
parser.add_argument('--boundedSteps', type=int, default=900)
parser.add_argument('--shearT', type=float, default=2.0)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import math

import numpy as np

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import ShiftApplication


def run(case, argv, mode):
    _orig = case.configureScheme

    def _wrapped(ctx):
        _orig(ctx)
        ctx.schemeConfig.solverConfig.shiftApplication = getattr(ShiftApplication, mode)

    case.configureScheme = _wrapped
    try:
        return caseMain(case, argv=argv + ['--quiet', '--no-store', '--no-plot'])
    finally:
        case.configureScheme = _orig


def finite(r):
    return [row for row in r.trajectory
            if all(not isinstance(v, float) or math.isfinite(v) for v in row.values())]


def header(cols):
    print(' '.join(cols))
    print('-' * (sum(len(c) for c in cols) + len(cols) - 1))


if 'tgv' in args.cases:
    from warpSPH.cases.periodic.tgv import tgvCase, analyticDecayRate

    print(f"\n=== tgv nx={args.nx} nSteps={args.tgvSteps} -- artificial viscosity ===")
    print("`decay/analytic` 1.0 would be the analytic rate; `tests/test_physics.py` "
          "documents\n0.55x for the default and attributes it to the Monaghan "
          "viscosity switch.\n")
    cols = [f"{'mode':>20}", f"{'decay/analytic':>15}", f"{'monotone':>9}",
            f"{'KE(end)/KE(0)':>14}", f"{'max rho':>9}", f"{'min rho':>9}",
            f"{'wall s':>7}"]
    header(cols)
    for mode in args.modes:
        r = run(tgvCase, ['--nx', str(args.nx), '--nSteps', str(args.tgvSteps)], mode)
        tr = finite(r)
        if len(tr) < 3:
            print(f"{mode:>20}   (no finite trajectory)", flush=True)
            continue
        ke = np.array([row['kineticEnergy'] for row in tr])
        t = np.array([row['t'] for row in tr])
        measured = -np.polyfit(t, np.log(ke), 1)[0]
        analytic = analyticDecayRate(r.ctx)
        rho = [(row.get('maxDensity'), row.get('minDensity')) for row in tr
               if row.get('maxDensity') is not None]
        maxRho = max(x[0] for x in rho) if rho else float('nan')
        minRho = min(x[1] for x in rho) if rho else float('nan')
        print(f"{mode:>20} {measured / analytic:15.4f} "
              f"{str(bool(np.all(np.diff(ke) < 0))):>9} {ke[-1] / ke[0]:14.6f} "
              f"{maxRho:9.5f} {minRho:9.5f} {r.wallTime:7.1f}", flush=True)

if 'bounded' in args.cases:
    from warpSPH.cases.periodic.randomFlowIncompressible import randomFlowIncompressibleCase as boundedCase

    print(f"\n=== randomFlowIncompressible --bounded nx={args.nx} "
          f"nSteps={args.boundedSteps}, published CFL, Part 14 defaults ===")
    print("The §6 table this replaces was taken at the legacy CFL, where Part 13 "
          "found no\nviable configuration at all.\n")
    cols = [f"{'mode':>20}", f"{'steps':>6}", f"{'div':>5}", f"{'min rho':>8}",
            f"{'max rho':>8}", f"{'band 2nd half':>14}", f"{'t_final':>8}",
            f"{'wall s':>7}"]
    header(cols)
    for mode in args.modes:
        r = run(boundedCase, ['--nx', str(args.nx), '--nSteps', str(args.boundedSteps),
                              '--tLimit', '1000.0', '--bounded'], mode)
        tr = finite(r)
        if not tr:
            print(f"{mode:>20}   (no finite step)", flush=True)
            continue
        tail = tr[len(tr) // 2:] or tr
        band = sum(max(abs(x['maxDensity'] - 1.0), abs(x['minDensity'] - 1.0))
                   for x in tail) / len(tail)
        print(f"{mode:>20} {len(tr):6d} {str(r.diverged):>5} "
              f"{min(x['minDensity'] for x in tr):8.5f} "
              f"{max(x['maxDensity'] for x in tr):8.5f} {band:14.4e} "
              f"{tr[-1].get('t', float('nan')):8.4f} {r.wallTime:7.1f}", flush=True)

if 'shear' in args.cases:
    from warpSPH.cases.periodic.shearWave import shearWaveCase

    print(f"\n=== shearWave nx={args.nx} t={args.shearT} nu=0 -- "
          f"dissipation and volume error, separated ===")
    cols = [f"{'mode':>20}", f"{'steps':>6}", f"{'amplitude':>10}", f"{'max rho':>9}",
            f"{'disorder':>10}", f"{'wall s':>7}"]
    header(cols)
    for mode in args.modes:
        r = run(shearWaveCase, ['--nx', str(args.nx), '--tLimit', str(args.shearT)], mode)
        tr = finite(r)
        if not tr:
            print(f"{mode:>20}   (no finite step)", flush=True)
            continue
        print(f"{mode:>20} {len(tr) - 1:6d} {tr[-1]['amplitudeRatio']:10.6f} "
              f"{max(x['maxDensity'] for x in tr):9.5f} "
              f"{tr[-1]['residualVelocity']:10.4e} {r.wallTime:7.1f}", flush=True)
