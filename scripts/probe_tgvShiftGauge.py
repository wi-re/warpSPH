"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-27): does making
`ShiftPressureGauge.minShift` the default change the Taylor-Green vortex?

`tgv` is the other case the gauge actually reaches: it is periodic with
complete kernel support and no boundary particles, so unlike every bounded
case it does *not* fall back to the historical clamp -- and it uses
`solveIncompressible` twice over, once for its 32-step lattice relaxation
and once per step inside `divergenceFree`'s own `finalize`.

Unlike `kolmogorovIncompressible` (forced turbulence, no reference solution),
TGV has an analytic answer to check against: `KE(t) = KE(0) exp(-4 nu k^2 t)`.
So this reports the measured decay rate as a fraction of the analytic one --
the same quantity `tests/test_physics.py::test_tgvKineticEnergyDecaysAtRoughly
TheAnalyticRate` asserts on, which sits near 0.55-0.6 for reasons that test's
docstring explains (the Monaghan viscosity switch, not discretisation error).
A gauge change must leave that fraction alone.

Usage: `python scripts/probe_tgvShiftGauge.py [--nx 64] [--nsteps 200]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--nsteps', type=int, default=200)
parser.add_argument('--gauges', nargs='*', default=['nonNegativeClamp', 'minShift'])
parser.add_argument('--shiftApplication', type=str, default=None,
                    choices=['positionShift', 'positionAndVelocity',
                             'inStepVelocity'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.periodic.tgv import tgvCase, analyticDecayRate
from warpSPH.runner import run
from warpSPH.configurations import ShiftPressureGauge

print(f"{'gauge':>18} {'steps':>7} {'diverged':>9} {'monotoneKE':>11} "
      f"{'rate/analytic':>14} {'KE_end/KE_0':>12} {'minRho':>9} {'maxRho':>9} {'wall s':>8}")
for gaugeName in args.gauges:
    gauge = ShiftPressureGauge[gaugeName]
    _orig = tgvCase.configureScheme

    def _wrapped(ctx, _orig=_orig, gauge=gauge):
        _orig(ctx)
        ctx.schemeConfig.solverConfig.shiftPressureGauge = gauge
        if args.shiftApplication is not None:
            from warpSPH.configurations import ShiftApplication
            ctx.schemeConfig.solverConfig.shiftApplication = \
                ShiftApplication[args.shiftApplication]

    tgvCase.configureScheme = _wrapped
    try:
        r = run(tgvCase, nx=args.nx, nSteps=args.nsteps, tLimit=1e9,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        tgvCase.configureScheme = _orig

    ke = r.series('kineticEnergy')
    t = np.array([row['t'] for row in r.trajectory if 'kineticEnergy' in row])
    # Least-squares slope of log(KE) against t is the measured decay rate.
    finite = np.isfinite(ke) & (ke > 0)
    rate = -np.polyfit(t[finite], np.log(ke[finite]), 1)[0] if finite.sum() > 2 else float('nan')
    analytic = analyticDecayRate(r.ctx)
    # `tgv`'s own diagnostics report energy only, so the density health check
    # comes off the final state instead of the trajectory.
    finalRho = r.state.state.densities
    minRho, maxRho = finalRho.min().item(), finalRho.max().item()
    print(f"{gaugeName:>18} {len(r.trajectory) - 1:7d} {str(r.diverged):>9} "
          f"{str(bool(np.all(np.diff(ke) < 0))):>11} {rate / analytic:14.4f} "
          f"{ke[-1] / ke[0]:12.5f} {minRho:9.5f} {maxRho:9.5f} {r.wallTime:8.1f}")
