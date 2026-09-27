"""Probe (`PST_ALE_PLAN.md` Part 8 step 2's last open item / `ACSPH_PLAN.md`'s
own "next action"): a multi-seed, multi-resolution version of
`probe_michelHydrostaticColumn.py`'s single-seed, single-resolution (nx=32)
table, before trusting that finding beyond "directionally, it replicates".

That script measured, once, at nx=32: the Michel shift takes `pairedFraction`
("exactly and only what particle shifting exists to prevent") from 0.065 to
exactly 0.000, but does not bound the near-wall corner velocity the way the
`noPenetrationShift` mDBC safeguard does (`||v||` peak ~2.3 with the shift
alone vs. ~0.75 with the safeguard alone). Both numbers came from one
particle configuration -- the perfectly regular sampled lattice, since
`hydrostaticColumn.initialConditions`'s own jitter call is dead code (commented
out; the case's `jitter` param exists but is never applied). A single regular
lattice can hide or manufacture symmetric artefacts a jittered one would not
share, and a single resolution cannot distinguish a real effect from one that
happens to appear only at nx=32.

This re-enables that dead jitter path from the outside (`shuffleParticles`,
`modules/noise/shuffleParticles.py` -- the same function the case's own
commented-out call names, called here with a seeded `torch.manual_seed` so
each seed is reproducible) via a wrapped `initialConditions`, exactly the
established pattern other probes use to wrap `configureScheme`
(`probe_michelHydrostaticColumn.py`, `probe_squarePatchAreaConservation.py`).
Sweeps `nx` in {24, 32, 48} x 3 seeds x the same four modes
(`neither`/`noPenetrationShift`/`michelShift`/`both`), at a reduced step count
(default 50, not 200 -- measured single-run cost is ~1 s/step at nx=32 on one
CPU core, ~2.1 s/step at nx=48, so the full 12-configuration x 3-seed matrix
at 200 steps would run well over 3 hours; `pairedFraction`/`nnDistP01` are
steady-state per-step diagnostics that stabilise long before 200 steps in the
single-seed run, so this trades run length for breadth, the right trade for a
*consistency* check across configurations rather than one long run).

Usage:
  python scripts/probe_michelHydrostaticColumnSweep.py [--nx 24 32 48]
      [--seeds 0 1 2] [--nSteps 50] [--jitter 0.1]
      [--modes neither noPenetrationShift michelShift both]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, nargs='+', default=[24, 32, 48])
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--nSteps', type=int, default=50)
parser.add_argument('--jitter', type=float, default=0.1,
                     help='shuffleParticles jitterAmount, as a fraction of the support radius')
parser.add_argument('--modes', nargs='+',
                     default=['neither', 'noPenetrationShift', 'michelShift', 'both'],
                     choices=['neither', 'noPenetrationShift', 'michelShift', 'both'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch

from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.runner import run
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingScheme, ShiftingProjectionScheme
from warpSPH.modules.noise.shuffleParticles import shuffleParticles


def _configure(mode, seed):
    baseConfigureScheme = case.configureScheme
    baseInitialConditions = case.initialConditions

    def wrappedConfigureScheme(ctx):
        baseConfigureScheme(ctx)
        sc = ctx.schemeConfig
        if mode in ('noPenetrationShift', 'both'):
            sc.noPenetrationShift = True
        if mode in ('michelShift', 'both'):
            sc.shiftProperties.active = True
            sc.shiftProperties.scheme = ShiftingScheme.michel2022
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.michel2022

    def wrappedInitialConditions(ctx, system):
        # Re-enable `hydrostaticColumn`'s own dead jitter path (see module
        # docstring) so each seed is a genuinely different particle
        # configuration, not the same regular lattice four times over.
        # Must run *before* `baseInitialConditions` seeds the analytic
        # pressure profile from `positions[:, 1]`.
        torch.manual_seed(seed)
        particles = system.state
        particles.positions = shuffleParticles(
            particles, ctx.config, ctx.schemeConfig, 0, jitterAmount=args.jitter)
        baseInitialConditions(ctx, system)

    return wrappedConfigureScheme, wrappedInitialConditions


hdr = (f"{'nx':>4} {'seed':>5} {'mode':>18} {'steps':>6} {'final t':>8} {'div':>4} "
       f"{'vMaxPeak':>9} {'pairedFrac':>11} {'nnDistP01':>10}")
print(hdr)
print('-' * len(hdr))

# Aggregated per (nx, mode) across seeds, for the summary table at the end.
agg: dict = {}

for nx in args.nx:
    for mode in args.modes:
        vmaxRuns, pairedRuns, nndRuns = [], [], []
        for seed in args.seeds:
            _origConfigureScheme = case.configureScheme
            _origInitialConditions = case.initialConditions
            case.configureScheme, case.initialConditions = _configure(mode, seed)
            try:
                r = run(case, scheme='artificialCompressible',
                        nx=nx, nSteps=args.nSteps,
                        store=False, plot=False, quiet=True, progress=False)
            finally:
                case.configureScheme = _origConfigureScheme
                case.initialConditions = _origInitialConditions

            tr = [row for row in r.trajectory if 'maxVelocity' in row]
            if not tr:
                print(f"{nx:4d} {seed:5d} {mode:>18}   (no finite step)")
                continue

            t = np.array([row['t'] for row in tr])
            vmax = np.array([row['maxVelocity'] for row in tr])
            paired = np.array([row.get('pairedFraction', np.nan) for row in tr])
            nnd = np.array([row.get('nnDistP01', np.nan) for row in tr])

            vmaxPeak, pairedFinal, nndFinal = vmax.max(), paired[-1], nnd[-1]
            print(f"{nx:4d} {seed:5d} {mode:>18} {len(tr)-1:6d} {t[-1]:8.3f} "
                  f"{('yes' if r.diverged else 'no'):>4} "
                  f"{vmaxPeak:9.3f} {pairedFinal:11.4f} {nndFinal:10.3f}", flush=True)

            vmaxRuns.append(vmaxPeak)
            pairedRuns.append(pairedFinal)
            nndRuns.append(nndFinal)

        if vmaxRuns:
            agg[(nx, mode)] = (np.array(vmaxRuns), np.array(pairedRuns), np.array(nndRuns))

print()
print("Summary across seeds (mean +/- std over "
      f"{len(args.seeds)} seeds per (nx, mode)):")
shdr = (f"{'nx':>4} {'mode':>18} {'vMaxPeak':>18} {'pairedFrac':>18} {'nnDistP01':>18}")
print(shdr)
print('-' * len(shdr))
for nx in args.nx:
    for mode in args.modes:
        if (nx, mode) not in agg:
            continue
        vmaxRuns, pairedRuns, nndRuns = agg[(nx, mode)]
        print(f"{nx:4d} {mode:>18} "
              f"{vmaxRuns.mean():8.3f}+/-{vmaxRuns.std():<7.3f} "
              f"{pairedRuns.mean():8.4f}+/-{pairedRuns.std():<7.4f} "
              f"{nndRuns.mean():8.3f}+/-{nndRuns.std():<7.3f}")
