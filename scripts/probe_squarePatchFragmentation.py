"""Why does the rotating square patch shatter at late tω when Sun et al. 2019
Fig. 14 keeps coherent arms to tω=8?  (`WCSPH_SHIFTING_PLAN.md`, the
"arms fragment" note.)

Answer (this probe): **under-resolution.** The shatter time (`tw@sf.9`, the tω
where `surfaceFraction >= 0.9`) scales ~linearly with `nx` -- nx 72 -> tω 1.6,
nx 216 -> tω 3.2 -- so the paper's L/Δx=400 holds the arms past tω=8. The arms
are pressureless SPH filaments with no surface tension; they hold only while a
few particles thick. Ruled out here: kernel (Wendland2 == Wendland4 shatter
time) and IC (`samplingScheme` regular == jittered, byte-identical).

Sweeps `--nx` × `--kernels` × `--samplings` × `--modes` and reports at the
final frame:
  - `tw@sf.9`      -- tω at which the patch is 90% "surface" (shattered);
  - `sf@t1`        -- surfaceFraction at the last frame;
  - `rms/rms0@t1`  -- cloud spread vs its t=0 value;
  - `maxRho@t1`.

Usage:
  python scripts/probe_squarePatchFragmentation.py [--nx 96 [144 ...]]
      [--tLimit 1.0] [--kernels Wendland2 Wendland4]
      [--samplings regular jittered] [--modes shiftOff surfaceNormal]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, nargs='+', default=[96])
parser.add_argument('--tLimit', type=float, default=1.0)
parser.add_argument('--kernels', nargs='+', default=['Wendland2', 'Wendland4'])
parser.add_argument('--modes', nargs='+', default=['shiftOff', 'surfaceNormal'])
parser.add_argument('--samplings', nargs='+', default=['regular'])
parser.add_argument('--every', type=int, default=200)
parser.add_argument('--omega', type=float, default=4.0)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase as case
from warpSPH.runner import run
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme


def _cfg(mode):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        sc = ctx.schemeConfig
        if mode == 'shiftOff':
            sc.shiftProperties.active = False
        elif mode == 'surfaceZeroed':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.mat
        elif mode == 'surfaceNormal':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal

    return wrapped


hdr = (f"{'nx':>5} {'kernel':>10} {'sampling':>9} {'mode':>13} "
       f"{'tw@sf.9':>8} {'sf@t1':>7} {'rms/rms0@t1':>11} {'maxRho@t1':>10}")
print(hdr)
print('-' * len(hdr))

for nx in args.nx:
    for kernel in args.kernels:
        for sampling in args.samplings:
            for mode in args.modes:
                _orig = case.configureScheme
                case.configureScheme = _cfg(mode)
                try:
                    r = run(case, params={'shape': 'box', 'omega': args.omega},
                            nx=nx, tLimit=args.tLimit, nSteps=None,
                            kernel=kernel, samplingScheme=sampling,
                            store=False, plot=False, quiet=True, progress=False)
                finally:
                    case.configureScheme = _orig

                rows = [row for row in r.trajectory
                        if 'rmsRadius' in row and row.get('step', -1) >= 0]
                rms0 = rows[0]['rmsRadius'] if rows else 1.0
                # tω at which the patch is 90% "surface" (shattered)
                twShatter = next((row['t'] * args.omega for row in rows
                                  if row['surfaceFraction'] >= 0.9), float('nan'))
                last = rows[-1] if rows else {}
                print(f"{nx:>5} {kernel:>10} {sampling:>9} {mode:>13} "
                      f"{twShatter:>8.2f} {last.get('surfaceFraction', float('nan')):>7.3f} "
                      f"{last.get('rmsRadius', rms0) / rms0:>11.3f} "
                      f"{last.get('maxDensity', float('nan')):>10.4f}"
                      + ('  DIVERGED' if r.diverged else ''))
