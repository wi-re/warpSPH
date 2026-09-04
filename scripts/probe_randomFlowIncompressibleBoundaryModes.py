"""Production-resolution follow-up to `DFSPH_IMPROVEMENT_PLAN.md` Part 2,
step 8's "still open" item: at nx=24 (smoke-test scale), bounded DFSPH's
density band ([0.78, 1.33]) was markedly looser than deltaSPH's ([0.998,
1.003]) over the same run, matched to physical time -- despite deltaSPH
showing a *larger* transient boundary velocity spike, which already ruled
out "coarse-grid velocity noise" as the whole story. This script reruns that
same comparison at the case's own production `nx=128` default, plus a
same-resolution, same-physical-time comparison of the `BoundaryPressureMode`
values (`plain`/`mdbcDensity`) against *each other*, which the nx=24 session
only ever compared at matched step-count (not meaningful once `dt` differs
run-to-run under DFSPH's adaptive timestep hook). (A third mode,
`mdbcMlsPressure`, used to be swept here too; removed in the pre-merge
cleanup pass, 09-04 -- see DFSPH_IMPROVEMENT_PLAN.md.)

Uses `warpSPH.runner.runner.run()` directly (not the CLI) so every mode/case
shares one in-process driver and `store=False, plot=False` keep this fast --
only the diagnostics trajectory (`weaklyCompressibleDiagnostics`: kinetic
energy, max velocity, min/max density over fluid particles) is collected.

Usage: `python scripts/probe_randomFlowIncompressibleBoundaryModes.py
[--nx 128] [--tlimit 1.5] [--print-every 20]`
"""

from __future__ import annotations

import argparse
import time

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.cases.randomFlow import randomFlowCase
from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase
from warpSPH.runner import run

BOUNDARY_MODES = ['plain', 'mdbcDensity']


def summarize(trajectory, tag, printEvery):
    minRho = min(row['minDensity'] for row in trajectory if 'minDensity' in row)
    maxRho = max(row['maxDensity'] for row in trajectory if 'maxDensity' in row)
    maxVel = max(row['maxVelocity'] for row in trajectory if 'maxVelocity' in row)
    finalT = trajectory[-1]['t']
    print(f'[{tag}] steps={len(trajectory) - 1} t_final={finalT:.4f} '
         f'rho=[{minRho:.5f},{maxRho:.5f}] maxVelocity_peak={maxVel:.4f}')
    if printEvery > 0:
        for row in trajectory:
            if row['step'] % printEvery == 0:
                print(f'  [{tag}] step={row["step"]:5d} t={row["t"]:8.5f} '
                     f'rho=[{row.get("minDensity", float("nan")):.5f},'
                     f'{row.get("maxDensity", float("nan")):.5f}] '
                     f'v={row.get("maxVelocity", float("nan")):.4f}')
    return dict(tag=tag, minRho=minRho, maxRho=maxRho, maxVel=maxVel, finalT=finalT,
               nSteps=len(trajectory) - 1)


def runDfsphMode(nx, tLimit, mode, printEvery):
    t0 = time.time()
    result = run(randomFlowIncompressibleCase, nx=nx, tLimit=tLimit,
                params={'bounded': True, 'boundaryPressureMode': mode},
                store=False, plot=False, quiet=True, progress=False)
    wall = time.time() - t0
    summary = summarize(result.trajectory, f'dfsph/{mode}', printEvery)
    summary['wall'] = wall
    print(f'[dfsph/{mode}] wall={wall:.1f}s diverged={result.diverged}')
    return summary


def runDeltaSphBaseline(nx, tLimit, printEvery):
    t0 = time.time()
    result = run(randomFlowCase, nx=nx, tLimit=tLimit, params={'bounded': True},
                store=False, plot=False, quiet=True, progress=False)
    wall = time.time() - t0
    summary = summarize(result.trajectory, 'deltaSPH', printEvery)
    summary['wall'] = wall
    print(f'[deltaSPH] wall={wall:.1f}s diverged={result.diverged}')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--tlimit', type=float, default=1.5)
    p.add_argument('--print-every', type=int, default=0)
    p.add_argument('--skip-deltasph', action='store_true')
    p.add_argument('--modes', type=str, default=','.join(BOUNDARY_MODES))
    args = p.parse_args()

    modes = args.modes.split(',')
    print(f'nx={args.nx} tLimit={args.tlimit} modes={modes} '
         f'skip_deltasph={args.skip_deltasph}')

    summaries = []
    for mode in modes:
        summaries.append(runDfsphMode(args.nx, args.tlimit, mode, args.print_every))
    if not args.skip_deltasph:
        summaries.append(runDeltaSphBaseline(args.nx, args.tlimit, args.print_every))

    print('\n=== summary ===')
    for s in summaries:
        print(f'{s["tag"]:20s} nSteps={s["nSteps"]:5d} t_final={s["finalT"]:.4f} '
             f'rho=[{s["minRho"]:.5f},{s["maxRho"]:.5f}] '
             f'maxVelocity_peak={s["maxVel"]:.4f} wall={s["wall"]:.1f}s')
