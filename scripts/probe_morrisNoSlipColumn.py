"""OPEN_PROBLEMS.md §7: does the Morris 1997 shear-carrying viscosity give
`hydrostaticColumn` a clean viscous no-slip wall?

Re-grades the post-Part-41 table of `docs/historic_plans/DFSPH_FINDINGS.md`
§1.14 (there: `iisph`, nx=128, 1200 steps, tail = last quarter; here by default
`divergenceFree`, since `iisph` no longer holds the free-slip arm — OPEN §13), where `wallBC=noSlip`
+ `nu=0.01` through the normal-projected term bounded the slosh (KE 4x down)
but roughened the surface (embMin 0.94 -> 0.60, |v|max 3.7), while Part 39's
since-removed bespoke shear Laplacian held embMin 0.94-0.97. Arms:

  freeSlip            nu = 0                      (the case default)
  noSlip_projected    nu = 0.01, monaghanGingold  (the Part 41 row)
  noSlip_morris       nu = 0.01, morris1997       (the new term)
  freeSlip_morris     nu = 0.01, morris1997       (viscosity without the wall)

Each arm: video, streamed rows, velocity alarm + stall watchdog. Summary table
-> <out>/SUMMARY.md.

    python scripts/probe_morrisNoSlipColumn.py [--steps 1200] [--nx 128] [--arms a,b]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_morrisNoSlipColumn')

ARMS = {
    'freeSlip': dict(wallBC='freeSlip', nu=0.0, viscousTerm='monaghanGingold'),
    'noSlip_projected': dict(wallBC='noSlip', nu=0.01, viscousTerm='monaghanGingold'),
    'noSlip_morris': dict(wallBC='noSlip', nu=0.01, viscousTerm='morris1997'),
    'freeSlip_morris': dict(wallBC='freeSlip', nu=0.01, viscousTerm='morris1997'),
}


def _tail(rows, key):
    n = len(rows)
    vals = [r[key] for r in rows[3 * n // 4:] if key in r and r[key] == r[key]]
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=128)
    ap.add_argument('--steps', type=int, default=1200)
    # divergenceFree: the case default; `iisph` (the DFSPH_FINDINGS 1.14 table's
    # scheme) no longer holds even the free-slip arm (OPEN_PROBLEMS.md §13)
    ap.add_argument('--scheme', default='divergenceFree')
    ap.add_argument('--integrationScheme', default=None,
                    help="None = the case's own; the 1.14 table ran iisph with semiImplicitEuler")
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--plotInterval', type=int, default=10)
    ap.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--out', default=DEFAULT_OUT)
    from _runWatch import addWatchArguments, watchOverrides
    addWatchArguments(ap)
    args = ap.parse_args()

    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.cases import hydrostaticColumn
    from warpSPH.runner import run

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for arm in args.arms.split(','):
        params = dict(ARMS[arm])
        kw = dict(nx=args.nx, nSteps=args.steps, scheme=args.scheme, params=params,
                  quiet=True, progress=True, store=False)
        if args.integrationScheme:
            kw['integrationScheme'] = args.integrationScheme
        if args.video:
            kw.update(plot=True, video=True, show=False, plotInterval=args.plotInterval,
                      exportRoot=os.path.join(args.out, arm))
        kw.update(watchOverrides(args))
        print(f'[{arm}] {params}', flush=True)
        r = run(hydrostaticColumn.hydrostaticColumnCase, **kw)
        rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
        vmax = _tail(rows, 'maxVelocity')
        ke = _tail(rows, 'kineticEnergy')
        emb = _tail(rows, 'embeddedMinDensity')
        slope = _tail(rows, 'pressureSlopeRatio')
        res = dict(
            diverged=bool(r.diverged), steps=r.nSteps, stopReason=r.stopReason,
            vmaxMean=sum(vmax) / len(vmax) if vmax else float('nan'),
            vmaxMax=max(vmax) if vmax else float('nan'),
            keMean=sum(ke) / len(ke) if ke else float('nan'),
            embMin=min(emb) if emb else float('nan'),
            slope=sum(slope) / len(slope) if slope else float('nan'),
            alarms=len(r.velocityAlarms), video=r.videoPath, params=params)
        results[arm] = res
        print(f'[{arm}] {json.dumps(res)}', flush=True)
        with open(os.path.join(args.out, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    lines = [f'# Morris no-slip column A/B ({args.scheme}, nx={args.nx}, {args.steps} steps, tail = last quarter)', '',
             '| arm | wallBC | nu | term | \\|v\\|max mean | \\|v\\|max max | KE mean | embMin | slope | diverged |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for arm, res in results.items():
        p = res['params']
        lines.append(f"| {arm} | {p['wallBC']} | {p['nu']:g} | {p['viscousTerm']} | {res['vmaxMean']:.3g} | "
                     f"{res['vmaxMax']:.3g} | {res['keMean']:.3g} | {res['embMin']:.3f} | {res['slope']:.3f} | "
                     f"{res['diverged']} |")
    with open(os.path.join(args.out, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
