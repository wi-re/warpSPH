"""Batch A/B of `dfsphReference` on `hydrostaticColumn`, for the late-time
free-surface degradation (DFSPH_IMPROVEMENT_PLAN.md active track, Parts 29-32).

The single-run `probe_dfsphReferenceColumn.py` prints one verbose per-step
trace; this one runs several arms x several runs to `--steps` (default 1500)
without the trace and reports, per run and per arm:

- `onset`   -- first step with fluid `rho_min < --onsetThresh` (default 0.50),
              the Part 31 metric for the degradation's start;
- `rhoMin`  -- fluid min density sampled at `--samples` (default
              300,600,900,1200,1500) and the run-minimum;
- `vMax`    -- fluid max |v| at the same samples and the run-maximum
              (the post-slump slosh -- does the lever decay it?);
- `end`     -- last recorded `minDensity` / whether the run diverged and at
              what step (`~isfinite`, runner.py).

Arms are `epsF:epsB[:flags]` tokens -- XSPH fluid coeff, XSPH boundary coeff,
and an optional flag string drawn from `w` (Part 31 damped warm start),
`c` (Part 33 rest-density calibration), `g` (Part 30 free-surface gauge),
`s` (Part 33 skip the divergence solve -- plain IISPH, [I]). A
token with no third field uses the global `--warmStart` / `--calibrate` /
`--gauge` switches; a token WITH a third field sets exactly those flags
(empty `epsF:epsB:` forces all three off for that arm). So
`--arms 0:0:,0:0:wc` is baseline vs calibration+damped-warm-start. GPU
run-to-run nondeterminism is real on this failure (Part 29: 2/3 vs 1/3),
so `--runs` >= 2 and read the spread, not one number. A run whose |v|max
exceeds 1e4 is counted a blowup even when the runner's `~isfinite` check
did not trip (the Part 29/30 inf-soup has finite components, overflowing
norm).

Usage:
    python scripts/probe_dfsphReferenceColumnBatch.py            # 0:0 vs 0:0.1, 2 runs
    python scripts/probe_dfsphReferenceColumnBatch.py --runs 3 --arms 0:0:,0:0:wc
    python scripts/probe_dfsphReferenceColumnBatch.py --steps 800 --runs 1   # quick look
"""
import argparse

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args.add_argument('--steps', type=int, default=1500)
args.add_argument('--runs', type=int, default=2)
args.add_argument('--arms', type=str, default='0:0,0:0.1',
                 help='comma list of epsF:epsB[:flags] tokens; flags subset of '
                      'w(arm) c(alibrate) g(auge)')
args.add_argument('--soupVMax', type=float, default=1e4,
                 help='|v|max above this counts the run a blowup (inf-soup '
                      'guard for the overflowing-norm case)')
args.add_argument('--onsetThresh', type=float, default=0.50)
args.add_argument('--samples', type=str, default='300,600,900,1200,1500')
args.add_argument('--gauge', action='store_true',
                 help='also enable the Part 30 free-surface kappa^v gauge')
args.add_argument('--warmStart', action='store_true',
                 help='also enable the Part 31 damped warm start')
args.add_argument('--calibrate', action='store_true',
                 help='Part 33: normalise the fluid mass so the at-rest bulk '
                      'reads rho0 -- stops the step-1 IC-seed cold start')
args = args.parse_args()

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.dfsphReference as ref

SAMPLES = [int(s) for s in args.samples.split(',')]
ARMS = []
for tok in args.arms.split(','):
    parts = tok.split(':')
    ef, eb = float(parts[0]), float(parts[1])
    if len(parts) >= 3:
        fl = parts[2]
        warm, cal, gau, skip = 'w' in fl, 'c' in fl, 'g' in fl, 's' in fl
    else:
        warm, cal, gau, skip = args.warmStart, args.calibrate, args.gauge, False
    ARMS.append((ef, eb, warm, cal, gau, skip))


def _run_one(epsF, epsB, warm, cal, gau, skip):
    ref.FREE_SURFACE_GAUGE = gau
    ref.DAMPED_WARM_START = warm
    ref.SKIP_DIVERGENCE_SOLVE = skip
    ref.XSPH_FLUID_EPSILON = epsF
    ref.XSPH_BOUNDARY_EPSILON = epsB
    try:
        r = run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx,
                nSteps=args.steps, scheme='dfsphReference', quiet=True,
                plot=False, store=False, progress=False,
                integrationScheme='semiImplicitEuler',
                params={'calibrateRestDensity': cal})
    finally:
        ref.FREE_SURFACE_GAUGE = False
        ref.DAMPED_WARM_START = False
        ref.SKIP_DIVERGENCE_SOLVE = False
        ref.XSPH_FLUID_EPSILON = 0.0
        ref.XSPH_BOUNDARY_EPSILON = 0.0

    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    rho = [x.get('minDensity', float('nan')) for x in rows]
    vel = [x.get('maxVelocity', float('nan')) for x in rows]
    slope = [x.get('pressureSlopeRatio', float('nan')) for x in rows]
    # late-run mean of the fitted dp/dy ratio (1.0 = exact hydrostatic
    # gradient), over the surviving finite tail of the last quarter.
    tail = [s for s in slope[3 * len(slope) // 4:] if s == s]
    slopeLate = sum(tail) / len(tail) if tail else float('nan')

    onset = next((i + 1 for i, v in enumerate(rho) if v < args.onsetThresh), None)

    def at(series, step):
        idx = step - 1
        return series[idx] if 0 <= idx < len(series) else float('nan')

    velMax = max(vel) if vel else float('nan')
    return {
        'onset': onset,
        'diverged': bool(r.diverged) or (velMax > args.soupVMax),
        'nSteps': len(rows),
        'rhoAt': [at(rho, s) for s in SAMPLES],
        'velAt': [at(vel, s) for s in SAMPLES],
        'rhoMin': min(rho) if rho else float('nan'),
        'velMax': velMax,
        'rhoEnd': rho[-1] if rho else float('nan'),
        'slopeLate': slopeLate,
    }


hdr_s = ' '.join(f'{s:>7d}' for s in SAMPLES)
print(f'dfsphReference  hydrostaticColumn  nx={args.nx}  steps={args.steps}  '
      f'runs={args.runs}  onset<{args.onsetThresh:g}  '
      f'gauge={"on" if args.gauge else "off"}  '
      f'warmStart={"damped" if args.warmStart else "full"}  '
      f'calibrate={"on" if args.calibrate else "off"}')

for epsF, epsB, warm, cal, gau, skip in ARMS:
    flagstr = ''.join(c for c, on in (('w', warm), ('c', cal), ('g', gau),
                                      ('s', skip)) if on) or '-'
    isBase = epsF == 0 and epsB == 0 and not (warm or cal or gau or skip)
    print(f'\n=== arm epsF={epsF:g} epsB={epsB:g} flags={flagstr} '
          f'{"(baseline)" if isBase else ""} ===')
    agg = []
    for k in range(args.runs):
        res = _run_one(epsF, epsB, warm, cal, gau, skip)
        agg.append(res)
        rhoS = ' '.join(f'{v:>7.3f}' for v in res['rhoAt'])
        velS = ' '.join(f'{v:>7.3f}' for v in res['velAt'])
        div = f'DIVERGED@{res["nSteps"]}' if res['diverged'] else f'ok({res["nSteps"]})'
        on = res['onset'] if res['onset'] is not None else '   -'
        print(f'  run {k}:  onset={on!s:>5}  {div:>14}   rhoEnd={res["rhoEnd"]:.3f}  '
              f'rhoMin={res["rhoMin"]:.3f}  vMax={res["velMax"]:.3f}  '
              f'slopeLate={res["slopeLate"]:.3f}')
        print(f'            step:  {hdr_s}')
        print(f'          rhoMin:  {rhoS}')
        print(f'           |v|max: {velS}')
    onsets = [a['onset'] for a in agg if a['onset'] is not None]
    ndiv = sum(a['diverged'] for a in agg)
    print(f'  -- arm summary: onset {min(onsets) if onsets else "-"}..'
          f'{max(onsets) if onsets else "-"}  '
          f'blowups {ndiv}/{args.runs}  '
          f'rhoEnd [{min(a["rhoEnd"] for a in agg):.3f},{max(a["rhoEnd"] for a in agg):.3f}]  '
          f'rhoMin [{min(a["rhoMin"] for a in agg):.3f},{max(a["rhoMin"] for a in agg):.3f}]  '
          f'vMax [{min(a["velMax"] for a in agg):.3f},{max(a["velMax"] for a in agg):.3f}]  '
          f'slopeLate [{min(a["slopeLate"] for a in agg):.3f},{max(a["slopeLate"] for a in agg):.3f}]')
