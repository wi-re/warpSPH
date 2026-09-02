"""Sweep `dfsph.XSPH_SCALE` on `hydrostaticColumn --scheme divergenceFree`.

The residual `|v|` is an undamped inviscid free-surface limit cycle; a light
post-solve XSPH velocity smoother is the only available sink. This runs the
fixed-dt calibrated column at a few scales and prints the FOMs, including
whether the cycle actually *decays* (|v| at 1/2 vs full run length).

    python scripts/probe_hydrostaticColumnDfsphXsph.py
    python scripts/probe_hydrostaticColumnDfsphXsph.py --nx 128 --steps 600 --scales 0,0.5,1,2
"""
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', type=int, default=500)
argp.add_argument('--scales', default='0,0.25,0.5,1.0,2.0')
argp.add_argument('--no-calibrate', dest='calibrate', action='store_false', default=True)
args = argp.parse_args()

from warpSPH.runner import run
from warpSPH.cases import hydrostaticColumn
from warpSPH.schemes import dfsph as D

FOMS = ['maxVelocity', 'kineticEnergy', 'densityP05', 'embeddedMinDensity',
        'minDensity', 'dispMax', 'pressureSlopeRatio', 'pressureResidual']


def run_one(scale):
    D.XSPH_SCALE = scale
    r = run(hydrostaticColumn.hydrostaticColumnCase,
            nx=args.nx, scheme='divergenceFree', kernel='Wendland2',
            integrationScheme='semiImplicitEuler', quiet=True, progress=False,
            plot=False, store=False, verbose=False,
            params={'calibrateRestDensity': args.calibrate},
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False, nSteps=args.steps)
    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    last = rows[-1] if rows else {}
    half = rows[len(rows) // 2] if rows else {}
    vpeak = max((x.get('maxVelocity', float('nan')) for x in rows), default=float('nan'))
    return r.diverged, last, half.get('maxVelocity', float('nan')), vpeak


hdr = (f'{"XSPH":>6} {"div?":>5} ' + ' '.join(f'{k[:12]:>12}' for k in FOMS)
       + f' {"|v|@half":>9} {"vmaxPeak":>9}')
print(f'hydrostaticColumn nx={args.nx}  {args.steps} steps  calibrate={args.calibrate}\n')
print(hdr)
print('-' * len(hdr))
for s in args.scales.split(','):
    s = float(s)
    try:
        diverged, last, vhalf, vpeak = run_one(s)
    except Exception as e:
        print(f'{s:>6}  ERROR: {type(e).__name__}: {e}')
        continue
    cells = ' '.join(f'{last.get(k, float("nan")):>12.5g}' for k in FOMS)
    print(f'{s:>6} {str(diverged):>5} {cells} {vhalf:>9.4g} {vpeak:>9.4g}')
