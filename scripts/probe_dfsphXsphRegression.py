"""Does the `divergenceFree.XSPH_SCALE` free-surface fix regress the periodic cases the
`divergenceFree` scheme is otherwise clean on? Runs `tgv` (analytic KE decay)
at a couple of scales and prints the final-vs-analytic KE ratio.

    python scripts/probe_dfsphXsphRegression.py
"""
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', type=int, default=400)
argp.add_argument('--scales', default='0,0.25,1.0')
args = argp.parse_args()

from warpSPH.runner import run
from warpSPH.cases import tgv as tgv_case
from warpSPH.schemes import divergenceFree as D


def run_one(scale):
    D.XSPH_SCALE = scale
    r = run(tgv_case.tgvCase, nx=args.nx, scheme='divergenceFree', quiet=True,
            progress=False, plot=False, store=False, nSteps=args.steps)
    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    last = rows[-1] if rows else {}
    return r.diverged, last, list(last.keys())


for i, s in enumerate(args.scales.split(',')):
    diverged, last, keys = run_one(float(s))
    if i == 0:
        print('trajectory keys:', keys)
    print(f'\nXSPH_SCALE={float(s):g}  diverged={diverged}')
    for k in ('t', 'kineticEnergy', 'kineticEnergyExact', 'keRatio',
              'maxVelocity', 'velocityError', 'l2Velocity', 'decayError'):
        if k in last:
            print(f'  {k:22} {last[k]:.6g}')
