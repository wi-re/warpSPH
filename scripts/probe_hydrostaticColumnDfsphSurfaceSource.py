"""Sweep `dfsph.SURFACE_SOURCE` on `hydrostaticColumn --scheme divergenceFree`.

The constant-density solve's `(1 - rho/rho0)` source reads a large positive
value in the kernel-truncated free-surface skin (`rho ~ 0.69 rho0` top row),
so the solve perpetually accelerates the skin outward chasing an unreachable
rho0 -- the residual `|v|`. This runs the fixed-dt calibrated column for each
`SURFACE_SOURCE` mode and prints the density / velocity / pressure FOMs.

    python scripts/probe_hydrostaticColumnDfsphSurfaceSource.py
    python scripts/probe_hydrostaticColumnDfsphSurfaceSource.py --nx 128 --steps 300
    python scripts/probe_hydrostaticColumnDfsphSurfaceSource.py --modes full,clamp
"""
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', type=int, default=250)
argp.add_argument('--modes', default='full,clamp,mask')
argp.add_argument('--no-calibrate', dest='calibrate', action='store_false', default=True)
args = argp.parse_args()

from warpSPH.runner import run
from warpSPH.cases import hydrostaticColumn
from warpSPH.schemes import builder, dfsph as dfsph_mod

FOMS = ['maxVelocity', 'kineticEnergy', 'densityP05', 'embeddedMinDensity',
        'minDensity', 'densityStd', 'dispMax', 'pressureSlopeRatio',
        'pressureResidual']


def run_one(mode):
    dfsph_mod.SURFACE_SOURCE = mode
    r = run(hydrostaticColumn.hydrostaticColumnCase,
            nx=args.nx, scheme='divergenceFree', kernel='Wendland2',
            integrationScheme='semiImplicitEuler', quiet=True, progress=False,
            plot=False, store=False, verbose=False,
            params={'calibrateRestDensity': args.calibrate},
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False,
            nSteps=args.steps)
    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    last = rows[-1] if rows else {}
    peak = {k: max((x.get(k, float('nan')) for x in rows), default=float('nan'))
            for k in ('maxVelocity', 'kineticEnergy', 'densityStd')}
    return r.diverged, last, peak


print(f'hydrostaticColumn nx={args.nx}  {args.steps} steps  '
      f'calibrate={args.calibrate}  (fixed dt=1e-3)\n')
hdr = f'{"mode":>8} {"diverged":>9} ' + ' '.join(f'{k[:13]:>13}' for k in FOMS) \
      + f' {"vmaxPeak":>10} {"KEpeak":>10}'
print(hdr)
print('-' * len(hdr))
for mode in args.modes.split(','):
    mode = mode.strip()
    try:
        diverged, last, peak = run_one(mode)
    except Exception as e:
        print(f'{mode:>8}  ERROR: {e}')
        continue
    cells = ' '.join(f'{last.get(k, float("nan")):>13.5g}' for k in FOMS)
    print(f'{mode:>8} {str(diverged):>9} {cells} '
          f'{peak["maxVelocity"]:>10.4g} {peak["kineticEnergy"]:>10.4g}')
