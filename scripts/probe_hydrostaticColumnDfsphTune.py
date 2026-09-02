"""Tune the `dfsph` inner-solve knobs on `hydrostaticColumn --scheme
divergenceFree` (fixed dt, calibrated rest density). Each named config sets
`dfsph` module constants, runs the column, and prints the FOMs -- the residual
`|v|` (a persistent free-surface limit cycle) is the target.

    python scripts/probe_hydrostaticColumnDfsphTune.py
    python scripts/probe_hydrostaticColumnDfsphTune.py --nx 128 --steps 400
"""
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', type=int, default=250)
argp.add_argument('--no-calibrate', dest='calibrate', action='store_false', default=True)
args = argp.parse_args()

from warpSPH.runner import run
from warpSPH.cases import hydrostaticColumn
from warpSPH.schemes import dfsph as D
from warpSPH.schemes import omniIncompressible as O

FOMS = ['maxVelocity', 'kineticEnergy', 'densityP05', 'embeddedMinDensity',
        'minDensity', 'dispMax', 'pressureSlopeRatio', 'pressureResidual']

DEFAULTS = dict(
    SURFACE_SOURCE=D.SURFACE_SOURCE,
    DIV_MIN_ITERS=D.DIV_MIN_ITERS, DIV_MAX_ITERS=D.DIV_MAX_ITERS, DIV_TOL=D.DIV_TOL,
    RHO_MIN_ITERS=D.RHO_MIN_ITERS, RHO_MAX_ITERS=D.RHO_MAX_ITERS, RHO_TOL=D.RHO_TOL,
)
O_DEFAULT_OMEGA = O.OMEGA

CONFIGS = {
    'baseline(full,2/3)':   dict(SURFACE_SOURCE='full'),
    'clamp,2/3':            dict(SURFACE_SOURCE='clamp'),
    'clamp,div6':           dict(SURFACE_SOURCE='clamp', DIV_MIN_ITERS=6),
    'clamp,div12':          dict(SURFACE_SOURCE='clamp', DIV_MIN_ITERS=12),
    'clamp,rho10':          dict(SURFACE_SOURCE='clamp', RHO_MIN_ITERS=10),
    'clamp,div6,rho10':     dict(SURFACE_SOURCE='clamp', DIV_MIN_ITERS=6, RHO_MIN_ITERS=10),
    'clamp,divtol1e-4':     dict(SURFACE_SOURCE='clamp', DIV_TOL=1e-4, DIV_MAX_ITERS=64),
    'clamp,rhotol1e-4':     dict(SURFACE_SOURCE='clamp', RHO_TOL=1e-4),
}


def run_one(overrides):
    for k, v in DEFAULTS.items():
        setattr(D, k, overrides.get(k, v))
    O.OMEGA = overrides.get('OMEGA', O_DEFAULT_OMEGA)
    r = run(hydrostaticColumn.hydrostaticColumnCase,
            nx=args.nx, scheme='divergenceFree', kernel='Wendland2',
            integrationScheme='semiImplicitEuler', quiet=True, progress=False,
            plot=False, store=False, verbose=False,
            params={'calibrateRestDensity': args.calibrate},
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False, nSteps=args.steps)
    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    last = rows[-1] if rows else {}
    vpeak = max((x.get('maxVelocity', float('nan')) for x in rows), default=float('nan'))
    return r.diverged, last, vpeak


hdr = f'{"config":>20} {"div?":>5} ' + ' '.join(f'{k[:12]:>12}' for k in FOMS) + f' {"vmaxPeak":>10}'
print(f'hydrostaticColumn nx={args.nx}  {args.steps} steps  calibrate={args.calibrate}\n')
print(hdr)
print('-' * len(hdr))
for name, ov in CONFIGS.items():
    try:
        diverged, last, vpeak = run_one(ov)
    except Exception as e:
        print(f'{name:>20}  ERROR: {type(e).__name__}: {e}')
        continue
    cells = ' '.join(f'{last.get(k, float("nan")):>12.5g}' for k in FOMS)
    print(f'{name:>20} {str(diverged):>5} {cells} {vpeak:>10.4g}')
