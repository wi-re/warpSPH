"""Run `hydrostaticColumn` through the standard divergence-free (DFSPH) path
-- `IncompressibleSPHScheme.divergenceFree`, solver `schemes/dfsph.py::dfsph_step`
-- at nx=64.

This is the failing baseline documented in `cases/hydrostaticColumn.py` /
DFSPH_FINDINGS.md Part 23: the quiescent wall-bounded free-surface column
should stay at rest (`v = 0`, `p(y) = rho0 g (y_surf - y)`), but the DF
projection cannot balance a uniform body force and the constant-density
fall-and-push-back cycle is unstable here. At nx=64 the run goes non-finite
around step ~19 (`t ~ 0.05`).

    python scripts/run_hydrostaticColumnDfsph.py                 # headless, prints FOM
    python scripts/run_hydrostaticColumnDfsph.py --steps 40      # cap the step count
    python scripts/run_hydrostaticColumnDfsph.py --video         # also render an mp4
    python scripts/run_hydrostaticColumnDfsph.py --nx 128        # other resolution

Equivalent CLI one-liner:
    python -m warpSPHRun hydrostaticColumn --nx 64 --scheme divergenceFree \
        --no-plot --no-store
"""
import argparse

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--nx', type=int, default=64,
               help='particles across the domain (default: 64)')
p.add_argument('--steps', type=int, default=None,
               help='stop after this many steps instead of at tLimit')
p.add_argument('--tLimit', type=float, default=1.0,
               help='simulated stop time when --steps is not given (default: 1.0)')
p.add_argument('--video', action='store_true',
               help='render frames and an mp4 (writes under exports/)')
p.add_argument('--store', action='store_true',
               help='write the particle trajectory to disk')
args = p.parse_args()

kwargs = dict(
    nx=args.nx,
    scheme='divergenceFree',
    kernel='Wendland2',
    integrationScheme='semiImplicitEuler',
    quiet=True,
    progress=False,
    plot=args.video,
    video=args.video,
    plotBackend='matplotlib',
    plotInterval=4,
    store=args.store,
)
if args.steps is not None:
    kwargs['nSteps'] = args.steps
else:
    kwargs['tLimit'] = args.tLimit

r = run(hydrostaticColumn.hydrostaticColumnCase, **kwargs)

rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
vmax = [x.get('maxVelocity', float('nan')) for x in rows]
vmax = [v for v in vmax if v == v]
tlast = rows[-1].get('t', float('nan')) if rows else float('nan')

print(f'nx={args.nx}  scheme=divergenceFree (dfsph_step)')
print(f'ran {len(rows)} steps  t={tlast:.5g}  diverged={r.diverged}')
if vmax:
    print(f'|v|max: peak {max(vmax):.5g}  last {vmax[-1]:.5g}  '
          f'(quiescent column should stay ~0)')
if r.diverged:
    print('=> DIVERGED, as expected for this baseline (DFSPH_FINDINGS.md Part 23).')
if getattr(r, 'videoPath', None):
    print(f'video: {r.videoPath}')
