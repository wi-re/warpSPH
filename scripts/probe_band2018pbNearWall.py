"""Look at what `band2018pb` actually does to the near-wall / free-surface
fluid distribution on `hydrostaticColumn` -- the scalar FOMs
(`maxVelocity`, `embeddedMinDensity`) hid an over-compressed, wall-penetrating
state at nx=64 (a frozen bad lattice reads as "quiescent"). This reports the
physical picture: bulk density, the compressed bottom layer, fluid sinking
into the floor band, and free-surface spray -- side by side for
`band2018pb` / `omniIncompressible` / `iisph`.

    python scripts/probe_band2018pbNearWall.py                 # nx=64, step 150
    python scripts/probe_band2018pbNearWall.py --nx 128 --step 200
    python scripts/probe_band2018pbNearWall.py --schemes band2018pb
    python scripts/probe_band2018pbNearWall.py --tik 0.0,0.05,0.1   # sweep band2018pb DIAG_TIKHONOV
"""
import argparse

import numpy as np

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--step', type=int, default=150)
argp.add_argument('--schemes', default='band2018pb,omniIncompressible,iisph')
argp.add_argument('--tik', default=None,
                  help='comma list of band2018pb.DIAG_TIKHONOV to sweep')
args = argp.parse_args()

from warpSPH.runner import run
from warpSPH.cases import hydrostaticColumn
from warpSPH.schemes import builder


def probe_once(scheme, tik=None):
    bundle = builder.buildScheme(scheme)
    step = bundle.stepFunction
    mod = __import__(step.__module__, fromlist=['x'])
    if tik is not None and scheme == 'band2018pb':
        mod.DIAG_TIKHONOV = tik
    out = {}

    def spy(system, dt, config, schemeConfig, verbose=False):
        n = getattr(spy, 'n', 0)
        spy.n = n + 1
        r = step(system, dt, config, schemeConfig, verbose)
        if n == args.step:
            st = system.state
            k = st.kinds.cpu().numpy()
            pos = st.positions.detach().cpu().numpy()
            rho = st.densities.detach().cpu().numpy()
            f = k == 0
            bd = k == 1
            fy = pos[f, 1]
            by = pos[bd, 1]
            floortop = by[by < -0.45].max() if (by < -0.45).any() else by.min()
            deep = np.argsort(fy)[:12]
            out.update(
                bulk_rho_med=float(np.median(rho[f])),
                bulk_rho_p95=float(np.percentile(rho[f], 95)),
                deep_layer_rho=float(np.mean(rho[f][deep])),
                sink_below_floortop=float(floortop - fy.min())
                if fy.min() < floortop else 0.0,
                fluid_ymin=float(fy.min()), fluid_ymax=float(fy.max()),
                n_spray_rho_lt_0p6=int((rho[f] < 0.6).sum()),
                n_fluid=int(f.sum()))
        return r

    setattr(mod, step.__name__, spy)
    try:
        rr = run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx,
                 nSteps=args.step + 10, scheme=scheme, quiet=True, plot=False,
                 store=False, progress=False,
                 integrationScheme='semiImplicitEuler')
        rows = [x for x in rr.trajectory if x.get('step', -1) >= 0]
        t = rows[-1] if rows else {}
        out['diverged'] = rr.diverged
        out['vmax_last'] = t.get('maxVelocity', float('nan'))
    finally:
        setattr(mod, step.__name__, step)
    return out


print(f'hydrostaticColumn nx={args.nx}  state at step {args.step}')
print('=' * 92)
hdr = ('bulk_rho_med', 'bulk_rho_p95', 'deep_layer_rho', 'sink_below_floortop',
       'n_spray_rho_lt_0p6', 'fluid_ymax', 'vmax_last', 'diverged')
if args.tik is not None:
    for tik in [float(x) for x in args.tik.split(',')]:
        d = probe_once('band2018pb', tik=tik)
        print(f'\nband2018pb  DIAG_TIKHONOV={tik:g}')
        for h in hdr:
            print(f'    {h:20s} {d.get(h)}')
else:
    for s in args.schemes.split(','):
        d = probe_once(s.strip())
        print(f'\n{s.strip()}')
        for h in hdr:
            print(f'    {h:20s} {d.get(h)}')
