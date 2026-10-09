#!/usr/bin/env python
"""Incompressible dam break (0.2 x 0.8 m column in a 1.5 x 0.9 m tank, the boundaries repo's omniSPH comparison) with analytic walls on the calibrated lattice (`calibratedLattice='auto'`), for
`omniIncompressible` and `divergenceFree`, with boundary particles for reference, against the stored compiled omniSPH series (`~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz`).

    python scripts/probe_incompressibleAnalyticDambreak.py --scheme dfsph --walls analytic [--tLimit 1.0] [--nh 2.57] [--xsph 1e-4 --boundaryFriction 5e-3] [--out DIR]

One run, video on (vispy), a flushed row per 0.05 s of the front displacement (x_max - x_max(0)), the mean height above the floor plane and max |v|, then the omniSPH rows at the same times.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

OMNI = os.path.expanduser('~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz')
SCHEMES = {'omni': 'omniIncompressible', 'dfsph': 'divergenceFree'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', choices=sorted(SCHEMES), default='dfsph')
    ap.add_argument('--walls', choices=('analytic', 'particles'), default='analytic')
    ap.add_argument('--tLimit', type=float, default=1.0)
    ap.add_argument('--nx', type=int, default=100)
    ap.add_argument('--nh', type=float, default=2.57)
    ap.add_argument('--xsph', type=float, default=0.0)
    ap.add_argument('--boundaryFriction', type=float, default=0.0)
    ap.add_argument('--wallPressure', default='hydrostatic')
    ap.add_argument('--projection', choices=('jacobi', 'compact'), default='jacobi')
    ap.add_argument('--noDensitySolve', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    a = ap.parse_args()

    bootstrap(precision='float32')
    import warnings
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    case = getCase('dambreak')
    L, W = 0.9, 1.5
    dx = L / a.nx
    rows = []
    _ps = case.postStep

    def _ps2(ctx, state, step):
        if _ps is not None:
            _ps(ctx, state, step)
        s = state.state
        f = s.kinds == 0
        t = float(state.t)
        if len(rows) == 0 or t >= rows[-1][0] + 0.05 - 1e-9:
            pos, v, rho = s.positions[f], s.velocities[f], s.densities[f]
            interior = ctx.scratch['interiorDomain']
            rows.append((t, float(pos[:, 0].max()) - (float(interior.min[0]) + 23 * dx), float(pos[:, 1].mean() - float(interior.min[1])), float(v.norm(dim=1).max()), float(rho.min()), float(rho.max())))
            print('   t={:.2f}: front {:.3f}  mean height {:.3f}  max|v| {:.2f}  rho [{:.3f}, {:.3f}]'.format(*rows[-1]), flush=True)
    case.postStep = _ps2
    params = {**case.params, 'wallRepresentation': a.walls, 'W': W, 'fluidWidth': 23 * dx / W, 'fillRatio': 91 * dx / L, 'wallBC': 'freeSlip',
              'xsphCoefficient': a.xsph, 'boundaryFriction': a.boundaryFriction, 'analyticWallPressure': a.wallPressure, 'projection': a.projection, 'densitySolve': not a.noDensitySolve}
    scheme = SCHEMES[a.scheme]
    spec = CaseSpec(caseName=f'dam-{a.scheme}-{a.walls}', scheme=scheme, params=params).merged(**case.defaults).merged(
        scheme=scheme, L=L, nx=a.nx, n_h=a.nh, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=1.0, dt=1e-3, minDt=1e-4, maxDt=1e-3,
        tLimit=a.tLimit, adaptiveDt=True, plot=a.video, video=a.video, show=False, store=False, progress=True, quiet=True, plotInterval=10, velocityAlarmPlotInterval=1, stallProgress=1e-3,
        **({'exportRoot': a.out} if a.out else {}))
    res = run(case, spec)
    print('finished t={:.3f} diverged={} alarms={}'.format(float(res.state.t), res.diverged, len(res.velocityAlarms)))
    d = np.load(OMNI)['omni']
    floor = 0.1 - 0.0089
    print('compiled omniSPH (front / mean height above its wall plane / max|v|):')
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        if t > a.tLimit + 1e-9:
            break
        i = min(np.searchsorted(d[:, 0], t - 1e-12), len(d) - 1)
        k = int(np.argmin([abs(r[0] - t) for r in rows]))
        print(f'  t={t:.1f}  omniSPH {d[i, 3] - d[0, 3]:.3f}/{d[i, 2] - floor:.3f}/{d[i, 1]:.2f}    this run (t={rows[k][0]:.2f}) {rows[k][1]:.3f}/{rows[k][2]:.3f}/{rows[k][3]:.2f}')


if __name__ == '__main__':
    main()
