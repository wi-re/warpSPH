"""Spatial anatomy of the near-surface error on `hydrostaticColumn --scheme
divergenceFree` (the DFSPH `divergenceFree_step` path).

The scalar FOMs say the column body is essentially perfect
(`embeddedMinDensity` ~1.000) while `densityP05` ~0.94 and `minDensity`
~0.69 -- i.e. all the error lives in the free-surface skin and the convex
corners. This dumps the per-particle picture at a few step counts, binned by
depth below the free surface and by distance to the nearest wall, so we can
see *which* rows read low and whether the velocity/pressure noise tracks
them.

    python scripts/probe_hydrostaticColumnDfsphSurface.py                 # nx=64, steps 0,5,20,60,120
    python scripts/probe_hydrostaticColumnDfsphSurface.py --nx 128 --steps 0,10,50,200
"""
import argparse

import numpy as np

argp = argparse.ArgumentParser()
argp.add_argument('--nx', type=int, default=64)
argp.add_argument('--steps', default='0,5,20,60,120')
argp.add_argument('--calibrate', action='store_true', default=True)
argp.add_argument('--no-calibrate', dest='calibrate', action='store_false')
args = argp.parse_args()

want = sorted(int(s) for s in args.steps.split(','))

from warpSPH.runner import run
from warpSPH.cases import hydrostaticColumn
from warpSPH.schemes import builder

bundle = builder.buildScheme('divergenceFree')
step = bundle.stepFunction
mod = __import__(step.__module__, fromlist=['x'])


def fmt_rows(title, labels, cols):
    print(f'  {title}')
    hdr = '    ' + f'{"":>16}' + ''.join(f'{c:>12}' for c in cols[0])
    print(hdr)
    for lab, row in zip(labels, cols[1]):
        print('    ' + f'{lab:>16}' + ''.join(
            (f'{v:>12.4g}' if isinstance(v, float) else f'{v:>12}') for v in row))


def report(st, n):
    k = st.kinds.detach().cpu().numpy()
    pos = st.positions.detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    vel = st.velocities.detach().cpu().numpy()
    p = (st.pressures.detach().cpu().numpy() if st.pressures is not None
         else np.zeros_like(rho))
    f = k == 0
    fx, fy = pos[f, 0], pos[f, 1]
    frho, fp = rho[f], p[f]
    fv = np.linalg.norm(vel[f], axis=-1)

    surf = np.quantile(fy, 0.99)
    dx = 1.0 / args.nx
    depth = (surf - fy) / dx                      # rows below surface, in dx
    # distance to nearest vertical wall (box is [-0.5, 0.5] in x); floor at y=-0.5
    wall_x = 0.5 - np.abs(fx)
    wall_y = fy - (-0.5)
    wall = np.minimum(wall_x, wall_y) / dx

    print(f'\n=== step {n}  (nfluid={f.sum()}, surfaceY99={surf:.4f}) ===')
    print(f'  fluid rho:  mean {frho.mean():.4f}  median {np.median(frho):.4f}  '
          f'p05 {np.quantile(frho,0.05):.4f}  min {frho.min():.4f}')
    print(f'  fluid |v|:  mean {fv.mean():.4e}  p95 {np.quantile(fv,0.95):.4e}  '
          f'max {fv.max():.4e}')

    # bin by depth below surface
    edges = [(-1, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 4), (4, 8), (8, 1e9)]
    labs = ['skin(<0.5)', '0.5-1.5', '1.5-2.5', '2.5-4', '4-8', 'bulk(>8)']
    rows = []
    for lo, hi in edges:
        m = (depth >= lo) & (depth < hi)
        if not m.any():
            rows.append(['-', '-', '-', '-', '-'])
            continue
        rows.append([int(m.sum()), float(frho[m].mean()), float(frho[m].min()),
                     float(fv[m].mean()), float(fv[m].max())])
    fmt_rows('by depth below surface [dx]',
             labs, (['n', 'rho.mean', 'rho.min', '|v|.mean', '|v|.max'], rows))

    # corners: near a vertical wall AND within ~2 dx of the surface
    corner = (wall_x / dx < 2.0) & (depth < 2.0)
    if corner.any():
        print(f'  convex corners (wall_x<2dx & depth<2dx): n={corner.sum()}  '
              f'rho.mean {frho[corner].mean():.4f}  rho.min {frho[corner].min():.4f}  '
              f'|v|.max {fv[corner].max():.4e}')

    # bin by distance to nearest wall (bulk depth only, to isolate the wall)
    deep = depth > 6
    edges_w = [(-1, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 4), (4, 1e9)]
    labs_w = ['<0.5', '0.5-1.5', '1.5-2.5', '2.5-4', '>4']
    rows = []
    for lo, hi in edges_w:
        m = deep & (wall >= lo) & (wall < hi)
        if not m.any():
            rows.append(['-', '-', '-', '-', '-'])
            continue
        rows.append([int(m.sum()), float(frho[m].mean()), float(frho[m].min()),
                     float(fv[m].mean()), float(fp[m].mean())])
    fmt_rows('by dist to nearest wall [dx], depth>6dx only',
             labs_w, (['n', 'rho.mean', 'rho.min', '|v|.mean', 'p.mean'], rows))


def spy(system, dt, config, schemeConfig, verbose=False):
    n = getattr(spy, 'n', 0)
    r = step(system, dt, config, schemeConfig, verbose)
    if n in want:
        report(system.state, n)
    spy.n = n + 1
    return r


setattr(mod, step.__name__, spy)
try:
    run(hydrostaticColumn.hydrostaticColumnCase,
        nx=args.nx, scheme='divergenceFree', kernel='Wendland2',
        integrationScheme='semiImplicitEuler', quiet=True, progress=False,
        plot=False, store=False, verbose=False,
        params={'calibrateRestDensity': args.calibrate},
        dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False,
        nSteps=max(want) + 2)
finally:
    setattr(mod, step.__name__, step)
