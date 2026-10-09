#!/usr/bin/env python
"""omniIncompressible dam break (0.2 x 0.8 m column in a 1.5 x 0.9 m tank, the boundaries repo's omniSPH comparison) with boundary particles and with analytic walls,
against the stored omniSPH series (`~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz`: compiled omniSPH, 2093 particles, walls one spacing outside the block).

    python scripts/probe_omniAnalyticDambreak.py [--times 0.1 0.2 0.3 0.4 0.5 0.6] [--offsets 0.5 0.05] [--nx 100] [--video]

Per time: front displacement (x_max - x_max(0)), mean height above the wall plane, max |v|, density range. The whole point is the *statistics* (the flow is chaotic after the wall
impact at t > 0.5): omniSPH's own table (docs/dfsph-validation.md of the boundaries repo) agrees with its reimplementation within 1 % in the front and 0.4 % in the mean height.
Short runs in a loop: video only with --video (one run per configuration, the last time).
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

OMNI = os.path.expanduser('~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz')


def omniRows(times):
    d = np.load(OMNI)
    out = {}
    for key in ('omni', 'hydrostatic'):
        a = d[key]
        floor = 0.1 - 0.0089                                                # the wall plane one spacing below the block
        rows = []
        for t in times:
            i = min(np.searchsorted(a[:, 0], t - 1e-12), len(a) - 1)
            rows.append((a[i, 0], a[i, 3] - a[0, 3], a[i, 2] - floor, a[i, 1]))
        out[key] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--times', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument('--offsets', type=float, nargs='+', default=[0.5, 0.05], help='analytic wall plane moved outward by this many dx')
    ap.add_argument('--nx', type=int, default=100)
    ap.add_argument('--nh', type=float, default=2.57)
    ap.add_argument('--particles', action='store_true', help='also the boundary-particle run')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--massScale', type=float, default=0.9715, help="particle mass factor: omniSPH's V = pi r^2 is 0.9715 of the lattice cell (rest density of the lattice ~0.98, the density solve inactive); 1 = rho0 dx^2")
    a = ap.parse_args()

    bootstrap(precision='float32')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    case = getCase('dambreak')
    _ic = case.initialConditions

    def _ic2(ctx, system):
        if _ic is not None:
            _ic(ctx, system)
        system.state.masses = system.state.masses * a.massScale
    case.initialConditions = _ic2
    L, W = 0.9, 1.5
    dx = L / a.nx
    ref = omniRows(a.times)
    print('reference (compiled omniSPH, 2093 particles) / (boundaries repo DFSPH2D, hydrostatic wall): front displacement, mean height above the wall plane, vmax')
    for k in ('omni', 'hydrostatic'):
        print(f'  {k:11s} ' + '  '.join(f't={t:.1f}: {f:.3f}/{h:.3f}/{v:.2f}' for (t, f, h, v) in ref[k]))

    configs = ([('particles', 0.0)] if a.particles else []) + [('analytic', o) for o in a.offsets]
    for rep, off in configs:
        rows = []
        for T in a.times:
            params = {**case.params, 'wallRepresentation': rep, 'analyticWallOffset': off, 'W': W, 'fluidWidth': 23 * dx / W, 'fillRatio': 91 * dx / L, 'wallBC': 'freeSlip'}
            last = (T == a.times[-1])
            spec = CaseSpec(caseName=f'omniDam-{rep}-{off}', scheme='omniIncompressible', params=params).merged(**case.defaults).merged(
                scheme='omniIncompressible', L=L, nx=a.nx, n_h=a.nh, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric',
                cflFactor=1.0, dt=1.0e-3, minDt=1.0e-4, maxDt=1.0e-3, tLimit=T, adaptiveDt=True,
                plot=a.video and last, video=a.video and last, show=False, store=False, progress=a.video and last, quiet=True, plotInterval=10,
                velocityAlarmPlotInterval=1, stallProgress=1e-3)
            res = run(case, spec)
            s = res.state.state
            f = s.kinds == 0
            pos, v, rho = s.positions[f], s.velocities[f], s.densities[f]
            if not rows:
                x0 = None
            interior = res.ctx.scratch['interiorDomain']
            yFloor = float(interior.min[1]) - off * dx
            rows.append((float(res.state.t), float(pos[:, 0].max()), float(pos[:, 1].mean() - yFloor), float(v.norm(dim=1).max()), float(rho.min()), float(rho.max()), int(f.sum()), res.diverged))
            print(f'   [{rep} offset {off}] t={T:.1f} done (n={int(f.sum())})', flush=True)
        x0 = rows[0][1]                                                      # front at the first time is not t = 0: take the initial column edge from the interior domain
        interior = res.ctx.scratch['interiorDomain']
        x0 = float(interior.min[0]) + 23 * dx
        print(f'{rep} offset {off} (n_h {a.nh}, nx {a.nx}):  ' + '  '.join(f't={t:.2f}: {x - x0:.3f}/{h:.3f}/{vm:.2f} rho[{r0:.2f},{r1:.2f}]{"  DIVERGED" if dv else ""}' for (t, x, h, vm, r0, r1, n, dv) in rows))


if __name__ == '__main__':
    main()
