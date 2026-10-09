#!/usr/bin/env python
"""The VD+PS particle shift of `IncompressibleSystem.finalize` (divergenceFree, no gravity, no density correction in the step) in a closed tank, analytic walls against boundary particles: a smooth swirl
(stream function psi = A sin(pi x'/W) sin(pi y'/H), zero normal velocity at the side walls and the floor, the top a free surface) decays only through numerical dissipation; kinetic energy, density
range and wall penetration over time.

    python scripts/probe_vdpsClosedTank.py [--tLimit 1.0] [--nx 100] [--walls analytic particles] [--out DIR]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

ROLL = os.path.join(REPO, 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tLimit', type=float, default=1.0)
    ap.add_argument('--nx', type=int, default=100)
    ap.add_argument('--amp', type=float, default=0.15, help='peak swirl velocity [m/s]')
    ap.add_argument('--walls', nargs='+', default=['analytic', 'particles'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    a = ap.parse_args()
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    for rep in a.walls:
        case = getCase('sloshingTank')
        ic, ps = case.initialConditions, case.postStep
        rows = []

        def ic2(ctx, system, ic=ic):
            ic(ctx, system)
            st = system.state
            f = st.kinds == 0
            x = st.positions[f]
            lo = x.min(0).values
            W = float(x[:, 0].max() - lo[0])
            H = float(x[:, 1].max() - lo[1])
            xp, yp = x[:, 0] - lo[0], x[:, 1] - lo[1]
            kx, ky = np.pi / W, np.pi / H
            u = a.amp * torch.sin(kx * xp) * torch.cos(ky * yp) / (1.0)
            v = -a.amp * (kx / ky) * torch.cos(kx * xp) * torch.sin(ky * yp)
            vel = torch.zeros_like(st.velocities)
            vel[f, 0], vel[f, 1] = u, v
            st.velocities = vel

        def ps2(ctx, state, step, ps=ps, rows=rows):
            if ps is not None:
                ps(ctx, state, step)
            s = state.state
            f = s.kinds == 0
            t = float(state.t)
            if not rows or t >= rows[-1][0] + 0.1 - 1e-9:
                ke = float(0.5 * (s.masses[f] * (s.velocities[f] ** 2).sum(1)).sum())
                rows.append((t, ke, float(s.densities[f].min()), float(s.densities[f].max())))
                print(f'   [{rep}] t={t:.2f}  KE {ke:.4e}  rho [{rows[-1][2]:.3f}, {rows[-1][3]:.3f}]', flush=True)
        case.initialConditions, case.postStep = ic2, ps2
        params = {**case.params, 'wallRepresentation': rep, 'rollDataFile': ROLL, 'rollStartTime': 100.0, 'gravityMagnitude': 0.0}
        spec = CaseSpec(caseName=f'vdps-{rep}', scheme='divergenceFree', params=params).merged(**case.defaults).merged(
            scheme='divergenceFree', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=a.nx, n_h=2.57,
            tLimit=a.tLimit, plot=a.video, video=a.video, show=False, store=False, progress=True, quiet=True, plotInterval=20, velocityAlarmPlotInterval=1, stallProgress=1e-3,
            **({'exportRoot': os.path.join(a.out, rep)} if a.out else {}))
        res = run(case, spec)
        ke0 = rows[0][1]
        print(f'RESULT {rep:9s}: KE/KE0 at ' + '  '.join(f't={r[0]:.1f}: {r[1] / ke0:.3f}' for r in rows[::2]) + f'   rho max {max(r[3] for r in rows):.3f}  diverged {res.diverged}', flush=True)


if __name__ == '__main__':
    main()
