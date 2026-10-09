#!/usr/bin/env python
"""English et al. 2022 s.4.1 still water over a sharp wedge (2.4 x 1.2 m tank, 0.5 m of water, a 0.24 m wedge at the bottom centre) with `omniIncompressible`: analytic walls
(calibrated lattice for the tank, `boundary/calibration.py`; the wedge is an analytic triangle) against boundary particles. Still water: the exact answer is p = rho0 g (H - y), kinetic energy ~ 0.

    python scripts/probe_omniEnglishWedge.py [--dp 0.02] [--tLimit 4] [--wedge/--no-wedge] [--rep analytic particles] [--video]

Prints, at t = 1, 2, 3, 4 s: max |v|, kinetic energy, p / hydrostatic in the bulk, near the walls and within 3 dx of the wedge, density range.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

TANK_W, TANK_H, WATER_H, WEDGE_H, G = 2.4, 1.2, 0.5, 0.24, 9.81
WEDGE_MAX_EXTENT = 0.30 * (WEDGE_H / 0.248)


def segDist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dp', type=float, default=0.02)
    ap.add_argument('--tLimit', type=float, default=4.0)
    ap.add_argument('--scheme', choices=('omni', 'dfsph'), default='omni')
    ap.add_argument('--wedge', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--rep', nargs='+', default=['analytic'], choices=('analytic', 'particles'))
    ap.add_argument('--nh', type=float, default=2.57)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--massScale', type=float, default=1.0, help='extra factor on the particle mass (particle walls: 0.9715 = omniSPH V = pi r^2)')
    a = ap.parse_args()
    SCHEME = {'omni': 'omniIncompressible', 'dfsph': 'divergenceFree'}[a.scheme]

    bootstrap(precision='float32')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    nx = int(round(TANK_H / a.dp))
    dx = TANK_H / nx
    for rep in a.rep:
        case = getCase('dambreak')
        _ic = case.initialConditions

        def _ic2(ctx, system, _ic=_ic, rep=rep):
            if _ic is not None:
                _ic(ctx, system)
            if rep == 'particles' and a.massScale != 1.0:
                system.state.masses = system.state.masses * a.massScale
        case.initialConditions = _ic2
        params = {**case.params, **dict(W=TANK_W, fillRatio=WATER_H / TANK_H, fluidWidth=1.0, gravityMagnitude=G, wallBC='freeSlip', wallRepresentation=rep, shifting='off', pressureProbeHeights=[])}
        if a.wedge:
            params.update(obstacleActive=True, obstacleType='equilateralBottom', offsetX=0.0, aoa=0.0, maxExtent=WEDGE_MAX_EXTENT, obstacleRepresentation=rep if rep == 'analytic' else 'particles')
        times = [1.0, 2.0, 3.0, 4.0]
        times = [t for t in times if t <= a.tLimit + 1e-9] or [a.tLimit]
        print(f'-- {rep} walls, wedge {a.wedge}, dp {dx:.4f}, n_h {a.nh}')
        for T in times:
            last = T == times[-1]
            spec = CaseSpec(caseName=f'omniWedge-{rep}', scheme=SCHEME, params=params).merged(**case.defaults).merged(
                scheme=SCHEME, L=TANK_H, nx=nx, n_h=a.nh, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=1.0,
                dt=2e-3, minDt=1e-4, maxDt=2e-3, tLimit=T, adaptiveDt=True, plot=a.video and last, video=a.video and last, show=False, store=False, progress=a.video and last, quiet=True,
                plotInterval=50, velocityAlarmPlotInterval=1, stallProgress=1e-3)
            res = run(case, spec)
            s = res.state.state
            f = s.kinds == 0
            pos = s.positions[f].double().cpu().numpy()
            v = s.velocities[f].double().cpu().numpy()
            p = s.pressures[f].double().cpu().numpy() if s.pressures is not None else np.zeros(len(pos))
            rho = s.densities[f].double().cpu().numpy()
            m = s.masses[f].double().cpu().numpy()
            yTop = pos[:, 1].max() + 0.5 * dx
            exp = G * (yTop - pos[:, 1])
            interior = res.ctx.scratch['interiorDomain']
            lo, hi = np.array([float(interior.min[0]), float(interior.min[1])]), np.array([float(interior.max[0]), float(interior.max[1])])
            dwall = np.minimum.reduce([pos[:, 0] - lo[0], hi[0] - pos[:, 0], pos[:, 1] - lo[1]])
            if a.wedge:
                r = WEDGE_MAX_EXTENT
                k = math.sqrt(3.0)
                cy = lo[1] + r / 4
                A = np.array([-r, cy - r / k / 2]); B = np.array([r, cy - r / k / 2]); C = np.array([0.0, cy + 2 * r / k / 2])
                dwedge = np.minimum(segDist(pos, A, C), segDist(pos, C, B))
            else:
                dwedge = np.full(len(pos), 9.0)
            nearW = (dwedge < 3 * dx)
            nearWall = (dwall < 3 * dx) & ~nearW
            bulk = ~nearW & ~nearWall & (pos[:, 1] < yTop - 3 * dx)
            ratio = lambda mask: float(np.mean(p[mask] / exp[mask].clip(min=1e-9))) if mask.any() else float('nan')
            rms = lambda mask: float(np.sqrt(np.mean(((p[mask] - exp[mask]) / (G * WATER_H)) ** 2))) if mask.any() else float('nan')
            ke = 0.5 * float((m * (v ** 2).sum(1)).sum())
            print(f'   t={float(res.state.t):.2f}: max|v| {np.linalg.norm(v, axis=1).max():.3f}  KE {ke:.3e}  p/hydro bulk {ratio(bulk):.3f} (rms err {rms(bulk):.3f})  near walls {ratio(nearWall):.3f} ({rms(nearWall):.3f})  '
                  f'near wedge {ratio(nearW):.3f} ({rms(nearW):.3f})  rho [{rho.min():.3f}, {rho.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()
