"""Find fluid rows that get behind the moving piston of `shockCylinder` (2D).

Prints, from the first step any fluid row sits behind the piston face, each such
row's position, velocity, density, pressure, and the nearest wall rows' state
and velocity -- to see where (corner / face) and how fluid leaks past a moving
wall that slides along a fixed one (COMPRESSIBLE_WALLS_PLAN.md).

    python scripts/probe_pistonLeak.py --tLimit 0.22
"""
import argparse
import dataclasses

import torch

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shockCylinder import cylinderStates  # noqa: E402
from warpSPH.cases.shockCylinder import diagnostics as caseDiagnostics  # noqa: E402
from warpSPH.cases.shockCylinder import shockCylinderCase  # noqa: E402
from warpSPH.runner import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='Monaghan')
    ap.add_argument('--tLimit', type=float, default=0.22)
    ap.add_argument('--maxReports', type=int, default=30)
    ap.add_argument('--exportRoot', default='results/compressibleWalls/probes')
    a = ap.parse_args()
    reports = {'n': 0}

    def diagnostics(ctx, state):
        out = caseDiagnostics(ctx, state)
        if reports['n'] >= a.maxReports:
            return out
        s, t = state.state, float(state.t)
        dx = ctx.config.dx
        face = cylinderStates(ctx)['u2'] * t
        fluid = s.kinds == 0
        behind = torch.nonzero(fluid & (s.positions[:, 0] < face - 0.5 * dx)).squeeze(-1)
        if behind.numel() == 0:
            return out
        reports['n'] += 1
        print(f'-- t={t:.5f} face={face:.4f} rows behind: {behind.numel()}', flush=True)
        wall = torch.nonzero(s.kinds != 0).squeeze(-1)
        for r in behind[:4].tolist():
            x = s.positions[r]
            print(f'   fluid x=({x[0]:+.4f},{x[1]:+.4f}) v=({s.velocities[r, 0]:+.3f},{s.velocities[r, 1]:+.3f}) '
                  f'rho={s.densities[r]:.3f} p={s.pressures[r]:.3f} h={s.supports[r]:.4f}', flush=True)
            d = torch.linalg.norm(s.positions[wall] - x, dim=-1)
            for w in wall[torch.argsort(d)[:3]].tolist():
                y = s.positions[w]
                print(f'      wall x=({y[0]:+.4f},{y[1]:+.4f}) v=({s.velocities[w, 0]:+.3f},{s.velocities[w, 1]:+.3f}) '
                      f'rho={s.densities[w]:.3f} p={s.pressures[w]:.3f} h={s.supports[w]:.4f}', flush=True)
        return out

    case = dataclasses.replace(shockCylinderCase, diagnostics=diagnostics)
    supportMode = 'KernelMeanSymmetric' if a.scheme == 'CRKSPH' else 'Gather'
    run(case, scheme=a.scheme, supportMode=supportMode, tLimit=a.tLimit, plot=True, video=True, progress=True,
        exportRoot=a.exportRoot, velocityAlarmPlotInterval=1, stallProgress=1e-3, plotInterval=10)


if __name__ == '__main__':
    main()
