"""Print the rows at the leading point of `shockCylinder`'s cylinder over a time
window: fluid rows within `--fluidRadius` spacings and wall rows within
`--wallRadius` spacings of the stagnation point, sorted by distance from the
cylinder centre (COMPRESSIBLE_WALLS_PLAN.md: the stagnation-point pile-up).

    python scripts/probe_cylinderFront.py --t0 0.36 --tLimit 0.42 --every 20
"""
import argparse
import dataclasses

import torch

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shockCylinder import diagnostics as caseDiagnostics  # noqa: E402
from warpSPH.cases.shockCylinder import shockCylinderCase  # noqa: E402
from warpSPH.runner import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='Monaghan')
    ap.add_argument('--t0', type=float, default=0.36)
    ap.add_argument('--tLimit', type=float, default=0.42)
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--fluidRadius', type=float, default=2.0)
    ap.add_argument('--wallRadius', type=float, default=1.6)
    ap.add_argument('--exportRoot', default='results/compressibleWalls/probes')
    a = ap.parse_args()
    count = {'n': 0}

    def diagnostics(ctx, state):
        out = caseDiagnostics(ctx, state)
        if float(state.t) < a.t0:
            return out
        count['n'] += 1
        if count['n'] % a.every:
            return out
        s, dx = state.state, ctx.config.dx
        cx, cy = ctx.param('centreX'), ctx.param('centreY')
        R = ctx.param('radius')
        lead = torch.tensor([cx - R, cy], dtype=s.positions.dtype, device=s.positions.device)
        d = torch.linalg.norm(s.positions - lead, dim=-1)
        r = torch.linalg.norm(s.positions - torch.tensor([cx, cy], dtype=s.positions.dtype, device=s.positions.device), dim=-1)
        rows = ((s.kinds == 0) & (d < a.fluidRadius * dx)) | ((s.kinds != 0) & (d < a.wallRadius * dx))
        idx = torch.nonzero(rows).squeeze(-1)
        idx = idx[torch.argsort(r[idx], descending=True)]
        print(f'-- t={float(state.t):.5f}  (r-R)/dx, then state', flush=True)
        n = s.wallNormals if getattr(s, 'wallNormals', None) is not None else torch.zeros_like(s.positions)
        for i in idx.tolist():
            print(f'   k={int(s.kinds[i])} (r-R)/dx={(r[i] - R) / dx:+6.2f} y/dx={(s.positions[i, 1] - cy) / dx:+6.2f} '
                  f'v=({s.velocities[i, 0]:+.3f},{s.velocities[i, 1]:+.3f}) rho={s.densities[i]:7.3f} p={s.pressures[i]:8.3f} '
                  f'h/dx={s.supports[i] / dx:.2f} m={s.masses[i]:.2e} n=({n[i, 0]:+.2f},{n[i, 1]:+.2f})', flush=True)
        return out

    case = dataclasses.replace(shockCylinderCase, diagnostics=diagnostics)
    supportMode = 'KernelMeanSymmetric' if a.scheme == 'CRKSPH' else 'Gather'
    run(case, scheme=a.scheme, supportMode=supportMode, tLimit=a.tLimit, plot=True, video=True, progress=True,
        exportRoot=a.exportRoot, velocityAlarmPlotInterval=1, stallProgress=1e-3, plotInterval=10)


if __name__ == '__main__':
    main()
