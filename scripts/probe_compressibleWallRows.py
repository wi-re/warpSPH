"""Print the rows either side of the right wall of `shockReflection`, every step
in a time window, to see how a wall interaction goes wrong (COMPRESSIBLE_WALLS_PLAN.md).

    python scripts/probe_compressibleWallRows.py --scheme CRKSPH --supportMode KernelMeanSymmetric \
        --t0 0.29 --tLimit 0.31

Runs with video and progress like every probe; the per-step table is printed
from the diagnostics hook, flushed.
"""
import argparse
import dataclasses

import torch

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shockReflection import diagnostics as caseDiagnostics  # noqa: E402
from warpSPH.cases.shockReflection import shockReflectionCase  # noqa: E402
from warpSPH.runner import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='Monaghan')
    ap.add_argument('--supportMode', default='Gather')
    ap.add_argument('--t0', type=float, default=0.29, help='start printing rows at this time')
    ap.add_argument('--tLimit', type=float, default=0.31)
    ap.add_argument('--fluidRows', type=int, default=5)
    ap.add_argument('--wallRows', type=int, default=3)
    ap.add_argument('--mach', type=float, default=2.0)
    ap.add_argument('--exportRoot', default='results/compressibleWalls/probes')
    a = ap.parse_args()

    def diagnostics(ctx, state):
        out = caseDiagnostics(ctx, state)
        if float(state.t) < a.t0:
            return out
        s = state.state
        x = s.positions[:, 0]
        order = torch.argsort(x)
        kinds = s.kinds[order]
        firstWall = int(torch.nonzero((kinds != 0) & (x[order] > 0)).min())
        rows = order[firstWall - a.fluidRows:firstWall + a.wallRows]
        print(f'-- t={float(state.t):.5f}', flush=True)
        for r in rows.tolist():
            print(f'   k={int(s.kinds[r])} x={x[r]:+.5f} rho={s.densities[r]:8.4f} p={s.pressures[r]:8.4f} '
                  f'u={s.internalEnergies[r]:8.4f} v={s.velocities[r, 0]:+8.4f} h={s.supports[r]:.5f} '
                  f'm={s.masses[r]:.5f}', flush=True)
        return out

    case = dataclasses.replace(shockReflectionCase, diagnostics=diagnostics)
    res = run(case, scheme=a.scheme, supportMode=a.supportMode, tLimit=a.tLimit,
              params=dict(mach=a.mach), plot=True, video=True, exportRoot=a.exportRoot,
              progress=True, velocityAlarmPlotInterval=1, stallProgress=1e-3, plotInterval=5)
    print(f'diverged={res.diverged} {getattr(res, "stopReason", "")}', flush=True)


if __name__ == '__main__':
    main()
