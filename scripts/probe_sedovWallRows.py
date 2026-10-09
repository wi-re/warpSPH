"""Run `sedovWalls` and list its extreme rows (COMPRESSIBLE_WALLS_PLAN.md).

At the end (and every `--every` steps) prints the `--top` rows by pressure and
by density, fluid and wall separately, with position, state, support and mass,
plus the fluid's minimum internal energy -- to see where a wall state goes
extreme (the box corners) and whether fluid u goes negative.

    python scripts/probe_sedovWallRows.py --scheme CRKSPH --goalRadius 1.4
"""
import argparse
import dataclasses

import torch

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.sedovWalls import diagnostics as caseDiagnostics  # noqa: E402
from warpSPH.cases.sedovWalls import sedovWallsCase  # noqa: E402
from warpSPH.runner import run  # noqa: E402


def table(s, mask, key, top, label):
    idx = torch.nonzero(mask).squeeze(-1)
    order = torch.argsort(getattr(s, key)[idx], descending=True)[:top]
    for r in idx[order].tolist():
        x = s.positions[r]
        print(f'   {label} {key[:-1]:9s} x=({x[0]:+.3f},{x[1]:+.3f}) rho={s.densities[r]:9.4f} p={s.pressures[r]:9.4f} '
              f'u={s.internalEnergies[r]:9.4f} |v|={torch.linalg.norm(s.velocities[r]):.3f} h={s.supports[r]:.4f} '
              f'm={s.masses[r]:.3e}', flush=True)


def dump(state, top):
    s = state.state
    fluid = s.kinds == 0
    print(f'-- t={float(state.t):.4f} fluid u min {s.internalEnergies[fluid].min():.3e}', flush=True)
    for key in ('pressures', 'densities'):
        table(s, fluid, key, top, 'fluid')
        table(s, ~fluid, key, top, 'wall ')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='CRKSPH')
    ap.add_argument('--supportMode', default=None)
    ap.add_argument('--nx', type=int, default=101)
    ap.add_argument('--goalRadius', type=float, default=1.4)
    ap.add_argument('--every', type=int, default=500)
    ap.add_argument('--top', type=int, default=4)
    ap.add_argument('--exportRoot', default='results/compressibleWalls/probes')
    a = ap.parse_args()
    supportMode = a.supportMode or ('KernelMeanSymmetric' if a.scheme == 'CRKSPH' else 'Gather')
    count = {'n': 0}

    def diagnostics(ctx, state):
        out = caseDiagnostics(ctx, state)
        count['n'] += 1
        if count['n'] % a.every == 0:
            dump(state, a.top)
        return out

    case = dataclasses.replace(sedovWallsCase, diagnostics=diagnostics)
    res = run(case, scheme=a.scheme, supportMode=supportMode, nx=a.nx, params=dict(goalRadius=a.goalRadius),
              plot=True, video=True, progress=True, exportRoot=a.exportRoot,
              velocityAlarmPlotInterval=1, stallProgress=1e-3)
    dump(res.state, a.top)
    print(f'diverged={res.diverged}', flush=True)


if __name__ == '__main__':
    main()
