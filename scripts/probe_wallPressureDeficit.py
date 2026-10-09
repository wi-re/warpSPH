"""Dissect the near-wall pressure deficit of the Monaghan shockReflection wall.

Hypothesis (from COMPRESSIBLE_WALLS_PLAN.md log): the last fluid row reads low
because the wall lattice (fixed spacing dx) is too coarse to resolve the
*compressed* fluid's smaller kernel width h, so the wall side of the density
sum is under-sampled.

This probe measures, at a fixed time:
  * the deficit p/p3-1 and rho/rho3-1 on the last K fluid rows
  * the kernel width h of those rows vs the wall spacing dx (h/dx)
  * the local (compressed) fluid spacing vs the wall spacing

Run:
  cd /home/lu26029/dev/warpSPH && /home/lu26029/miniconda3/envs/warp/bin/python \
      scripts/probe_wallPressureDeficit.py --t 0.45 --K 12
"""
import argparse

import torch

from warpSPH.cases.shockReflection import shockReflectionCase
from warpSPH.runner import run


def exactStates(gamma, M, rho1, p1):
    c1 = (gamma * p1 / rho1) ** 0.5
    p2 = p1 * (1 + 2 * gamma / (gamma + 1) * (M * M - 1))
    rho2 = rho1 * (gamma + 1) * M * M / ((gamma - 1) * M * M + 2)
    u2 = 2 / (gamma + 1) * c1 * (M - 1 / M)
    r = ((3 * gamma - 1) * p2 / p1 - (gamma - 1)) / ((gamma - 1) * p2 / p1 + (gamma + 1))
    p3 = p2 * r
    rho3 = rho2 * ((gamma + 1) * r + (gamma - 1)) / ((gamma - 1) * r + (gamma + 1))
    return dict(rho2=rho2, p2=p2, u2=u2, p3=p3, rho3=rho3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--t', type=float, default=0.45, help='simulated time to measure at')
    ap.add_argument('--K', type=int, default=14, help='number of near-wall fluid rows to dissect')
    ap.add_argument('--nx', type=int, default=400)
    ap.add_argument('--mach', type=float, default=2.0)
    ap.add_argument('--scheme', default='Monaghan')
    ap.add_argument('--supportMode', default=None, help="unset: Gather, KernelMeanSymmetric for CRKSPH")
    ap.add_argument('--exportRoot', default='results/compressibleWalls/probes')
    ap.add_argument('--adaptiveSupportScheme', default=None, help="'Owen' (case default) or 'Monaghan' (Newton on h(rho))")
    ap.add_argument('--supportVolumeClamp', choices=('always', 'walls', 'off'), default=None,
                    help='SimulationConfig.supportVolumeClamp (OPEN_PROBLEMS §20); unset: the default')
    a = ap.parse_args()
    case = shockReflectionCase
    if a.supportVolumeClamp is not None:
        import dataclasses
        configure = case.configureScheme

        def configureScheme(ctx):
            configure(ctx)
            ctx.config.supportVolumeClamp = a.supportVolumeClamp
        case = dataclasses.replace(case, configureScheme=configureScheme)
    supportMode = a.supportMode or ('KernelMeanSymmetric' if a.scheme == 'CRKSPH' else 'Gather')

    res = run(case, quiet=True, progress=True, plot=True, video=True,
              exportRoot=a.exportRoot, velocityAlarmPlotInterval=1, stallProgress=1e-3,
              scheme=a.scheme, supportMode=supportMode,
              tLimit=a.t, nx=a.nx, params=dict(mach=a.mach, **({'adaptiveSupportScheme': a.adaptiveSupportScheme} if a.adaptiveSupportScheme else {})))
    if res.diverged:
        print(f'diverged: {res.stopReason}')
        return

    sys_ = res.state
    s = sys_.state
    e = exactStates(1.4, a.mach, 1.0, 1.0)

    dx = 2.0 / a.nx  # wall spacing == fluid spacing, fixed at build time
    fluid = s.kinds == 0
    order = torch.argsort(s.positions[fluid, 0])
    x = s.positions[fluid, 0][order].cpu()
    rho = s.densities[fluid][order].cpu()
    p = s.pressures[fluid][order].cpu()
    h = s.supports[fluid][order].cpu()

    t = float(sys_.t)
    print(f'{a.scheme}  t={t:.4f}  nx={a.nx}  M={a.mach}  wallSpacing dx={dx:.5f}')
    print(f'  state3: rho3={e["rho3"]:.4f}  p3={e["p3"]:.4f}   (state2 rho2={e["rho2"]:.4f} p2={e["p2"]:.4f})')
    print()
    hdr = f'{"dW":>3} {"x":>8} {"rho":>8} {"p":>8} {"h":>8} {"h/dx":>6} {"dx_c/dx":>8} {"p/p3-1":>8} {"rho/rho3-1":>10}'
    print(hdr)
    n = x.numel()
    # last K fluid rows (right-wall side), dW=1 is the outermost fluid row next to the wall
    for k in range(1, a.K + 1):
        i = n - k
        dxc = (x[i] - x[i - 1]) / dx if i > 0 else float('nan')  # local (compressed) fluid spacing
        print(f'{k:>3} {x[i]:>8.4f} {rho[i]:>8.4f} {p[i]:>8.4f} {h[i]:>8.4f} '
              f'{h[i] / dx:>6.3f} {dxc:>8.3f} {p[i] / e["p3"] - 1:>8.3f} {rho[i] / e["rho3"] - 1:>10.4f}')


if __name__ == '__main__':
    main()
