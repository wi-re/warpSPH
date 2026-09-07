"""Probe (`DELTASPH_VALIDATION_PLAN.md` Part 2): is `modules/shifting/delta.py`
actually Sun et al. 2017 Eq. (7)?

Eq. (7) verbatim (`literature/sun2017_delta-plus-sph-model.pdf`, p. 28), with
`phi_ij = 1` and `h_ij = h` at uniform resolution:

    delta_r_i := -CFL * Ma * (2 h)^2 * sum_j [ 1 + R (W_ij/W(dx_i))^n ]
                 grad_i W_ij * 2 m_j / (rho_i + rho_j)

with `R = 0.2`, `n = 4`.  The paper also states a *measurable* property of that
formula, Eq. (8): across every simulation in the paper,

    |delta_r_i| / dx_i  <  0.05,

"and we found that this value tends to zero as the resolution increases".  That
makes the shift magnitude an absolute, dimensionless check on the
implementation rather than a matter of reading the code -- which is what this
probe measures, on a relaxed Taylor-Green state (Sun's own benchmark no. 1), at
several resolutions.

It reports, per configuration:

* `|dr|/dx` mean and max -- against Eq. (8)'s `< 0.05`;
* the ratio to the **literal Eq. (7)** shift, recomputed here from the same raw
  kernel sum, so "the implementation is X times Eq. (7)" is a measured number
  and not an argument about which factor of two belongs where.

Usage:
  python scripts/probe_deltaPlusShiftMagnitude.py                 # nx 50/100/200
  python scripts/probe_deltaPlusShiftMagnitude.py --nx 100 --steps 40
"""
from __future__ import annotations

import argparse
import math


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='*', default=[50, 100, 200])
    ap.add_argument('--steps', type=int, default=25,
                    help='real steps to advance before measuring, so the state is a '
                         'running one rather than the relaxed initial lattice')
    ap.add_argument('--Re', type=float, default=100.0)
    args = ap.parse_args(argv)

    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    from warpSPHCore import sphKernelScale
    from warpSPH.cases.tgvWeaklyCompressible import tgvWeaklyCompressibleCase
    from warpSPH.modules.shifting.delta import computeDeltaShift
    from warpSPH.runner import run
    from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp
    from warpSPHCore import OperationProperties, SupportScheme, WarpOperation

    print(f'{"L/dx":>6} {"|dr|/dx mean":>13} {"|dr|/dx max":>12} '
          f'{"Eq.(8) < 0.05":>14} {"impl / Eq.(7)":>14}')
    for nx in args.nx:
        nu = 1.0 / args.Re
        r = run(tgvWeaklyCompressibleCase, scheme='sun2017DeltaSPH', L=1.0, nx=nx,
                kernel='Wendland2', integrationScheme='rungeKutta4',
                nSteps=args.steps, quiet=True, store=False, progress=False,
                params=dict(k=4 * math.pi, phase=math.pi / 2, uMag=1.0, nu=nu,
                            inviscid=False, machTarget=0.1, referenceVelocity=1.0,
                            initialPressure=True, shuffleIters=128))
        state = r.state
        ctx = r.ctx
        dx = float(ctx.config.dx)

        # What the production path actually applies, once.
        shift, _adj = computeDeltaShift(state.state, ctx.config, ctx.schemeConfig,
                                        ctx.config.domain, state.adjacency, iters=1)
        mag = torch.linalg.norm(shift, dim=-1) / dx

        # The literal Eq. (7), rebuilt from the same raw kernel sum.  The raw sum
        # carries the weight `0.5 m_j/(rho_i+rho_j)`; Eq. (7) asks for
        # `2 m_j/(rho_i+rho_j)`, hence the factor 4, and its prefactor is
        # `(2h)^2 = 4 h^2` where the production path applies `2 h^2`.
        raw = computeDeltaShiftWarp(
            state.state,
            operationProperties=OperationProperties(
                operation=WarpOperation.Density, kernel=ctx.config.kernel,
                supportMode=SupportScheme.Gather),
            referenceParticles=state.state, domain=ctx.config.domain,
            adjacency=state.adjacency,
            CFL=ctx.schemeConfig.shiftProperties.CFL,
            computeMach=ctx.schemeConfig.shiftProperties.computeMach,
            c_max=0.1, rho0=1.0, dx=dx, R=0.2, n=4)
        c0 = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
        vMax = float(torch.linalg.norm(state.state.velocities, dim=-1).max())
        Ma = vMax / c0
        kernelScale = float(sphKernelScale(ctx.config.kernel.value, ctx.config.dim))
        h = state.state.supports / kernelScale
        exact = raw * (-ctx.schemeConfig.shiftProperties.CFL * Ma
                       * 4.0 * (4.0 * h ** 2)).unsqueeze(-1)
        exactMag = torch.linalg.norm(exact, dim=-1) / dx

        ratio = float(mag.mean() / exactMag.mean()) if float(exactMag.mean()) else float('nan')
        print(f'{1.0 / dx:6.0f} {float(mag.mean()):13.5f} {float(mag.max()):12.5f} '
              f'{"yes" if float(mag.max()) < 0.05 else "NO":>14} {ratio:14.4f}')


if __name__ == '__main__':
    main()
