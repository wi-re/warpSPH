"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 16, 2026-08-29): the shear-wave case,
and the `ShiftApplication` question it was ported to settle.

`cases/periodic/shearWave.py` is Cornelis et al.'s reference case: a transverse
sinusoidal shear wave on a periodic box, whose exact solution is
`u_x = u0 sin(k_w y) exp(-nu k_w^2 t)` with a **constant pressure**. Every
pressure the solver produces is therefore an artifact, and at the default
`nu = 0` the exact answer is that nothing happens at all.

That gives two independent axes, which is the whole reason [C] reports two
figures on this case and the reason §1.2's question could not be settled on
`tgv`:

  amplitudeRatio   the velocity projected onto the analytic mode over the
                   analytic amplitude. 1.0 is exact; below 1.0 is artificial
                   viscosity ([C] Fig. 3).
  maxDensity       volume error and disorder ([C] Fig. 4).

A scheme can hold one and wreck the other, and §1.2 says the `ShiftApplication`
modes do exactly that: the position shift is momentum-neutral, so it should
cost amplitude nothing, while the two velocity modes feed a permanent residual
into momentum and should show up here as dissipation. On `tgv` that shows up
as a decay rate 0.55x / 3.3x analytic, but `tgv`'s decay has no published
counterpart (§5 Q5) and its analytic solution carries a real pressure field, so
the measurement conflates pressure error with dissipation. Here it does not.

  --mode shift        The three `ShiftApplication` modes on both axes.
  --mode resolution   Does the artifact converge? Same simulated time, four
                      resolutions, at the shipped defaults.
  --mode viscous      Grade against a *nontrivial* analytic decay by turning
                      `nu` on -- the amplitude then has an exponential to
                      follow rather than a constant, which tests the solver
                      against a moving target rather than a stationary one.

Usage:
  python scripts/probe_shearWave.py --mode shift --nx 128 --tLimit 2.0
  python scripts/probe_shearWave.py --mode resolution
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='shift', choices=['shift', 'resolution', 'viscous'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--tLimit', type=float, default=2.0)
parser.add_argument('--resolutions', nargs='*', type=int, default=[32, 64, 128, 256])
parser.add_argument('--nus', nargs='*', type=float, default=[0.0, 0.001, 0.01])
parser.add_argument('--nu', type=float, default=None,
                    help="--mode shift: run the three modes at this viscosity instead of 0")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import math

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import ShiftApplication
from warpSPH.cases.periodic.shearWave import shearWaveCase as case


def run(nx, tLimit, extra=(), configure=None):
    _orig = case.configureScheme

    def _wrapped(ctx):
        _orig(ctx)
        if configure is not None:
            configure(ctx.schemeConfig.solverConfig)

    case.configureScheme = _wrapped
    try:
        return caseMain(case, argv=[
            '--nx', str(nx), '--tLimit', str(tLimit),
            '--quiet', '--no-store', '--no-plot',
        ] + list(extra))
    finally:
        case.configureScheme = _orig


def summarise(r):
    """Final and worst values of both axes, plus the cost."""
    tr = [row for row in r.trajectory
          if all(isinstance(v, float) is False or math.isfinite(v) for v in row.values())]
    if not tr:
        return None
    last = tr[-1]
    return {
        'steps': len(tr) - 1,
        't': last.get('t', float('nan')),
        'amp': last.get('amplitudeRatio', float('nan')),
        'ampMin': min(row.get('amplitudeRatio', float('nan')) for row in tr),
        'rho': max(row.get('maxDensity', float('nan')) for row in tr),
        'rhoFinal': last.get('maxDensity', float('nan')),
        'resid': last.get('residualVelocity', float('nan')),
        'trans': max(row.get('transverseVelocity', float('nan')) for row in tr),
        'wall': r.wallTime,
    }


def header(cols):
    print(' '.join(cols))
    print('-' * (sum(len(c) for c in cols) + len(cols) - 1))


if args.mode == 'shift':
    nu = 0.0 if args.nu is None else args.nu
    extra = [] if args.nu is None else ['--nu', str(nu)]
    print(f"\n=== shearWave nx={args.nx} t={args.tLimit} nu={nu} -- "
          f"ShiftApplication on both axes ===")
    print("amplitude 1.0 is exact (no dissipation); maxDensity 1.0 is exact "
          "(no volume error)\n")
    cols = [f"{'mode':>20}", f"{'steps':>6}", f"{'amplitude':>10}", f"{'worst amp':>10}",
            f"{'max rho':>9}", f"{'final rho':>10}", f"{'residual v':>11}",
            f"{'max |v_y|':>10}", f"{'wall s':>7}"]
    header(cols)
    for name in ('positionShift', 'positionAndVelocity', 'inStepVelocity'):
        def cfg(sc, name=name):
            sc.shiftApplication = getattr(ShiftApplication, name)
        s = summarise(run(args.nx, args.tLimit, extra=extra, configure=cfg))
        if s is None:
            print(f"{name:>20}   (no finite step)")
            continue
        print(f"{name:>20} {s['steps']:6d} {s['amp']:10.6f} {s['ampMin']:10.6f} "
              f"{s['rho']:9.5f} {s['rhoFinal']:10.5f} {s['resid']:11.4e} "
              f"{s['trans']:10.4e} {s['wall']:7.1f}", flush=True)

elif args.mode == 'resolution':
    print(f"\n=== shearWave t={args.tLimit} nu=0 -- resolution, shipped defaults ===")
    cols = [f"{'nx':>6}", f"{'steps':>6}", f"{'amplitude':>10}", f"{'worst amp':>10}",
            f"{'max rho':>9}", f"{'residual v':>11}", f"{'max |v_y|':>10}", f"{'wall s':>7}"]
    header(cols)
    for nx in args.resolutions:
        s = summarise(run(nx, args.tLimit))
        if s is None:
            print(f"{nx:6d}   (no finite step)")
            continue
        print(f"{nx:6d} {s['steps']:6d} {s['amp']:10.6f} {s['ampMin']:10.6f} "
              f"{s['rho']:9.5f} {s['resid']:11.4e} {s['trans']:10.4e} "
              f"{s['wall']:7.1f}", flush=True)

else:
    print(f"\n=== shearWave nx={args.nx} t={args.tLimit} -- against a nontrivial "
          f"analytic decay ===")
    print("amplitude is measured *relative to* exp(-nu k_w^2 t), so 1.0 is exact "
          "at every nu\n")
    cols = [f"{'nu':>8}", f"{'steps':>6}", f"{'analytic decay':>15}",
            f"{'amplitude':>10}", f"{'worst amp':>10}", f"{'max rho':>9}",
            f"{'residual v':>11}", f"{'wall s':>7}"]
    header(cols)
    for nu in args.nus:
        s = summarise(run(args.nx, args.tLimit, extra=['--nu', str(nu)]))
        if s is None:
            print(f"{nu:8.4g}   (no finite step)")
            continue
        kw = 2.0 * math.pi * case.params['k'] / case.defaults['L']
        decay = math.exp(-nu * kw ** 2 * s['t'])
        print(f"{nu:8.4g} {s['steps']:6d} {decay:15.6f} {s['amp']:10.6f} "
              f"{s['ampMin']:10.6f} {s['rho']:9.5f} {s['resid']:11.4e} "
              f"{s['wall']:7.1f}", flush=True)
