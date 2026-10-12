"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 10, 2026-08-28): is the initial
particle sampling consistent with `rho0`, and does the transient it causes
matter?

`sample/regular.py:70` gives every particle the nominal mass `rho0 * dx^d`.
That is the *continuum* mass of its cell, not the mass that makes the discrete
SPH summation `rho_i = sum_j m_j W_ij` come out at `rho0` -- those differ by
the kernel's discrete normalisation on the actual sampling, and the difference
does not vanish with resolution (it is a property of the lattice-plus-kernel
pair, not a truncation error). `kolmogorovIncompressible.py:71` and `tgv.py:53`
correct for it by rescaling mass so `mean(rho) == rho0`; the `randomFlow`
family, and every case built through `buildRegionSystem`, does not.

Any mismatch is a step-0 density error the pressure solvers immediately try to
remove, i.e. a startup transient in a case whose whole diagnostic is
`|rho - 1|`. Two modes:

  --mode measure  Step 0 only: the sampled density against `rho0`, split by
      distance to the wall (a wall band samples differently from the bulk, so a
      single global factor is not obviously the right correction), plus the
      normalisation factor each candidate rule would apply.

  --mode ab       Run the case under each rule and report the transient (peak
      `|rho-1|` over the first `--transient` steps, and when it peaks) next to
      the long-run numbers.

Usage:
  python scripts/probe_initialSampling.py --mode measure
  python scripts/probe_initialSampling.py --mode ab --nsteps 900
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='measure', choices=['measure', 'ab'])
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="0.4 is Bender & Koschier's published constant, and since Part 12 `cflFactor` multiplies the particle diameter, so the number here is theirs. Numbers recorded in DFSPH_IMPROVEMENT_PLAN.md before Part 12 say `cflFactor=0.1`, which was the same timestep under the old support-radius convention")
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--trace', type=int, default=0,
                    help="--mode ab: also print the first N steps' density error, "
                         "which is where a sampling-induced startup shock would be")
parser.add_argument('--transient', type=int, default=50,
                    help="--mode ab: steps counted as the startup transient")
parser.add_argument('--rules', nargs='*', default=['none', 'meanAll', 'bulk'],
                    help="mass-normalisation rules to compare")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

import torch

from warpSPH.runner.cli import caseMain
from warpSPH.modules.density import computeDensities
from warpSPH.cases.common.weaklyCompressible import domainBoundarySdf

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

BINS = [(-1e9, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]


def binLabel(lo, hi):
    if lo < -1e8:
        return "(<0 inside)"
    return f"[{lo:g},inf)" if hi > 1e8 else f"[{lo:g},{hi:g})"


def sampledDensity(ctx, system):
    """The plain SPH summation density on the as-sampled state -- the same
    `computeDensities` the scheme calls on step 0 before anything has moved."""
    return computeDensities(system.state, ctx.config, ctx.schemeConfig, None)


def normalisationFactor(rule, rho, fluid, depth, rho0):
    """The factor `masses` would be multiplied by. Densities are linear in
    mass, so scaling every particle by `rho0 / <rho>` lands `<rho>` on `rho0`
    exactly in one shot -- no iteration needed."""
    if rule == 'none':
        return 1.0
    if rule == 'meanAll':
        return rho0 / float(rho[fluid].mean())
    if rule == 'bulk':
        # Bulk only: a wall band samples differently from the interior, so a
        # global mean lets the band set the interior's rest density.
        m = fluid & (depth >= 4)
        if not bool(m.any()):
            m = fluid
        return rho0 / float(rho[m].mean())
    if rule == 'median':
        return rho0 / float(rho[fluid].median())
    raise ValueError(rule)


def withRule(rule):
    """Rescale every particle's mass right after the case builds, the way
    `kolmogorovIncompressible.buildSystem` already does for its own case."""
    _orig = case.buildSystem

    def _wrapped(ctx, _orig=_orig, rule=rule):
        system = _orig(ctx)
        if rule != 'none':
            rho0 = ctx.param('rho0')
            rho = sampledDensity(ctx, system)
            fluid = system.state.kinds == 0
            depth = wallDepth(ctx, system.state)
            f = normalisationFactor(rule, rho, fluid, depth, rho0)
            system.state.masses = system.state.masses * f
        return system

    case.buildSystem = _wrapped
    return lambda: setattr(case, 'buildSystem', _orig)


def wallDepth(ctx, state):
    try:
        sdf = domainBoundarySdf(ctx)
        d, _ = sdf(state.positions.detach().clone().requires_grad_(True))
        return d.detach() / ctx.config.dx
    except Exception:  # noqa: BLE001 - periodic cases have no wall
        return torch.full_like(state.densities, 1e9)


def runMeasure():
    cap = {}
    _orig = case.buildSystem

    def _wrapped(ctx):
        system = _orig(ctx)
        cap.update(ctx=ctx, system=system)
        raise SystemExit  # stop before stepping: step 0 is the whole question

    case.buildSystem = _wrapped
    try:
        caseMain(case, argv=['--nx', str(args.nx), '--nSteps', '1', '--tLimit', '1000.0',
                             '--quiet', '--no-store', '--no-plot'] + args.extra)
    except SystemExit:
        pass
    finally:
        case.buildSystem = _orig

    ctx, system = cap['ctx'], cap['system']
    state = system.state
    rho0 = ctx.param('rho0')
    rho = sampledDensity(ctx, system)
    fluid = state.kinds == 0
    depth = wallDepth(ctx, state)
    dx = ctx.config.dx

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx}, as sampled (t=0) ===")
    print(f"rho0={rho0}  dx={dx:.6g}  m={float(state.masses[fluid][0]):.6g}  "
          f"m/dx^2={float(state.masses[fluid][0]) / dx ** 2:.6f}  "
          f"h/dx={float(state.supports.median()) / dx:.3f}")
    print(f"{int(fluid.sum())} fluid, {int((~fluid).sum())} static")

    f = rho[fluid]
    print(f"\nfluid summation density: mean={float(f.mean()):.6f} "
          f"std={float(f.std()):.6f} min={float(f.min()):.6f} max={float(f.max()):.6f}")
    print(f"  -> mean is {float(f.mean()) / rho0 - 1:+.4%} off rho0")

    print(f"\n{'depth':>12} {'n':>7} {'mean rho':>10} {'std':>10} {'min':>9} {'max':>9}")
    for lo, hi in BINS:
        m = fluid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0:
            continue
        r = rho[m]
        print(f"{binLabel(lo, hi):>12} {n:7d} {float(r.mean()):10.6f} {float(r.std()):10.6f} "
              f"{float(r.min()):9.6f} {float(r.max()):9.6f}")

    print(f"\n{'rule':>10} {'mass factor':>12} {'resulting mean rho':>19}")
    for rule in ('none', 'meanAll', 'bulk', 'median'):
        fac = normalisationFactor(rule, rho, fluid, depth, rho0)
        print(f"{rule:>10} {fac:12.6f} {float(f.mean()) * fac:19.6f}")


def runAB():
    rows = []
    for rule in args.rules:
        restore = withRule(rule)
        try:
            r = caseMain(case, argv=[
                '--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
                '--cflFactor', str(args.cflFactor), '--quiet', '--no-store', '--no-plot',
            ] + args.extra)
        finally:
            restore()
        rows.append((rule, r))

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} nSteps={args.nsteps} "
          f"cflFactor={args.cflFactor} ===")
    print(f"{'rule':>10} {'steps':>7} {'div':>5} {'peak |rho-1|':>13} {'at step':>8} "
          f"{'|rho-1| first ' + str(args.transient):>20} {'|rho-1| 2nd half':>17} {'t_final':>9}")
    for rule, r in rows:
        tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
        err = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0)) for row in tr]
        head = err[:args.transient] or err
        tail = err[len(err) // 2:] or err
        peak = max(head) if head else float('nan')
        print(f"{rule:>10} {len(tr):7d} {str(r.diverged):>5} {peak:13.4e} "
              f"{(head.index(peak) if head else -1):8d} "
              f"{sum(head) / max(1, len(head)):20.4e} "
              f"{sum(tail) / max(1, len(tail)):17.4e} "
              f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):9.4f}")

    if args.trace:
        print(f"\nfirst {args.trace} steps, max(|maxRho-1|, |minRho-1|):")
        for rule, r in rows:
            tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
            err = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0))
                   for row in tr][:args.trace]
            print(f"{rule:>10} " + " ".join(f"{e:.2e}" for e in err))


if args.mode == 'measure':
    runMeasure()
else:
    runAB()
