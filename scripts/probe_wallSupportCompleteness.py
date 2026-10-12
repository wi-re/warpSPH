"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 8, 2026-08-28): is kernel support
actually truncated at the wall in the bounded case?

Several claims in this document assume it is. `ShiftPressureGauge`'s docstring
justifies falling back to `nonNegativeClamp` on any wall-bounded solve by
arguing that "where the support is truncated (against a wall, at a free
surface) the kernel gradients no longer sum to zero, so a *constant* pressure
exerts a large real force". Part 8 then explained the MINRES result away as "a
complete-support result only".

**The project owner's objection is that the premise is wrong**: this codebase
samples the boundary as a *solid band*, not a single surface layer
(`randomFlow.BOUNDED_BAND = 5` particle layers, against a support radius of
`h = n_h * dx = 4` spacings), so a fluid particle sitting on the wall still has
a full kernel's worth of neighbours -- they are simply `kind==1` rather than
`kind==0`. That is a deliberate deviation from the one-layer boundaries of
Akinci et al. (used by both [BK] and [C]), and a common one, bought with extra
particles and therefore extra compute.

This measures it directly rather than arguing it. For fluid particles binned by
signed distance to the wall (in particle spacings) it reports:

  - **Shepard sum** `sum_j V_j W_ij` -- the partition-of-unity residual. ~1
    means the neighbourhood is complete; a shortfall is missing support.
  - **|sum_j V_j grad W_ij|** -- the quantity the docstring's argument is
    actually about. If this is ~0 at the wall, a constant pressure is forceless
    there and the truncation argument does not apply.
  - **|A . 1|** -- the real PPE operator applied to a constant field, binned the
    same way, against |A . rand| for scale. This is the decisive one: it is the
    operator the solve actually iterates, and whether its constant mode is
    forceless near the wall is exactly what `ShiftPressureGauge` turns on.

Usage:
  python scripts/probe_wallSupportCompleteness.py --nx 128 --warmup 120
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--warmup', type=int, default=120,
                    help="steps before probing, so the configuration is developed")
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="0.4 is Bender & Koschier's published constant, and since Part 12 `cflFactor` multiplies the particle diameter, so the number here is theirs. Numbers recorded in DFSPH_IMPROVEMENT_PLAN.md before Part 12 say `cflFactor=0.1`, which was the same timestep under the old support-radius convention")
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import torch

from warpSPH.runner.cli import caseMain
from warpSPHCore import (SupportScheme, OperationProperties, WarpOperation,
                         OperationDirection, GradientScheme, warpOperation)
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.cases.common.weaklyCompressible import domainBoundarySdf

import warpSPH.systems.incompressible as sysmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

_real = sysmod.solveIncompressible
cap = {}


def _capture(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    cap.update(particles=particles, config=config, adjacency=adjacency, dt=dt)
    return _real(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)


sysmod.solveIncompressible = _capture
try:
    r = caseMain(case, argv=['--nx', str(args.nx), '--nSteps', str(args.warmup),
                             '--tLimit', '1000.0', '--cflFactor', str(args.cflFactor),
                             '--quiet', '--no-store', '--no-plot'] + args.extra)
finally:
    sysmod.solveIncompressible = _real

particles = cap['particles']
config = cap['config']
adjacency = cap['adjacency']
dt = cap['dt']

ones = torch.ones_like(particles.densities)
volumes = particles.masses / particles.densities

# Shepard sum: sum_j V_j W_ij, via an interpolation of the constant 1 field.
shepard = warpOperation(
    particles,
    operationProperties=OperationProperties(
        operation=WarpOperation.Interpolate, kernel=config.kernel,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
    queryValues=ones, domain=config.domain, adjacency=adjacency)

# sum_j V_j grad W_ij, via a Summation-mode gradient of the same constant.
gradSum = warpOperation(
    particles,
    operationProperties=OperationProperties(
        operation=WarpOperation.Gradient, kernel=config.kernel,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Summation),
    queryValues=ones, domain=config.domain, adjacency=adjacency)
gradSumMag = gradSum.norm(dim=-1) if gradSum.dim() > 1 else gradSum.abs()


def applyOperator(p):
    ap = computePressureAccelIISPH(state=particles, pressureValues=p, config=config,
                                   supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    return dt ** 2 * computePressureShiftIISPH(state=particles, config=config,
                                               pressureAccels=ap,
                                               supportScheme=SupportScheme.Scatter,
                                               adjacency=adjacency)


aOnes = applyOperator(ones).abs()
g = torch.Generator(device=ones.device); g.manual_seed(0)
aRand = applyOperator(torch.rand(ones.shape, generator=g, device=ones.device,
                                 dtype=ones.dtype)).abs()

ctx = r.ctx
sdf = domainBoundarySdf(ctx)
dxs = ctx.config.dx
d, _ = sdf(particles.positions.detach().clone().requires_grad_(True))
depth = (d.detach() / dxs)

fluid = particles.kinds == 0
bins = [(-1e9, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]

print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} after {args.warmup} steps ===")
print(f"BOUNDED_BAND = 5 layers, support radius h = {float(particles.supports.median()) / dxs:.2f} spacings")
print(f"\nfluid particles binned by distance to wall (particle spacings)")
print(f"{'depth':>12} {'n':>7} {'Shepard':>10} {'|sum V gradW|':>14} {'|A.1|':>11} "
      f"{'|A.rand|':>11} {'ratio':>9}")
for lo, hi in bins:
    m = fluid & (depth >= lo) & (depth < hi)
    n = int(m.sum())
    if n == 0:
        continue
    ao, ar = float(aOnes[m].mean()), float(aRand[m].mean())
    label = f"[{lo:g},{hi:g})" if lo > -1e8 else "(<0 inside)"
    label = label if hi < 1e8 else f"[{lo:g},inf)"
    print(f"{label:>12} {n:7d} {float(shepard[m].mean()):10.6f} "
          f"{float(gradSumMag[m].mean()):14.4e} {ao:11.4e} {ar:11.4e} "
          f"{(ao / ar if ar else float('nan')):9.4f}")

allFluid = fluid
print(f"\nall fluid: Shepard mean={float(shepard[allFluid].mean()):.6f} "
      f"min={float(shepard[allFluid].min()):.6f}")
print(f"           |A.1|/|A.rand| = {float(aOnes[allFluid].mean() / aRand[allFluid].mean()):.5f}")
print("\nA Shepard sum at ~1 and a |A.1|/|A.rand| ratio at ~0 in the near-wall bins")
print("means the constant mode is forceless there, i.e. support is NOT truncated")
print("and ShiftPressureGauge's truncation argument does not apply to this wall.")
