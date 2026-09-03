"""Audit of `ShiftProperties.correctdrhodt` (`WCSPH_SHIFTING_PLAN.md` §2d):
why does feeding the δ⁺ shift into the continuity equation collapse ρ to ~0.4
on the rotating square patch?

Sun et al. 2019 Eq. (9)-(10): the consistent formulation adds
`Δρ_i = ⟨div(ρ δr)⟩_i − ρ_i ⟨div(δr)⟩_i` (sum form for the first, difference
form for the second) to the density each step, `δr` the applied shift
displacement. The paper's `δr` is tiny (`δr/Δx ≤ 0.01`, their Fig. 18), so
this term is a small volume-conserving nudge. If warpSPH's `δr` is much
larger, the same term becomes a per-step ρ sledgehammer.

Runs `squarePatch --scheme deltaSPH` with `correctdrhodt` on, and every
`--every` steps reports, on the live state:
  - `ρ` bounds (the divergence signal);
  - the shift `|dx| / spacing` bulk vs surface -- compare to the paper's
    `<= 0.01` (their Fig. 18);
  - the per-step `Δρ / ρ` the `correctdrhodt` block applies, bulk vs surface;
  - the global `Σ_i Δρ_i V_i` (Eq. 16 says ~0 for a volume-conserving
    correction; a nonzero, sign-consistent value is a volume bias).

Usage:
  python scripts/probe_correctdrhodtAudit.py [--nx 96] [--tLimit 0.6]
      [--every 100] [--projection surfaceNormal|mat] [--noCorrect]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=96)
parser.add_argument('--tLimit', type=float, default=0.6)
parser.add_argument('--every', type=int, default=100)
parser.add_argument('--projection', default='surfaceNormal',
                    choices=['surfaceNormal', 'mat', 'dot'])
parser.add_argument('--noCorrect', action='store_true',
                    help='leave correctdrhodt off (baseline shift magnitudes)')
parser.add_argument('--omega', type=float, default=4.0)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase as case
from warpSPH.runner import run
from warpSPHCore import (OperationProperties, WarpOperation, SupportScheme,
                         OperationDirection, GradientScheme, warpOperation)
from warpSPH.modules.shifting.wrapper import solveShifting
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme


def _q(x):
    return torch.quantile(x.float(), 0.99).item() if x.numel() else float('nan')


hdr = (f"{'step':>6} {'t':>7} {'minRho':>8} {'maxRho':>8}  "
       f"{'|dx|/dx bulk p99':>16} {'|dx|/dx surf p99':>16} {'|dx|/dx surf max':>16}  "
       f"{'Δρ/ρ surf p99':>14}  {'ΣΔρV / ΣρV':>13}")


def postStep(ctx, system, i):
    if i == 0 or i % args.every != 0:
        return
    cfg, sc = ctx.config, ctx.schemeConfig
    p = system.state
    dt = float(cfg.dt)
    fluid = p.kinds == 0
    surf = (p.surfaceIndicators == 1) if getattr(p, 'surfaceIndicators', None) is not None \
        else torch.zeros_like(fluid)
    bulk = fluid & ~surf
    rho0 = sc.fluid.restDensity
    spacing = torch.pow(p.masses / rho0, 1.0 / p.positions.shape[1]).mean().item()

    rho = p.densities.clone()
    adj = getattr(system, 'adjacency', None)
    dx = solveShifting(systemState=p, config=cfg, schemeConfig=sc, adjacency=adj, dt=dt)
    mag = torch.linalg.norm(dx, dim=-1) / spacing

    du = dx / dt
    dRho = dt * (warpOperation(
        p, operationProperties=OperationProperties(
            operation=WarpOperation.Divergence, kernel=cfg.kernel,
            supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
            gradientMode=GradientScheme.Summation),
        queryValues=rho.view(-1, 1) * du, domain=cfg.domain, adjacency=adj)
        - rho * warpOperation(
        p, operationProperties=OperationProperties(
            operation=WarpOperation.Divergence, kernel=cfg.kernel,
            supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
            gradientMode=GradientScheme.Difference),
        queryValues=du, domain=cfg.domain, adjacency=adj))
    rel = (dRho / rho).abs()
    V = p.masses / p.densities
    bias = float((dRho * V)[fluid].sum()) / float((rho * V)[fluid].sum())

    print(f"{i:>6} {float(system.t):>7.4f} "
          f"{p.densities[fluid].min():>8.4f} {p.densities[fluid].max():>8.4f}  "
          f"{_q(mag[bulk]):>16.4f} {_q(mag[surf & fluid]):>16.4f} "
          f"{mag[surf & fluid].max().item() if (surf & fluid).any() else float('nan'):>16.4f}  "
          f"{_q(rel[surf & fluid]):>14.4f}  {bias:>+13.3e}")


_baseCfg, _basePost = case.configureScheme, case.postStep


def _cfg(ctx):
    _baseCfg(ctx)
    sc = ctx.schemeConfig
    sc.shiftProperties.projectionScheme = ShiftingProjectionScheme[args.projection]
    sc.shiftProperties.correctdrhodt = not args.noCorrect


case.configureScheme = _cfg
case.postStep = postStep
print(f"# nx={args.nx} projection={args.projection} correctdrhodt={not args.noCorrect}")
print(hdr)
try:
    run(case, params={'shape': 'box', 'omega': args.omega},
        nx=args.nx, tLimit=args.tLimit, nSteps=None,
        store=False, plot=False, quiet=True, progress=False)
finally:
    case.configureScheme, case.postStep = _baseCfg, _basePost
