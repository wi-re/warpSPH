"""A cylinder flying at Mach `mach` through gas at rest (2D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 7: the bow shock. Supersonic flow past a
cylinder is run in the gas's frame -- the cylinder (wall rows moving at
`U = mach * c0`) is started impulsively inside a closed box of gas at rest --
so no inflow or outflow boundary is needed. Once the bow shock has settled,
in the cylinder's frame it is the classic steady problem:

* shock standoff on the axis, Billig (1967) for a cylinder:
  `Delta / R = 0.386 exp(4.67 / M^2)` (0.649 at Mach 3);
* the pressure at the nose is the pitot pressure (Rayleigh):
  `p0' / p1 = ((g+1) M^2 / 2)^(g/(g-1)) ((g+1) / (2 g M^2 - (g-1)))^(1/(g-1))`
  (12.06 at Mach 3, g = 1.4).

`standoffRatio` is the measured standoff over Billig's: the shock is the
farthest fluid row on the axis (|y| < dx) ahead of the nose whose density is
past the midpoint between rho0 and the normal-shock density.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from .compressibleWalls import WALL_PARAMS, buildWalledBox, circleSDF, walledDomainBounds
from .plotting import Field, particlePlot

__all__ = ['bowShockCase', 'bowShockStates']


def _box(ctx: RunContext):
    return (0.0, -0.5 * ctx.param('height')), (ctx.spec.L, 0.5 * ctx.param('height'))


def bowShockStates(ctx: RunContext) -> Dict[str, float]:
    g, M = ctx.param('gamma'), ctx.param('mach')
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    c0 = math.sqrt(g * p0 / rho0)
    pitot = ((g + 1) * M * M / 2) ** (g / (g - 1)) * ((g + 1) / (2 * g * M * M - (g - 1))) ** (1 / (g - 1))
    return dict(U=M * c0, c0=c0, pitot=pitot * p0,
                rhoShock=rho0 * (g + 1) * M * M / ((g - 1) * M * M + 2),
                standoff=0.386 * math.exp(4.67 / (M * M)) * ctx.param('radius'))


def _centre(ctx: RunContext, t: float):
    cx, cy = ctx.param('start')
    return cx + bowShockStates(ctx)['U'] * t, cy


def configureScheme(ctx: RunContext) -> None:
    walledDomainBounds(ctx, *_box(ctx))
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    e = bowShockStates(ctx)
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    start, R = ctx.param('start'), ctx.param('radius')
    lo, hi = _box(ctx)

    def stateAt(x):
        return (torch.full_like(x[:, 0], rho0), torch.full_like(x[:, 0], p0), torch.zeros_like(x))

    def wallVelocity(x):
        # rows of the cylinder move, rows of the box do not
        v = torch.zeros_like(x)
        v[:, 0] = torch.where(circleSDF(x, start, R + 1e-6) <= 0, e['U'], 0.0).to(x.dtype)
        return v

    return buildWalledBox(ctx, lo, hi, stateAt, solidSDF=lambda x: circleSDF(x, start, R),
                          wallVelocity=wallVelocity)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, e, t = state.state, bowShockStates(ctx), float(state.t)
    fluid = s.kinds == 0
    out = compressibleDiagnostics(ctx, state)
    dx, R = ctx.config.dx, ctx.param('radius')
    cx, cy = _centre(ctx, t)
    x, y = s.positions[:, 0], s.positions[:, 1]
    axis = fluid & ((y - cy).abs() < dx) & (x > cx + R)
    threshold = 0.5 * (ctx.param('rho0') + e['rhoShock'])
    shocked = axis & (s.densities > threshold)
    nan = float('nan')
    out['standoffRatio'] = ((x[shocked].max() - (cx + R)) / e['standoff']).item() if shocked.any() else nan
    nose = torch.tensor([cx + R, cy], dtype=s.positions.dtype, device=s.positions.device)
    near = fluid & (torch.linalg.norm(s.positions - nose, dim=-1) < ctx.param('probeRadius') * dx)
    out['pitotRatio'] = (s.pressures[near].median() / e['pitot']).item() if near.any() else nan
    inside = circleSDF(s.positions, (cx, cy), R) < 0
    out['penetrating'] = (fluid & inside).sum().item()
    out['maxVelocity'] = torch.linalg.norm(s.velocities[fluid], dim=-1).max().item()
    return out


BOW_SHOCK_FIELDS = [
    Field('densities', 'Density', colorMap='viridis', gridResolution=768),
    Field('pressures', 'Pressure', colorMap='inferno', gridResolution=768),
]
setupPlot, updatePlot = particlePlot(BOW_SHOCK_FIELDS, figsize=(12, 4))


bowShockCase = registerCase(Case(
    name='bowShock',
    scheme='Monaghan',
    description='Cylinder at Mach 3 through gas at rest: bow shock standoff (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=compressibleTimestep,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='07-bowShock',
        dim=2,
        nx=400,
        L=2.6,
        periodic=False,
        supportMode='Gather',
        tLimit=0.6,
        cflFactor=0.2,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho0=1.4,
        p0=1.0,
        mach=3.0,
        height=1.2,
        start=(0.3, 0.0),
        radius=0.1,
        probeRadius=2.5,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(bowShockCase)
