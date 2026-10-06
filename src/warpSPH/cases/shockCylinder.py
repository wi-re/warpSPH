"""A planar shock hitting a fixed cylinder in a closed channel (2D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 6: the curved wall. A Mach-`mach` shock runs
right through gas at rest (`rho0`, `p0`) towards a cylinder of radius
`radius` at `centre`. Behind the shock the gas is the Rankine-Hugoniot state 2
moving at `u2`, kept that way by the left wall moving at `u2` (a piston), so no
rarefaction follows the shock. The channel's other walls are fixed.

Exact numbers the run is judged on, at the cylinder's leading point:

* right after impact the reflection there is normal, so the pressure is the
  reflected state p3 of `shockReflection`;
* it then relaxes to the stagnation pressure of the post-shock flow,
  `p2 (1 + (g-1)/2 M2^2)^(g/(g-1))` (state 2 is subsonic for Mach < ~2.07).

`stagnationPressureRatio` is the median pressure of the fluid rows within
`probeRadius` spacings of the leading point, over p3.
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
from .shockReflection import reflectionStates

__all__ = ['shockCylinderCase', 'cylinderStates']


def _box(ctx: RunContext):
    return (0.0, -0.5 * ctx.param('height')), (ctx.spec.L, 0.5 * ctx.param('height'))


def cylinderStates(ctx: RunContext) -> Dict[str, float]:
    e = reflectionStates(ctx)
    g = ctx.param('gamma')
    c2 = math.sqrt(g * e['p2'] / e['rho2'])
    M2 = e['u2'] / c2
    e['stagnation'] = e['p2'] * (1 + 0.5 * (g - 1) * M2 * M2) ** (g / (g - 1))
    e['M2'] = M2
    return e


def configureScheme(ctx: RunContext) -> None:
    walledDomainBounds(ctx, *_box(ctx))
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    e, x0 = cylinderStates(ctx), ctx.param('shockStart')
    lo, hi = _box(ctx)

    def stateAt(x):
        post = x[:, 0] < x0
        rho = torch.where(post, e['rho2'], e['rho1']).to(x.dtype)
        p = torch.where(post, e['p2'], e['p1']).to(x.dtype)
        v = torch.zeros_like(x)
        v[:, 0] = torch.where(post, e['u2'], 0.0).to(x.dtype)
        return rho, p, v

    def wallVelocity(x):
        v = torch.zeros_like(x)
        v[:, 0] = torch.where(x[:, 0] < lo[0], e['u2'], 0.0).to(x.dtype)   # the left wall is the piston
        return v

    return buildWalledBox(ctx, lo, hi, stateAt,
                          solidSDF=lambda x: circleSDF(x, ctx.param('centre'), ctx.param('radius')),
                          wallVelocity=wallVelocity)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, e = state.state, cylinderStates(ctx)
    fluid = s.kinds == 0
    out = compressibleDiagnostics(ctx, state)
    dx = ctx.config.dx
    cx, cy = ctx.param('centre')
    lead = torch.tensor([cx - ctx.param('radius'), cy], dtype=s.positions.dtype, device=s.positions.device)
    near = fluid & (torch.linalg.norm(s.positions - lead, dim=-1) < ctx.param('probeRadius') * dx)
    nan = torch.full((), float('nan'), dtype=s.pressures.dtype, device=s.pressures.device)
    out['stagnationPressureRatio'] = (s.pressures[near].median() / e['p3']).item() if near.any() else nan.item()
    out['stagnationOverSteady'] = (s.pressures[near].median() / e['stagnation']).item() if near.any() else nan.item()
    inside = circleSDF(s.positions, ctx.param('centre'), ctx.param('radius')) < 0
    lo, hi = _box(ctx)
    x = s.positions
    outsideBox = (x[:, 1] < lo[1]) | (x[:, 1] > hi[1]) | (x[:, 0] > hi[0])
    out['penetrating'] = (fluid & (inside | outsideBox)).sum().item()
    out['maxVelocity'] = torch.linalg.norm(s.velocities[fluid], dim=-1).max().item()
    return out


SHOCK_CYLINDER_FIELDS = [
    Field('densities', 'Density', colorMap='viridis', gridResolution=768),
    Field('pressures', 'Pressure', colorMap='inferno', gridResolution=768),
]
setupPlot, updatePlot = particlePlot(SHOCK_CYLINDER_FIELDS, figsize=(12, 4))


shockCylinderCase = registerCase(Case(
    name='shockCylinder',
    scheme='Monaghan',
    description='Planar shock hitting a fixed cylinder in a channel (2D), compressible SPH.',
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
        caseName='06-shockCylinder',
        dim=2,
        nx=200,
        L=2.0,
        periodic=False,
        supportMode='Gather',
        tLimit=0.5,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho0=1.0,
        p0=1.0,
        mach=2.0,
        shockStart=0.4,
        height=1.0,
        centre=(0.8, 0.0),
        radius=0.15,
        probeRadius=2.5,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(shockCylinderCase)
