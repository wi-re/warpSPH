"""Woodward-Colella blast waves between two solid walls (1D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 4. The gas fills x in [0, 1] (shifted to
[-0.5, 0.5] here) with `wallLayers` wall rows beyond each end, and starts
from the same three states as `woodwardColella` (`WOODWARD_REGIONS`): two
blasts at the ends, launched against the walls as much as into the tube.

The periodic `woodwardColella` case is the exact reference: its initial state
is mirror-symmetric about x = 0 and x = 1 on [-1, 1], which is what reflecting
walls at 0 and 1 are. `scripts/compare_wallMirror.py` runs both and compares
the profiles. The domain is sized from `nx` so the fluid spacing is 1/(nx -
2 wallLayers): the default nx gives the reference's 1/1000.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from .compressibleWalls import WALL_PARAMS, buildWalledSystem
from .plotting import ProfileAxis, profilePlot
from .woodwardColella import WOODWARD_REGIONS

__all__ = ['woodwardColellaWallsCase']


def configureScheme(ctx: RunContext) -> None:
    nx, nw = ctx.spec.nx, ctx.param('wallLayers')
    L = nx / (nx - 2 * nw)
    ctx.spec = ctx.spec.merged(L=L)
    domain = ctx.config.domain
    domain.min = torch.full_like(domain.min, -0.5 * L)
    domain.max = torch.full_like(domain.max, 0.5 * L)
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    regions = ctx.param('regions')

    def stateAt(x):
        s = x + 0.5   # the tube's own coordinate, [0, 1]
        rho, p = torch.ones_like(x), torch.ones_like(x)
        for r in regions:
            inside = (s >= r['begin']) & (s < r['end'])
            rho = torch.where(inside, torch.full_like(x, r['density']), rho)
            p = torch.where(inside, torch.full_like(x, r['pressure']), p)
        return rho, p, torch.zeros_like(x)

    return buildWalledSystem(ctx, stateAt)


setupPlot, updatePlot, drawWoodwardColellaWalls = profilePlot(
    [
        ProfileAxis('densities', 'Density'),
        ProfileAxis('internalEnergies', 'Internal energy'),
        ProfileAxis('pressures', 'Pressure'),
        ProfileAxis('velocities', 'Velocity', component=0),
    ],
    shape=(2, 2), figsize=(9, 6), xlim=(-0.5, 0.5),
)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s = state.state
    fluid = s.kinds == 0
    lo, hi = -0.5, 0.5
    out = compressibleDiagnostics(ctx, state)
    x = s.positions[fluid, 0]
    out['penetrating'] = ((x < lo) | (x > hi)).sum().item()
    out['maxVelocity'] = s.velocities[fluid].abs().max().item()
    return out


woodwardColellaWallsCase = registerCase(Case(
    name='woodwardColellaWalls',
    scheme='CRKSPH',
    description='Woodward-Colella blast waves between solid walls (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    timestep=compressibleTimestep,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='04-woodwardColellaWalls',
        dim=1,
        nx=1032,
        L=1.032,
        periodic=False,
        tLimit=0.038,
        cflFactor=0.2,
        plotInterval=50,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        regions=WOODWARD_REGIONS,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(woodwardColellaWallsCase)
