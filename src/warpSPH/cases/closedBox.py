"""Closed box of gas at rest between two solid walls (1D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 1. Uniform gas, v = 0, with `wallLayers`
wall particles (`kinds == 1`) on each side: the exact solution is "nothing
happens", so `maxVelocity`, `densityError` and `pressureError` (fluid rows
only) are pure wall error, and mass/energy are conserved exactly.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleTimestep, configureCompressible,
                           paramExtraData)
from .compressibleWalls import WALL_PARAMS, buildWalledSystem, nearWallRows
from .plotting import ProfileAxis, profilePlot

__all__ = ['closedBoxCase']


def buildSystem(ctx: RunContext):
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    return buildWalledSystem(ctx, lambda x: (rho0 * torch.ones_like(x), p0 * torch.ones_like(x),
                                             torch.zeros_like(x)))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s = state.state
    fluid = s.kinds == 0
    m, v = s.masses[fluid], s.velocities[fluid]
    kinetic = 0.5 * (m * (v ** 2).sum(-1)).sum()
    thermal = (m * s.internalEnergies[fluid]).sum()
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    near = nearWallRows(ctx, s)
    out = dict(
        kineticEnergy=kinetic, thermalEnergy=thermal, totalEnergy=kinetic + thermal,
        maxVelocity=v.abs().max(),
        densityError=((s.densities[fluid] - rho0).abs().max() / rho0),
        pressureError=((s.pressures[fluid] - p0).abs().max() / p0),
        wallPressureError=((s.pressures[near] - p0).abs().max() / p0),
    )
    return {k: t.detach().cpu().item() for k, t in out.items()}


setupPlot, updatePlot, drawClosedBox = profilePlot(
    [
        ProfileAxis('densities', 'Density'),
        ProfileAxis('pressures', 'Pressure'),
        ProfileAxis('velocities', 'Velocity', component=0),
    ],
    shape=(1, 3), figsize=(10, 4), xlim=(-0.5, 0.5),
)


closedBoxCase = registerCase(Case(
    name='closedBox',
    scheme='Monaghan',
    description='Quiescent gas in a closed box with solid walls (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=compressibleTimestep,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='01-closedBox',
        dim=1,
        nx=200,
        L=1.0,
        periodic=False,
        supportMode='Gather',
        tLimit=2.0,
        plotInterval=25,
        storeInterval=100,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        rho0=1.0,
        p0=1.0,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(closedBoxCase)
