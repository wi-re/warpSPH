"""Woodward-Colella blast-wave interaction (1D), compressible.

The script form of this case was
`examples/compressible/05-woodward-colella.ipynb`. Two strong blasts launched
from opposite ends of the tube collide near x = 0.7; that collision is why the
run needs the adaptive `timestep` hook and a tighter CFL factor than the other
1D examples.
"""

from __future__ import annotations

from typing import Dict

from ...runner import Case, RunContext, caseMain, registerCase
from ...sample.compressible import sampleShockRegions1D
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from ..common.plotting import ProfileAxis, profilePlot

__all__ = ['woodwardColellaCase', 'WOODWARD_REGIONS', 'drawWoodwardColella']

#: The three initial states, as `sampleShockRegions1D` wants them. Positions are
#: |x|, so each entry is mirrored about the origin.
WOODWARD_REGIONS = [
    {'begin': 0.0, 'end': 0.1, 'pressure': 1000.0, 'density': 1.0},
    {'begin': 0.1, 'end': 0.9, 'pressure': 0.1, 'density': 1.0},
    {'begin': 0.9, 'end': 1.9, 'pressure': 100.0, 'density': 1.0},
]


def buildSystem(ctx: RunContext):
    return sampleShockRegions1D(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                                ctx.SimulationState, ctx.SimulationSystem,
                                ctx.param('regions'))


setupPlot, updatePlot, drawWoodwardColella = profilePlot(
    [
        ProfileAxis('densities', 'Density'),
        ProfileAxis('internalEnergies', 'Internal energy'),
        ProfileAxis('pressures', 'Pressure'),
        ProfileAxis('soundspeeds', 'Sound speed'),
    ],
    shape=(2, 2), figsize=(9, 6), xlim=(0, 1),
)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


woodwardColellaCase = registerCase(Case(
    name='woodwardColella',
    scheme='CRKSPH',
    description='Woodward-Colella interacting blast waves (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    timestep=compressibleTimestep,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='05-woodwardColella',
        dim=1,
        nx=2000,
        L=2.0,
        tLimit=0.038,
        cflFactor=0.2,
        plotInterval=50,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        regions=WOODWARD_REGIONS,
    ),
))


if __name__ == '__main__':
    caseMain(woodwardColellaCase)
