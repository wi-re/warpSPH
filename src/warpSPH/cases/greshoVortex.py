"""Gresho-Chan vortex (2D), compressible.

The script form of this case was
`examples/compressible/09-gresho-chan-vortex.ipynb`. A steady rotating vortex
balanced by its own pressure gradient: the exact solution is time-independent,
so any drift in the velocity profile is scheme error.
"""

from __future__ import annotations

from typing import Dict

from ..caseUtils import sampleGreshoVortex
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['greshoVortexCase', 'GRESHO_FIELDS']


def buildSystem(ctx: RunContext):
    return sampleGreshoVortex(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                              ctx.SimulationState, ctx.SimulationSystem,
                              cuspWidth=ctx.param('cuspWidth'))


#: The two panels, exported so a notebook can pass them to
#: `buildFieldPlotter`/`refreshFieldPlotter` directly (see `hydrostatic.py`).
GRESHO_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('pressures', 'pressures', colorMap='cividis', flip=True),
]

setupPlot, updatePlot = particlePlot(GRESHO_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


greshoVortexCase = registerCase(Case(
    name='gresho',
    scheme='CRKSPH',
    description='Gresho-Chan rotating vortex (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='09-greshoChanVortex',
        dim=2,
        nx=200,
        L=1.0,
        tLimit=3.0,
        plotInterval=1,
        storeInterval=500,
    ),
    # cuspWidth > 0: smoothed-cusp variant (see `greshoProfile`); 0 = the standard profile
    params=dict(COMPRESSIBLE_PARAMS, markerSize=4, cuspWidth=0.0),
))


if __name__ == '__main__':
    caseMain(greshoVortexCase)
