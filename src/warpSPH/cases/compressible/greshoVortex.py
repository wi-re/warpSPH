"""Gresho-Chan vortex (2D), compressible.

The script form of this case was
`examples/compressible/09-gresho-chan-vortex.ipynb`. A steady rotating vortex
balanced by its own pressure gradient: the exact solution is time-independent,
so any drift in the velocity profile is scheme error.
"""

from __future__ import annotations

from typing import Dict

from ...caseUtils import sampleGreshoVortex, sampleGreshoVortexMeshless
from ...configurations.meshless import MeshlessConfig
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import Field, particlePlot

from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS, COMPRESSIBLE_SCHEMES, SPH_SCHEMES
from ...runner.parameters import Parameter as P

__all__ = ['greshoVortexCase', 'GRESHO_FIELDS']


def buildSystem(ctx: RunContext):
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        return sampleGreshoVortexMeshless(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                                          cuspWidth=ctx.param('cuspWidth'),
                                          jitter=ctx.param('jitter', 0.0))
    return sampleGreshoVortex(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                              ctx.SimulationState, ctx.SimulationSystem,
                              cuspWidth=ctx.param('cuspWidth'),
                              jitter=ctx.param('jitter', 0.0))


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
    supportedSchemes=COMPRESSIBLE_SCHEMES,
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'cuspWidth': P('number', 'Width of the smoothed Gresho cusp; zero uses the piecewise standard profile.', 'initial', units='length', minimum=0),
        'jitter': P('number', 'Initial displacement as a fraction of lattice spacing.', 'sampling', minimum=0),
    },
    simulationParameters={'dim': P('integer', 'Dimension supported by this initial-condition sampler.', 'domain', choices=(2,))},
    variants={'smoothed': {'params': {'cuspWidth': 0.02}}},
    params=dict(COMPRESSIBLE_PARAMS, markerSize=4, cuspWidth=0.0, jitter=0.0),
))


if __name__ == '__main__':
    caseMain(greshoVortexCase)
