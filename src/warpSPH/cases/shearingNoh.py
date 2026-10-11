"""Shearing Noh implosion (2D), compressible.

The script form of this case was
`examples/compressible/11-shearing-noh-implosion-2d.ipynb`. A Noh implosion
with a transverse shear of amplitude `vs` superimposed, so the converging shock
has to survive a velocity discontinuity it is not aligned with.
"""

from __future__ import annotations

from typing import Dict

from ..caseUtils import sampleShearingNoh
from ..configurations.meshless import MeshlessConfig
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['shearingNohCase', 'SHEARING_NOH_FIELDS']


def buildSystem(ctx: RunContext):
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        from ..caseUtils.meshless import toMeshlessSystem
        from ..systems.compressibleMonaghan import CompressibleState, CompressibleSystem
        comp = sampleShearingNoh(ctx.param('vs'), ctx.spec.nx, ctx.config, ctx.schemeConfig,
                                 dict(ctx.spec.params), CompressibleState, CompressibleSystem)
        return toMeshlessSystem(comp, ctx.config, ctx.schemeConfig)
    return sampleShearingNoh(ctx.param('vs'), ctx.spec.nx, ctx.config, ctx.schemeConfig,
                             dict(ctx.spec.params), ctx.SimulationState, ctx.SimulationSystem)


#: The two panels, exported so a notebook can pass them to
#: `buildFieldPlotter`/`refreshFieldPlotter` directly (see `hydrostatic.py`).
SHEARING_NOH_FIELDS = [
    # x-velocity rather than its magnitude: the shear is what this case is for.
    Field('velocities', 'velocities', colorMap='viridis', mapping='x'),
    Field('densities', 'densities', colorMap='cividis', flip=True, gridResolution=512),
]

setupPlot, updatePlot = particlePlot(SHEARING_NOH_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


shearingNohCase = registerCase(Case(
    name='shearingNoh',
    scheme='CRKSPH',
    description='Shearing Noh implosion (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='11-shearingNohImplosion',
        dim=2,
        nx=200,
        L=2.0,
        tLimit=0.6,
        # `sampleShearingNoh` leaves dt unset; this is the notebook's value.
        dt=2.5e-4,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(COMPRESSIBLE_PARAMS, vs=5.0),
))


if __name__ == '__main__':
    caseMain(shearingNohCase)
