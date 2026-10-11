"""Triple-point shock interaction (2D), compressible.

The script forms of this case are
`examples/compressible/14-triplePoint/triplePoint_equalSpacing.ipynb` and
`triplePoint_equalMass.ipynb` (formerly flat notebooks in slots 14/15; merged
into one directory the way `sedov`'s 1D/2D/3D variants were). Both run the
same three-region initial state in the same 14x6 box; they differ only in how
the regions are sampled -- one particle spacing everywhere (`--no-equalMass`),
or one particle *mass* everywhere, which means the light region is sampled
sqrt(8) times coarser (`--equalMass`).

The original equal-mass notebook also carried an inline copy of
`sampleTriplePointEqualMass`, re-deriving the region masks by hand after
calling the helper. The packaged helper is used here instead; the two agree
apart from the sign of `splitY`, which is symmetric anyway.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..caseUtils import sampleTriplePointEqualMass, sampleTriplePointEqualResolution
from ..configurations.meshless import MeshlessConfig
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['triplePointCase', 'TRIPLE_POINT_FIELDS']


def configureScheme(ctx: RunContext) -> None:
    domain = ctx.config.domain
    domain.min = torch.tensor([0.0, -3.0], device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([14.0, 3.0], device=ctx.device, dtype=ctx.dtype)
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    states = dict(
        rho_I=ctx.param('rho_I'), p_I=ctx.param('p_I'),
        rho_II=ctx.param('rho_II'), p_II=ctx.param('p_II'),
        rho_III=ctx.param('rho_III'), p_III=ctx.param('p_III'),
    )
    common = dict(
        splitX=ctx.param('splitX'), splitY=ctx.param('splitY'),
        config=ctx.config, schemeConfig=ctx.schemeConfig,
        extraData=dict(ctx.spec.params),
        **states,
    )
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        from ..caseUtils.meshless import toMeshlessSystem
        from ..configurations import CompressibleSPHConfig
        from ..systems.compressibleMonaghan import CompressibleState, CompressibleSystem
        # the region sampler reads `schemeConfig.rho0` / `gamma`, which the
        # meshless scheme config does not carry
        compConfig = CompressibleSPHConfig(gamma=ctx.param('gamma'), rho0=ctx.param('rho0'))
        m = dict(common, schemeConfig=compConfig,
                 SimulationState=CompressibleState, SimulationSystem=CompressibleSystem)
    else:
        m = dict(common, SimulationState=ctx.SimulationState, SimulationSystem=ctx.SimulationSystem)
    if ctx.param('equalMass'):
        # sqrt(8) is the density ratio between regions I/III and region II, so
        # sampling region II that much coarser equalises the particle masses.
        ratio = math.sqrt(8)
        nxs = [ctx.spec.nx * ratio, ctx.spec.nx, ctx.spec.nx * ratio, ctx.spec.nx * ratio]
        system = sampleTriplePointEqualMass(nxs=nxs, **m)
    else:
        system = sampleTriplePointEqualResolution(nx=ctx.spec.nx, **m)
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        return toMeshlessSystem(system, ctx.config, ctx.schemeConfig)
    return system


#: The one panel, exported so a notebook can pass it to
#: `buildFieldPlotter`/`refreshFieldPlotter` directly (see `hydrostatic.py`).
TRIPLE_POINT_FIELDS = [
    Field('densities', 'Density field (log scale)', colorMap='RdBu',
          colorMapKind='diverging', flip=True, scaling='Logarithmic',
          gridResolution=1024, vMin=0.2, vMax=7.0,
          # The vispy default (0.04) crowds the panel title against the plot
          # box with only one panel; multi-panel cases don't need this.
          plotTitleGap=0.15),
]

setupPlot, updatePlot = particlePlot(TRIPLE_POINT_FIELDS, figsize=(12, 6))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


triplePointCase = registerCase(Case(
    name='triplePoint',
    scheme='CRKSPH',
    description='Triple-point shock interaction (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    # Only meaningful for the equal-mass sampling, where the coarse region's
    # sound speed changes a lot; equal-resolution runs pin dt below instead.
    timestep=compressibleTimestep,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='14-triplePoint',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=10.0,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        equalMass=True,
        splitX=1.0,
        splitY=1.5,
        rho_I=1.0, p_I=1.0,
        rho_II=0.125, p_II=0.1,
        rho_III=1.0, p_III=0.1,
        aspect=2.0,
        band=20,
    ),
))


if __name__ == '__main__':
    caseMain(triplePointCase)
