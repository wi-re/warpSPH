"""Kolmogorov flow (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/08-kolmogorov-flow.ipynb`. A periodic box driven by
a sinusoidal body force in x, with a small noise component in y to break the
symmetry so the shear layers can actually go unstable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ...configurations import BoundaryCondition, BoundaryConditionType
from ...modules.noise.sampleDivergenceFree import generateNoiseInterpolator
from ...runner import Case, RunContext, caseMain, registerCase
from ...utils import buildDomainDescription
from ...math import getPeriodicPositions
from ..common.plotting import particlePlot
from ..common.weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 configureWeaklyCompressible, domainFluidSdf,
                                 fluidRegion, paramExtraData, paramShapeSdf,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['kolmogorovCase']


def buildSystem(ctx: RunContext):
    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    regions = [fluidRegion(ctx, fluidSdf)]
    if ctx.param('obstacle'):
        from ..common.weaklyCompressible import boundaryRegion
        regions.append(boundaryRegion(ctx, paramShapeSdf(ctx)))
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    setupTimestep(ctx, system)

    # The noise interpolator is built on CPU because it is evaluated with
    # numpy-side Perlin sampling; only its output crosses to the device.
    domainCpu = buildDomainDescription(
        ctx.spec.L + ctx.config.dx * ctx.param('band') * 2, ctx.spec.dim,
        ctx.spec.periodic, torch.device('cpu'), ctx.dtype)
    noise = generateNoiseInterpolator(
        ctx.spec.nx * 2, ctx.spec.nx * 2, domainCpu, dim=ctx.config.domain.dim,
        octaves=ctx.param('octaves'), lacunarity=ctx.param('lacunarity'),
        persistence=ctx.param('persistence'), baseFrequency=ctx.param('baseFrequency'),
        tileable=ctx.param('tileable'), kind=ctx.param('kind'), seed=ctx.param('seed'))

    domain = ctx.config.domain
    xi = ctx.param('xi')
    k = ctx.param('k')
    noiseLevel = ctx.param('noiseLevel')

    def forcing(state, config, schemeConfig, x, d, n, t, dt):
        positions = getPeriodicPositions(x, domain)
        u_x = xi * torch.sin(k * np.pi * positions[:, 1])
        u_y = noise(positions.detach().cpu()).to(dtype=x.dtype, device=x.device) * noiseLevel
        return torch.stack([u_x, u_y], dim=1) * state.masses.unsqueeze(1)

    ctx.schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=ctx.scratch['fluidSdf'],
        forcingFunctions=[forcing],
    )]


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


kolmogorovCase = registerCase(Case(
    name='kolmogorov',
    scheme='deltaSPH',
    description='Kolmogorov flow (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='08-kolmogorovFlow',
        nx=128,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        xi=1.0,
        k=4,
        noiseLevel=0.01,
        obstacle=False,
        # Shape/size/aspect/rotation/offset of that obstacle -- a circle by
        # default, any other key of `SHAPE_PRESETS` on request.
        **OBSTACLE_PARAMS,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        tileable=True,
        kind='perlin',
        seed=45906734,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(kolmogorovCase)
