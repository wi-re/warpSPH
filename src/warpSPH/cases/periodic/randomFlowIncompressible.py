"""Decaying divergence-free random flow using native pressure projections."""

from __future__ import annotations

from typing import Any, Dict

from ...runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .randomFlow import BOUNDED_BAND, buildSystem, noiseVelocities
from ..common.weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 configureWeaklyCompressible, paramExtraData,
                                 weaklyCompressibleDiagnostics)
from ..common.plotting import particlePlot

__all__ = ['randomFlowIncompressibleCase']


def configureScheme(ctx: RunContext) -> None:
    if ctx.param('bounded') and not ctx.param('band'):
        ctx.spec.params['band'] = BOUNDED_BAND
    configureWeaklyCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    schemeConfig.shiftProperties.active = ctx.param('shifting')
    if ctx.param('boundaryPressureMode', 'mdbcDensity') != 'mdbcDensity':
        raise ValueError('boundaryPressureMode is retired; configure iisph.boundaryPressure instead')
    # The surface-detection bandwidth is measured in particle spacings.
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def initialConditions(ctx: RunContext, system) -> None:
    system.state.velocities[:] = noiseVelocities(ctx, system)
    # DFSPH has no acoustic term to derive a fixed `dt` from the way
    # `randomFlow`'s `setupTimestep` does -- seed the run's starting `dt`
    # directly from `targetDt` and let the `timestep` hook below take over
    # adaptively from the first step.
    ctx.config.dt = ctx.param('targetDt')


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


randomFlowIncompressibleCase = registerCase(Case(
    name='randomFlowIncompressible',
    scheme='dfsph',
    description='Decaying divergence-free random flow (2D), DFSPH, periodic or bounded.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='06-randomFlowIncompressible',
        # `kolmogorovIncompressibleTimestep` applies `cflFactor` to the
        # particle diameter, so this is Bender & Koschier's published 0.4
        # rather than the h-based 0.3 the weakly-compressible cases share.
        cflFactor=0.4,
        nx=128,
        L=2.0,
        tLimit=10.0,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        dt=1e-3,
        minDt=1e-7,
        maxDt=1e-1,
    ),
    params=dict(
        rho0=1.0,
        nu=0.0,
        targetDt=0.00025,
        band=0,
        freeSurface=False,
        # `--bounded` mirrors `randomFlow`'s 07-notebook variant; `--obstacle`
        # its circular cylinder.
        bounded=False,
        obstacle=False,
        **OBSTACLE_PARAMS,
        bandWidth=16.0,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        tileable=True,
        kind='perlin',
        seed=45906734,
        shifting=False,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(randomFlowIncompressibleCase)
