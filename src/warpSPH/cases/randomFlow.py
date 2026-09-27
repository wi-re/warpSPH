"""Decaying random flow (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/06-periodic-random-flow.ipynb` and
`07-bounded-random-flow.ipynb`, plus their incompressible twin
`examples/incompressible/periodic-random-flow.ipynb`. All three seed the same
divergence-free noise field; they differ only in whether the box has walls
(`--bounded`) and which scheme integrates it (`--scheme`).

The noise is divergence-free by construction, so the initial state is a valid
incompressible field and the run measures how the scheme decays it rather than
how it recovers from a bad start.
"""

from __future__ import annotations

from typing import Dict

from ..enumTypes import isArtificialCompressibleScheme
from ..modules.noise.sampleDivergenceFree import sampleDivergenceFreeNoise
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureArtificialCompressible,
                                 configureWeaklyCompressible,
                                 domainBoundarySdf, domainFluidSdf, fluidRegion,
                                 paramExtraData, paramShapeSdf, setupTimestep,
                                 weaklyCompressibleDiagnostics)

__all__ = ['randomFlowCase', 'noiseVelocities', 'BOUNDED_BAND']

#: Particle layers of wall `--bounded` gets when `--band` is left at its
#: (periodic) default of 0. The walls are cut from the *interior* domain, which
#: `band` is what widens the simulated box beyond -- so at `band=0` the wall
#: region encloses no volume and `--bounded` silently samples **zero** boundary
#: particles, i.e. runs as the periodic variant. 5 is what the 07 notebook used
#: and what `lidDrivenCavity` defaults to.
BOUNDED_BAND = 5


def configureScheme(ctx: RunContext) -> None:
    if ctx.param('bounded') and not ctx.param('band'):
        ctx.spec.params['band'] = BOUNDED_BAND
    if isArtificialCompressibleScheme(ctx.scheme):
        # ACSPH has no `diffusionParams` block, so the shared WCSPH
        # configurator would crash on it; route to the ACSPH-specific one
        # (same `configureDomain` domain/resolution work, ACSPH scheme knobs).
        configureArtificialCompressible(ctx)
    else:
        configureWeaklyCompressible(ctx)
    # The surface-detection bandwidth is measured in particle spacings.
    ctx.schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    regions = [fluidRegion(ctx, domainFluidSdf(ctx))]
    if ctx.param('bounded'):
        regions.append(boundaryRegion(ctx, domainBoundarySdf(ctx)))
    if ctx.param('obstacle'):
        regions.append(boundaryRegion(ctx, paramShapeSdf(ctx)))
    return buildRegionSystem(ctx, regions)


def noiseVelocities(ctx: RunContext, system):
    """The divergence-free Perlin field the notebooks seeded the run with."""
    return sampleDivergenceFreeNoise(
        system.state, ctx.config.domain, ctx.config, ctx.schemeConfig,
        ctx.spec.nx * 2,
        octaves=ctx.param('octaves'), lacunarity=ctx.param('lacunarity'),
        persistence=ctx.param('persistence'), baseFrequency=ctx.param('baseFrequency'),
        tileable=ctx.param('tileable'), kind=ctx.param('kind'), seed=ctx.param('seed'))


def initialConditions(ctx: RunContext, system) -> None:
    system.state.velocities[:] = noiseVelocities(ctx, system)
    if isArtificialCompressibleScheme(ctx.scheme):
        # No sound speed to back-solve a `dt` from; seed `targetDt` and let
        # `randomFlowTimestep`'s Eq. (46) branch take over from step 1 on.
        ctx.config.dt = ctx.param('targetDt')
        # `U_char` in Eq. (48), which De Courcy et al. leave undefined per case
        # (ACSPH_PLAN.md 5.5). This field has no closed-form velocity scale --
        # it is band-limited noise -- so take it from the seeded field itself,
        # which is the only honest characteristic speed available. Without it
        # the metric falls back to the instantaneous `|v|_max`, which drifts
        # with the flow and so makes a fixed `eps_v` target mean different
        # things at different times.
        if ctx.schemeConfig.acParams.uChar is None:
            fluid = system.state.kinds == 0
            if bool(fluid.any()):
                ctx.schemeConfig.acParams.uChar = float(
                    system.state.velocities[fluid].norm(dim=-1).max())
        return
    setupTimestep(ctx, system)


def randomFlowTimestep(ctx: RunContext, state) -> float:
    """Per-step dt, dispatched by scheme.

    * `artificialCompressible` -- De Courcy et al. 2024 Eq. (46)
      (`modules.timestep.computeTimestep` -> ACSPH branch).
    * everything else (the `deltaSPH` default) -- fixed `targetDt`, the
      pre-existing behaviour: this case historically had no `timestep` hook,
      so the runner left `config.dt` at whatever `setupTimestep` set.
    """
    if isArtificialCompressibleScheme(ctx.scheme):
        from ..modules.timestep import computeTimestep
        return computeTimestep(state, ctx.config, ctx.schemeConfig,
                               dt=ctx.config.dt)
    return ctx.config.dt


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


randomFlowCase = registerCase(Case(
    name='randomFlow',
    scheme='deltaSPH',
    description='Decaying divergence-free random flow (2D), periodic or bounded.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=randomFlowTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='06-randomFlow',
        nx=128,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        targetDt=0.00025,
        # `--bounded` is the 07 notebook; `--obstacle` adds the circular
        # cylinder both notebooks carried but only 06 switched on.
        bounded=False,
        obstacle=False,
        # Shape/size/aspect/rotation/offset of that cylinder -- a circle by
        # default, any other key of `SHAPE_PRESETS` on request.
        **OBSTACLE_PARAMS,
        bandWidth=16.0,
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
    caseMain(randomFlowCase)
