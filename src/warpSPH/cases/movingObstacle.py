"""Flow past a spinning obstacle (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/10-moving-obstacle.ipynb`. A hexagonal rigid body
spins in a periodic box while the domain-mean velocity is driven towards
`U_target`. The hexagon is only the default: `--obstacleShape` takes any key of
:data:`~warpSPH.cases.weaklyCompressible.SHAPE_PRESETS`, with
`--obstacleAspect`, `--obstacleRotation` and `--obstacleOffset` placing it, so
the wake behind a spinning star and behind a spinning circle (which sheds
nothing until the flow separates) are the same run twice.

The forcing acts on the *mean* velocity only, deliberately: forcing every
particle towards the target would damp the wake fluctuations that are the whole
point of the case.
"""

from __future__ import annotations

from typing import Dict

from ..configurations.region import BCType
from ..enumTypes import isArtificialCompressibleScheme
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS,
                                 boundaryRegion, buildRegionSystem,
                                 configureArtificialCompressible,
                                 configureWeaklyCompressible, domainFluidSdf, fluidRegion,
                                 meanFlowForcingBC, paramExtraData, paramShapeSdf,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['movingObstacleCase']


def buildSystem(ctx: RunContext):
    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    return buildRegionSystem(ctx, [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, paramShapeSdf(ctx), kind=BCType.constant),
    ])


def configureScheme(ctx: RunContext) -> None:
    # ACSPH has no `diffusionParams`, so the WCSPH configurer would crash on
    # it -- same dispatch as `randomFlow`.
    if isArtificialCompressibleScheme(ctx.scheme):
        configureArtificialCompressible(ctx)
    else:
        configureWeaklyCompressible(ctx)


def initialConditions(ctx: RunContext, system) -> None:
    if isArtificialCompressibleScheme(ctx.scheme):
        # No sound speed to back-solve a `dt` from: seed `targetDt` and let
        # `movingObstacleTimestep`'s Eq. (46) branch take over. Eq. (48)'s
        # `U_char` is the driven mean speed, the case's own velocity scale.
        ctx.config.dt = ctx.param('targetDt')
        if ctx.schemeConfig.acParams.uChar is None:
            ctx.schemeConfig.acParams.uChar = float(abs(ctx.param('U_target')))
    else:
        setupTimestep(ctx, system)

    ctx.config.rigidBodies[0].angularVelocity = ctx.param('obstacleOmega')
    ctx.schemeConfig.rigidBodies = ctx.config.rigidBodies

    ctx.schemeConfig.boundaryConditions = [meanFlowForcingBC(
        ctx.scratch['fluidSdf'], ctx.param('U_target'), ctx.param('forcingTau'))]


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def movingObstacleTimestep(ctx: RunContext, state) -> float:
    """De Courcy et al. 2024 Eq. (46) under ACSPH; the fixed `setupTimestep`
    dt otherwise (unchanged delta-SPH behaviour, as in `randomFlowTimestep`)."""
    if isArtificialCompressibleScheme(ctx.scheme):
        from ..modules.timestep import computeTimestep
        return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)
    return ctx.config.dt


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


movingObstacleCase = registerCase(Case(
    name='movingObstacle',
    scheme='deltaSPH',
    description='Flow past a spinning rigid obstacle (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    timestep=movingObstacleTimestep,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='10-movingObstacle',
        nx=128,
        L=2.0,
        tLimit=10.0,
        # `symplecticEuler` override, not the shared `WEAKLY_COMPRESSIBLE_
        # DEFAULTS['integrationScheme']` ('rungeKutta2', still shared with
        # several incompressible-scheme cases untested under symplecticEuler)
        # -- WCSPH_DEFAULT_CLOSEOUT_PLAN.md item E. This is the case whose
        # rotating rigid body's real (nonzero, centripetal) boundary
        # acceleration was wired into `english2025.py`'s `a_b` this same
        # session (item D) -- short sanity run confirmed no divergence with
        # this integrator; not yet run at the case's own full 10s `tLimit`.
        integrationScheme='symplecticEuler',
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        # Shape/size/aspect/rotation/offset; the hexagon is what the notebook
        # used, any other key of `SHAPE_PRESETS` works.
        **dict(OBSTACLE_PARAMS, obstacleShape='hexagon'),
        obstacleOmega=1.0,
        U_target=1.0,
        forcingTau=0.5,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(movingObstacleCase)
