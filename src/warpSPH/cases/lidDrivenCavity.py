"""Lid-driven cavity (2D), weakly compressible.

The script form of this case was `examples/weaklyCompressible/09-LDC.ipynb`.
A no-slip box whose top wall is dragged sideways at unit speed, imposed as a
dynamic Dirichlet condition on the velocity rather than as a moving boundary
region.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..configurations import BoundaryCondition, BoundaryConditionType
from ..configurations.region import BCType
from ..modules import enforceDirichlet
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 domainBoundarySdf, domainFluidSdf, fluidRegion,
                                 paramExtraData, setupTimestep,
                                 weaklyCompressibleDiagnostics)

__all__ = ['lidDrivenCavityCase']


def buildSystem(ctx: RunContext):
    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    return buildRegionSystem(ctx, [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, domainBoundarySdf(ctx), kind=BCType.noSlip),
    ])


def initialConditions(ctx: RunContext, system) -> None:
    setupTimestep(ctx, system)

    lidHeight = ctx.param('lidHeight')
    lidVelocity = ctx.param('lidVelocity')
    regularizeLid = ctx.param('regularizeLid', False)
    halfWidth = ctx.spec.L / 2.0

    def lidDirichlet(state, config, schemeConfig, positions, d, n, t, dt):
        velocities = state.velocities.clone()
        if regularizeLid:
            # Botella & Peyret 1998's regularized-cavity lid profile,
            # `u(x) = lidVelocity * (1 - x^2)^2` on the interior half-width --
            # zero velocity *and* zero slope at the two top corners, instead
            # of the hard step every other particle above `lidHeight` gets.
            # The un-regularized cavity's corner is a genuine singularity
            # (present in the continuum solution too, not just this
            # discretization): default off so no existing run changes: RK2/
            # RK4 already handle it, and this exists for schemes that don't
            # (`--integrationScheme symplecticEuler`).
            xi = torch.clamp(positions[:, 0] / halfWidth, -1.0, 1.0)
            target = lidVelocity * (1.0 - xi ** 2) ** 2
        else:
            target = torch.full_like(positions[:, 0], lidVelocity)
        velocities[:, 0] = torch.where(positions[:, 1] > lidHeight, target,
                                       velocities[:, 0])
        return velocities

    ctx.schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=ctx.scratch['fluidSdf'],
        dirichletFunctions={'velocities': lidDirichlet},
    )]
    enforceDirichlet(system, system.t, ctx.config.dt, ctx.config, ctx.schemeConfig)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


lidDrivenCavityCase = registerCase(Case(
    name='ldc',
    scheme='deltaSPH',
    description='Lid-driven cavity (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='09-lidDrivenCavity',
        nx=128,
        L=2.0,
        # Long enough for the primary vortex to reach steady state.
        tLimit=30.0,
        # `symplecticEuler` override, not the shared `WEAKLY_COMPRESSIBLE_
        # DEFAULTS['integrationScheme']` ('rungeKutta2', still shared with
        # several incompressible-scheme cases untested under symplecticEuler)
        # -- WCSPH_DEFAULT_CLOSEOUT_PLAN.md item E. This is the exact case
        # `symplecticEuler`'s own boundary-position-drift bug was found and
        # fixed in (`DELTASPH_VALIDATION_PLAN.md` §9.2, `systems/
        # weaklyCompressible.py`'s `finalize` now restores boundary/ghost
        # positions every step) -- see WCSPH_DEFAULT_CLOSEOUT_PLAN.md item F
        # for the full-`tLimit` regression run this flip is gated on.
        integrationScheme='symplecticEuler',
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        band=5,
        lidHeight=1.0,
        lidVelocity=1.0,
        regularizeLid=False,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(lidDrivenCavityCase)
