"""Hydrostatic column: gravity balance, pressure profile, density and wall
penetration diagnostics. Native solver qualification and prepared initial
states are documented in docs/incompressible; historical legacy outcomes
are not expectations for these implementations."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ...configurations.moduleConfigurations.gravity import GravityType
from ...configurations.region import BCType
from ...enumTypes import IncompressibleSPHScheme, ViscosityTerm
from ...modules import shuffleParticles
from ...runner import Case, RunContext, caseMain, registerCase
from ..periodic.kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from ..common.plotting import Field, particlePlot
from ..periodic.randomFlow import BOUNDED_BAND
from ..common.weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureArtificialCompressible,
                                 configureWeaklyCompressible,
                                 domainBoundarySdf, filletedDomainBoundarySdf,
                                 fluidRegion, shapeSdf,
                                 particleDistributionMetrics)
from ...enumTypes import isArtificialCompressibleScheme

__all__ = ['hydrostaticColumnCase']


def columnSdf(ctx: RunContext):
    """The fluid box: full interior width, bottom-anchored, `fillRatio` deep."""
    L = ctx.spec.L
    fill = ctx.param('fillRatio')
    halfHeight = 0.5 * fill * L
    centre = [0.0, -0.5 * L + halfHeight]
    return shapeSdf('box', args=[[0.5 * L, halfHeight]], offset=centre)


def configureScheme(ctx: RunContext) -> None:
    # The container walls are the point of the case: with `band` at its
    # (periodic) default of 0 the wall region encloses no volume and the case
    # would silently run as a free-falling column, so fall back to
    # `randomFlow`'s bounded band exactly the way `randomFlow --bounded` does.
    if not ctx.param('band'):
        ctx.spec.params['band'] = BOUNDED_BAND

    if isArtificialCompressibleScheme(ctx.scheme):
        return _configureArtificialCompressible(ctx)

    configureWeaklyCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = True
    # `configureWeaklyCompressible` wires the shared `inviscid`/`alpha`
    # (artificial-viscosity) knobs `deltaSPH` uses; DFSPH has no such term
    # (see `kolmogorovIncompressible`), so this scheme runs with a plain
    # physical viscosity instead.
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    # which physical-viscosity term (`ViscosityTerm` name): the historical
    # normal-projected `monaghanGingold`, or `morris1997` (OPEN_PROBLEMS.md §7)
    schemeConfig.diffusionParams.viscousTerm = ViscosityTerm[ctx.param('viscousTerm', 'monaghanGingold')]
    schemeConfig.shiftProperties.active = ctx.param('shifting', False)
    # Post-solve fluid XSPH velocity smoother in `divergenceFree_step`, in units of
    # omniSPH's own `XSPH_FLUID = 0.05` (DFSPH_FINDINGS.md 1.16). The residual
    # `|v|` this case shows under `divergenceFree` is a bounded, undamped
    # free-surface limit cycle; this otherwise-inviscid scheme has no other
    # sink for it. `1.0` takes nx=64 KE 3e-5 -> 5e-7 and nx=128 KE 7e-4 ->
    # 3e-7 (`densityP05` / `embeddedMinDensity` -> 1.000 at nx=128) for a ~3%
    # `densityP05` cost at nx=64. Default 0.0 -- opt-in, and only this case.
    schemeConfig.gravityConfig.active = True
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.direction = ctx.param('gravityDirection')
    # The surface-detection bandwidth is measured in particle spacings.
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def _configureArtificialCompressible(ctx: RunContext) -> None:
    """The ACSPH branch of `configureScheme`.

    This is the paper's own Sec. 4.1.1 case and the first thing the scheme is
    validated on (`ACSPH_PLAN.md` Part 7): a column at rest under gravity, whose
    exact solution is `v = 0` with `p` linear in depth. It is the *operator*
    discriminator -- AC-2L holds that linear gradient and AC-2 provably cannot
    -- which is why the pressure-slope and pressure-residual diagnostics below
    are the figures of merit and not just health checks.

    Differences from the DFSPH branch, all structural rather than tuning:
    ACSPH has no `diffusionParams` (its viscosity is `acParams.nu`), no XSPH
    filter, and no equation of state. Shifting stays off until the Michel
    et al. law lands (`ACSPH_PLAN.md` step 7) -- the existing delta+ shift is
    Mach-scaled, and ACSPH has no Mach number.
    """
    configureArtificialCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = True
    schemeConfig.gravityConfig.active = True
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.direction = ctx.param('gravityDirection')
    schemeConfig.shiftProperties.active = False
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx
    # Eq. (48)'s U_char. The paper never defines it per case (ACSPH_PLAN.md
    # Sec. 5.5); for a column at rest the only velocity scale in the problem is
    # sqrt(g H), the free-fall speed over the column depth.
    if schemeConfig.acParams.uChar is None:
        depth = ctx.param('fillRatio') * ctx.spec.L
        schemeConfig.acParams.uChar = float(
            (ctx.param('gravityMagnitude') * depth) ** 0.5)


def columnTimestep(ctx: RunContext, state) -> float:
    """Eq. (46) for ACSPH, the DFSPH advective/viscous CFL otherwise.

    `kolmogorovIncompressibleTimestep` exists because the generic dispatcher
    used to send everything non-weakly-compressible down the fully compressible
    formula; ACSPH now has its own branch there, and it is Eq. (46) with the
    BDF2 step-ratio clamp, which that DFSPH formula does not have.
    """
    if isArtificialCompressibleScheme(ctx.scheme):
        from ...modules.timestep import computeTimestep
        return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)
    return kolmogorovIncompressibleTimestep(ctx, state)


def buildSystem(ctx: RunContext):
    # `wallBC` selects the container's boundary condition: `freeSlip` (the
    # historical default -- reflect the normal component, no tangential drag)
    # or `noSlip` (mirror the fluid velocity -> a real viscous wall once
    # `nu > 0`, `DFSPH_FINDINGS.md` 1.14). Both go through
    # `computeBoundaryVelocities`; the schemes pick it up in step 2.
    wallBC = BCType[ctx.param('wallBC')]
    # `cornerFilletRadius` (0 = off, the default -- plain sharp corners,
    # unchanged behaviour): rounds the bottom-left/right corners the fluid
    # column actually touches, in units of `dx`. See
    # `filletedDomainBoundarySdf`'s own docstring for why -- the
    # periodic-vs-walled A/B this option exists to follow up on.
    filletRadiusDx = ctx.param('cornerFilletRadius')
    if filletRadiusDx:
        boundarySdf = filletedDomainBoundarySdf(
            ctx, radius=float(filletRadiusDx) * ctx.config.dx, dx=ctx.config.dx)
    else:
        boundarySdf = domainBoundarySdf(ctx)
    regions = [fluidRegion(ctx, columnSdf(ctx)),
               boundaryRegion(ctx, boundarySdf, kind=wallBC)]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    # The runner's velocity alarm: a column at rest has no velocity scale but
    # the free-fall speed over its depth (the same sqrt(g H) as ACSPH's U_char).
    ctx.velocityScale = float((ctx.param('gravityMagnitude') * ctx.param('fillRatio') * ctx.spec.L) ** 0.5)
    ctx.velocityScaleSource = 'free-fall speed sqrt(g H)'

    # Jitter the fluid only -- see the module docstring for why the
    # constant-density pre-relaxation is not run on a free-surface state.
    particles = system.state
    # particles.positions = shuffleParticles(
    #     particles, ctx.config, ctx.schemeConfig, 0,
    #     jitterAmount=ctx.param('jitter'))
    particles.velocities[:] = 0.0

    # Use the shared kernel-normalization calibration, preserving sampled mass.
    # All projection families receive the same physical hydrostatic profile in
    # the pressure gauge actually consumed by their selected solve route.
    if ctx.param('calibrateRestDensity'):
        from ..common.weaklyCompressible import calibrateRestDensity
        calibrateRestDensity(ctx, system, verbose=ctx.spec.verbose)

    from ...caseUtils.projection import initializeHydrostaticPressure
    if isinstance(ctx.scheme, IncompressibleSPHScheme):
        initializeHydrostaticPressure(ctx, system)
    else:
        # ACSPH needs the absolute hydrostatic field as its integrated pressure.
        particles = system.state
        fluid = particles.kinds == 0
        surface = particles.positions[fluid, 1].max()
        particles.pressures = (ctx.schemeConfig.fluid.restDensity*ctx.param('gravityMagnitude')
                               *(surface-particles.positions[:, 1]).clamp_min(0))

    ctx.scratch['initialPositions'] = particles.positions.clone()


def hydrostaticDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0
    positions = particles.positions[fluid]
    velocities = particles.velocities[fluid]
    masses = particles.masses[fluid]
    densities = particles.densities[fluid]

    d = {
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'kineticEnergy': (0.5 * masses * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'densityStd': densities.std().detach().cpu().item(),
    }

    # Spray-robust density FOMs (plan item 1 / Part 33). The plain `minDensity`
    # on this case is dominated by 1-3 fluid particles thrown 1-3 dx above the
    # free surface by the bulk slosh: isolated, reading low by kernel deficiency,
    # never the same particles two samples running (measured in
    # `probe_dfsphReferenceColumnSurface.py`). Cosmetic ballistic spray, not
    # structural loss. These two grade the column body instead:
    #   `densityP05`         -- 5th percentile of the fluid density, so a
    #                           handful of low outliers cannot move it;
    #   `embeddedMinDensity` -- the min over fluid rows more than 1 dx below the
    #                           95th-percentile surface height, i.e. with the
    #                           ballistic skin removed outright.
    dx = ctx.config.dx
    fluidY = positions[:, 1]
    surfY95 = torch.quantile(fluidY, 0.95)
    d['densityP05'] = torch.quantile(densities, 0.05).detach().cpu().item()
    embedded = fluidY < surfY95 - dx
    if bool(embedded.any()):
        d['embeddedMinDensity'] = densities[embedded].min().detach().cpu().item()

    initial = ctx.scratch.get('initialPositions')
    if initial is not None:
        disp = positions - initial[fluid]
        d['dispRms'] = torch.sqrt((disp ** 2).sum(dim=-1).mean()).detach().cpu().item()
        d['dispMax'] = torch.linalg.norm(disp, dim=-1).max().detach().cpu().item()
    d.update(particleDistributionMetrics(ctx, state))

    pressures = particles.pressures
    if pressures is None:
        # Before the first step the divergence-free solve has not run yet.
        return d

    L = ctx.spec.L
    dx = ctx.config.dx
    g = ctx.param('gravityMagnitude')
    rho0 = ctx.schemeConfig.fluid.restDensity
    surfaceY = positions[:, 1].max()
    bulk = ((positions[:, 1] < surfaceY - ctx.param('bulkMargin') * dx)
            & (positions[:, 1] > -0.5 * L + ctx.param('wallMargin') * dx))
    if bulk.sum() < 8:
        return d

    y = positions[bulk, 1]
    p = pressures[fluid][bulk]
    # Least-squares line p = a + b y.
    yBar, pBar = y.mean(), p.mean()
    denom = ((y - yBar) ** 2).sum()
    # `denom == 0` (a single bulk y-level, e.g. a very coarse column) collapses
    # the fit -- fall back to a zero slope, and keep `b` a tensor so the
    # `.detach()` calls below still work.
    b = (((y - yBar) * (p - pBar)).sum() / denom if denom > 0
         else torch.zeros((), device=y.device, dtype=y.dtype))
    a = pBar - b * yBar
    residual = p - (a + b * y)
    # The analytic gradient: dp/dy = -rho0 g for gravity (0, -g).
    d['pressureSlope'] = b.detach().cpu().item()
    d['pressureSlopeRatio'] = (b / (-rho0 * g)).detach().cpu().item()
    d['pressureResidual'] = (
        residual.pow(2).mean().sqrt() / (rho0 * g * ctx.param('fillRatio') * L)
    ).detach().cpu().item()
    return d


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm', boundary = 'Visualize'),
    # Pressure is the point of the case; span the hydrostatic range the
    # defaults imply (rho0 g H = 9.81 * 0.5, gauge-centred, so ~ +/- 2.5).
    Field('densities', 'densities', colorMap='viridis', boundary = 'Visualize'),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in hydrostaticColumnCase.params}


hydrostaticColumnCase = registerCase(Case(
    name='hydrostaticColumn',
    scheme='dfsph',
    description='Hydrostatic column in a walled box (2D): at rest under gravity, '
                'pressure linear in depth.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=hydrostaticDiagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=columnTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='07-hydrostaticColumn',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=1.0,
        periodic=True,
        # `kolmogorovIncompressibleTimestep` applies `cflFactor` to the
        # particle diameter, so this is Bender & Koschier's published 0.4,
        # as on the other adaptive incompressible cases.
        cflFactor=0.4,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        dt=1e-3,
        minDt=1e-8,
        maxDt=1e-2,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        # The column fills the bottom half of the box: a 0.5-deep column in a
        # 1.0 box, hydrostatic pressure drop rho0 g H = 9.81 * 0.5.
        fillRatio=0.5,
        gravityMagnitude=9.81,
        gravityDirection=[0.0, -1.0],
        bandWidth=16.0,
        shifting=False,
        # Part 33: normalise the fluid mass so the at-rest bulk summation
        # density lands on `rho0` (see `initialConditions`). Off keeps the
        # historical sampling; `dfsphReference` probes turn it on to stop the
        # step-1 IC-seed cold start.
        calibrateRestDensity=False,
        # Container boundary condition (`BCType` name): `freeSlip` (default,
        # historical) or `noSlip` (a viscous no-slip wall once `nu > 0` --
        # decays the free-slip bulk slosh, `DFSPH_FINDINGS.md` 1.14).
        wallBC='freeSlip',
        # Physical-viscosity term (`ViscosityTerm` name) for the `nu` above:
        # `monaghanGingold` (default, normal-projected) or `morris1997`
        # (shear-carrying, OPEN_PROBLEMS.md §7).
        viscousTerm='monaghanGingold',
        # Post-solve fluid XSPH scale for `divergenceFree` (`DFSPH_FINDINGS.md`
        # 1.16), units of `XSPH_FLUID = 0.05`. 0 = the graded default; ~1.0
        # decays the free-surface residual limit cycle.
        # Lattice de-correlation (`shuffleParticles`, `shiftIters=0`); see
        # `initialConditions` for why the constant-density pre-relaxation
        # `tgv`/`shearWave` use is not run here.
        jitter=0.01,
        # Bulk margins for the pressure fit, in particle spacings.
        bulkMargin=8.0,
        wallMargin=6.0,
        markerSize=8,
        # ACSPH only: override `acParams.epsilonV`/`noPenetrationShift` via
        # the shared `configureArtificialCompressible` helper. `None`/`False`
        # leave the paper's defaults alone -- see that helper's docstring.
        epsilonV=None,
        noPenetrationShift=False,
        # 0 (default, off) = plain sharp bottom corners, unchanged behaviour.
        # > 0 rounds the bottom-left/right corners with a concave fillet of
        # this radius in units of `dx` -- see `buildSystem` and
        # `filletedDomainBoundarySdf`'s docstring.
        cornerFilletRadius=0.0,
    ),
))


if __name__ == '__main__':
    caseMain(hydrostaticColumnCase)
