"""Plane Poiseuille flow between two analytic plates, periodic in x, driven by a body force (rung 3 of the boundaries repo's validation ladder, the gate of the channel-flow group,
ANALYTIC_BOUNDARIES_PLAN.md stage 3).

    box [0, 1] x [0, W], the plates are two `BoxRep` bodies that span the periodic box (a wall that is itself periodic: `schemeConfig.analyticPeriodicWalls`),
    `schemeConfig.bodyForce = (f, 0)` the pressure-gradient driver (momentum only, also in the wall pressure condition)

Steady state: `u(y) = f y (W - y) / (2 nu)`, mean `f W^2 / (12 nu)`, the plate loads sum to `rho f W L`. The viscosity is the artificial one (`alpha`, `nu = alpha c0 h / (8 xi)`: the wall term of the
no-slip mirror is the same operator): `nuReference` overrides it with a measured value (the shear-wave decay of the same discretisation, `shearWave=True`). The diagnostics report the fitted
amplitude of the profile over the parabola (1 = the wall is no-slip at y = 0, W with the bulk viscosity: free slip or a too weak wall flatten / sharpen it) and the plate loads.

`shearWave=True`: no plates, fully periodic, `u_x = u0 sin(2 pi y / W)`, no force: the amplitude decays as `exp(-nu k^2 t)`, which measures the viscosity of the discretisation.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from ..caseUtils.weaklyCompressible import analyticTankBody  # noqa: F401  (kept next to the tank helper: the plates are built the same way)
from ..configurations.region import BCType
from ..enumTypes import *  # noqa: F401,F403
from ..enumTypes import isIncompressibleScheme
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils import buildDomainDescription
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, fluidRegion, setupTimestep, shapeSdf, weaklyCompressibleDiagnostics)

__all__ = ['periodicChannelCase']

XI = {'Wendland2': 2.8213846683502197, 'Wendland4': 3.56734561920166}      # warpSPHCore sphKernel_xi(kernel, 2D)


def _plateBody(yCenter: float, bodyId: int, offset: float = 0.0):
    """A plate of the channel: a box 1.6 long (spanning the periodic box) and 0.3 thick. `offset`: the wall plane moved away from the fluid by this much (the calibrated lattice), by shrinking the box
    from the fluid side so that the body centre (which the integrated state owns) stays put."""
    from warpSPHBoundaries.scene import Body
    from warpSPHBoundaries.scene.scene import BoxRep
    lower = yCenter < 0.25
    lo, hi = ((-0.15, 0.15 - offset) if lower else (-0.15 + offset, 0.15))
    return Body(bodyId=bodyId, center=(0.5, yCenter), reps=[BoxRep((-0.8, lo), (0.8, hi))])


def configureScheme(ctx: RunContext) -> None:
    Lx = ctx.spec.L
    dx = Lx / ctx.spec.nx
    W = ctx.param('W')
    shear = ctx.param('shearWave')
    ny = ctx.spec.nx if shear else int(round(W / dx))          # the shear-wave box is square (wavelength 1: the long-wave viscosity, the finite-k correction of the symbol is (k dx)^2 small)
    ctx.scratch['W'] = ny * dx                                  # the lattice height: the channel is a whole number of rows
    h = ctx.spec.n_h * dx
    pad = 2.0 * h
    # the sampler puts lattice nodes on the non-periodic axes and cell centres on the periodic ones: the half-spacing shift of the lower bound puts the first rows at dx / 2 from the plates, as the boundaries repo's lattice
    ymin, ymax = (0.0, ny * dx) if shear else (-pad - 0.5 * dx, ny * dx + pad + 0.5 * dx)
    domain = buildDomainDescription(Lx, ctx.spec.dim, True, ctx.device, ctx.dtype)
    domain.min = torch.tensor([0.0, ymin], device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([Lx, ymax], device=ctx.device, dtype=ctx.dtype)
    domain.periodic[:] = torch.tensor([True, bool(shear)], device=domain.periodic.device)
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx

    sc = ctx.schemeConfig
    sc.surfaceDetectionConfig.active = False
    sc.gravityConfig.active = False
    sc.diffusionParams.inviscid = True
    sc.diffusionParams.inviscidAlpha = ctx.param('alpha')
    sc.wallViscosityClosure = ctx.param('wallViscosityClosure')
    if ctx.param('fluidViscosity') == 'morris':
        # Morris et al. 1997 viscous operator (carries shear: what a no-slip wall closure needs), nu such that the REALISED long-wave viscosity is the alpha one: nu_used = nominal / cal
        from ..enumTypes import ViscosityTerm
        from ..modules.incompressible.compactProjection import morrisCalibration
        h = float(ctx.spec.n_h) * dx
        cal = morrisCalibration(h, dx, dx * dx)
        sc.diffusionParams.inviscid = False
        sc.diffusionParams.viscousTerm = ViscosityTerm.morris1997
        sc.diffusionParams.viscidNu = ctx.param('alpha') * ctx.param('soundSpeed') * h / (8.0 * XI[ctx.spec.kernel]) / cal
        sc.morrisCalibration = None
        sc.complementMoments = ctx.param('complementMoments')
        ctx.scratch['morrisCal'] = cal
    inc = isIncompressibleScheme(ctx.scheme)
    if inc:
        # the incompressible loops (omniIncompressible, divergenceFree, ...): the viscosity is the physical Morris one (`nuPhysical`, the realised long-wave value), the loop's own explicit term
        from ..enumTypes import ViscosityTerm
        from ..modules.incompressible.compactProjection import morrisCalibration
        h = float(ctx.spec.n_h) * dx
        cal = morrisCalibration(h, dx, dx * dx)
        sc.diffusionParams.inviscid = False
        sc.diffusionParams.viscousTerm = ViscosityTerm.morris1997
        sc.diffusionParams.viscidNu = ctx.param('nuPhysical') / cal
        sc.morrisCalibration = None
        sc.complementMoments = ctx.param('complementMoments')
        sc.omniViscosity = True
        sc.projection, sc.projectionTol, sc.densitySolve = ctx.param('projection'), ctx.param('projectionTol'), ctx.param('densitySolve')
        sc.shifting, sc.shiftA = ctx.param('particleShift'), ctx.param('shiftA')
        sc.closedPreset = ctx.param('closedPreset')
        sc.freeSurface = False
        ctx.scratch['morrisCal'] = cal
    sc.analyticPeriodicWalls = not shear
    sc.bodyForce = (ctx.param('fShear') if shear else ctx.param('f'), 0.0)
    sc.bodyForceAtWall = ctx.param('bodyForceAtWall')
    if not shear:
        sc.analyticWallLoads = True
    if hasattr(sc, 'shiftProperties'):
        from ..configurations.moduleConfigurations.shifting import ShiftingProjectionScheme, ShiftingScheme
        sc.shiftProperties.active = ctx.param('shifting')
        sc.shiftProperties.scheme = ShiftingScheme[ctx.param('shiftScheme')]
        sc.shiftProperties.projectionScheme = ShiftingProjectionScheme[ctx.param('shiftProjection')]


def buildSystem(ctx: RunContext):
    Lx, W = ctx.spec.L, ctx.scratch['W']
    dx = ctx.config.dx
    fluidSdf = shapeSdf('box', args=[[0.5 * Lx, 0.5 * W]], offset=[0.5 * Lx, 0.5 * W])
    regions = [fluidRegion(ctx, fluidSdf)]
    if not ctx.param('shearWave'):
        kind = BCType[ctx.param('wallBC')]
        for i, yc in enumerate((-0.15, W + 0.15)):
            plateSdf = shapeSdf('box', args=[[0.8, 0.15]], offset=[0.5 * Lx, yc])
            regions.append(boundaryRegion(ctx, plateSdf, kind=kind, representation=_plateBody(yc, i)))
    return buildRegionSystem(ctx, regions)


def calibrateChannel(ctx: RunContext, system) -> None:
    """The calibrated lattice of the incompressible loops for the channel (`boundary/calibration.py` for the tank): particle mass `rho0 V'`, wall mass `mu`, the plates moved so that the first rows sit at the distance
    that gives them the bulk density (`latticeCalibration`, flat wall, Wendland C2); the plates are rebuilt with the new box extents and the provider with them."""
    from ..boundary import buildBoundaryProvider
    from ..modules.analyticBoundary import latticeCalibration
    sc = ctx.schemeConfig
    st = system.state
    fluid = st.kinds == 0
    dx = ctx.config.dx
    h = float(st.supports[fluid].max())
    cal = latticeCalibration(dx, dx, h, ctx.config.kernel)
    st.masses = torch.where(fluid, torch.full_like(st.masses, sc.fluid.restDensity * cal['V']), st.masses)
    sc.analyticWallMass = cal['mu']
    off = cal['dwallY'] - 0.5 * dx                              # the first row is at dx / 2 from the plate plane: the plane moves outward by the difference
    W = ctx.scratch['W']
    old = sc.boundaryProvider
    regions = [r for r in ctx.config.regions if getattr(r, 'representation', None) is not None]
    for i, (rb, region, yc) in enumerate(zip(old.rigidBodies, regions, (-0.15, W + 0.15))):
        body = _plateBody(yc, i, off)
        body.bodyId = rb.representation.bodyId
        region.representation = body
        rb.representation = body
    newProvider = buildBoundaryProvider(ctx.config.regions, st.positions.device, domain=ctx.config.domain, support=float(st.supports.max()))
    newProvider.rigidBodies = old.rigidBodies
    sc.boundaryProvider = newProvider
    ctx.scratch['latticeCalibration'] = cal


def initialConditions(ctx: RunContext, system) -> None:
    st = system.state
    st.velocities[:] = 0.0
    if ctx.param('shearWave'):
        W = ctx.scratch['W']
        fluid = st.kinds == 0
        st.velocities[fluid, 0] = ctx.param('u0') * torch.sin(2.0 * math.pi * st.positions[fluid, 1] / W)
    if isIncompressibleScheme(ctx.scheme):
        ctx.scratch['nuNominal'] = ctx.param('nuPhysical')
        if ctx.param('shearWave'):
            from ..modules.analyticBoundary import latticeCalibration
            fl = st.kinds == 0
            cal = latticeCalibration(ctx.config.dx, ctx.config.dx, float(st.supports[fl].max()), ctx.config.kernel)
            st.masses = torch.where(fl, torch.full_like(st.masses, ctx.schemeConfig.fluid.restDensity * cal['V']), st.masses)           # a fluid-only lattice: the bulk density is exactly rho0
        elif ctx.param('calibratedLattice'):
            calibrateChannel(ctx, system)
        if ctx.config.dt is None:
            ctx.config.dt = ctx.spec.dt if ctx.spec.dt is not None else 2e-3
        return
    ctx.scratch['nuNominal'] = ctx.param('alpha') * ctx.param('soundSpeed') * float(ctx.spec.n_h) * ctx.config.dx / (8.0 * XI[ctx.spec.kernel])
    setupTimestep(ctx, system)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    st = state.state
    f = st.kinds == 0
    if isIncompressibleScheme(ctx.scheme):
        v = st.velocities[f]
        out = {'maxVelocity': float(torch.linalg.norm(v, dim=-1).max()), 'kineticEnergy': float(0.5 * (st.masses[f] * (v ** 2).sum(-1)).sum()),
               'minDensity': float(st.densities[f].min()), 'maxDensity': float(st.densities[f].max())}
    else:
        out = weaklyCompressibleDiagnostics(ctx, state)
    W = ctx.scratch['W']
    y, u = st.positions[f, 1].double(), st.velocities[f, 0].double()
    if ctx.param('shearWave'):
        s = torch.sin(2.0 * math.pi * y / W)
        out['modeAmplitude'] = float((u * s).sum() / (s * s).sum()) / (ctx.param('u0') or 1.0)
        return out
    nu = ctx.param('nuReference') or ctx.scratch['nuNominal']
    exact = ctx.param('f') * y * (W - y) / (2.0 * nu)
    out['profileAmplitude'] = float((u * exact).sum() / (exact * exact).sum())
    out['meanVelocity'] = float(u.mean())
    out['meanVelocityRef'] = ctx.param('f') * W * W / (12.0 * nu)
    loads = [rb.load[:, 0].sum() for rb in ctx.schemeConfig.boundaryProvider.rigidBodies if getattr(rb, 'load', None) is not None]
    if loads:
        out['plateLoad'] = float(sum(loads))                 # the load of the fluid on both plates, x (pressure + viscous), per unit depth
        out['fluidMass'] = float(st.masses[f].sum())
    return out


def extraData(ctx: RunContext, state) -> Dict:
    return {k: ctx.param(k) for k in periodicChannelCase.params}


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)

periodicChannelCase = registerCase(Case(
    name='periodicChannel',
    scheme='deltaSPH',
    description='Plane Poiseuille flow between two periodic analytic plates, driven by a body force (delta+ on analytic walls).',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='periodicChannel',
        nx=48,                      # particles per unit length: the box is 1 long
        L=1.0,
        n_h=4.0,
        kernel='Wendland2',          # the analytic wall shifting is C2 only
        integrationScheme='rungeKutta2',
        periodic=True,
        tLimit=12.0,
        plotInterval=200,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        alpha=0.5,
        soundSpeed=10.0,
        targetDt=2e-4,
        W=0.5,
        f=0.05,
        wallBC='noSlip',
        fluidViscosity='alpha',           # 'alpha' (the artificial viscosity, delta+'s default) | 'morris' (Morris 1997, the shear-carrying operator the no-slip closure is built for)
        wallViscosityClosure='mirror',    # 'mirror' (the exact wall Laplacian with the antisymmetric mirror) | 'noslipMoment' (the moment closure, Morris only)
        complementMoments=True,
        bodyForceAtWall=True,
        shearWave=False,
        fShear=0.0,                  # the body force of the plate-free box (a uniform force on a periodic fluid: the mean velocity grows as f t)
        u0=0.05,
        nuReference=None,
        # the incompressible loops (--scheme omniIncompressible | divergenceFree): physical Morris viscosity, calibrated lattice, the projection / shift options of the stage 2 schemes
        nuPhysical=0.0185,
        calibratedLattice=True,
        projection='jacobi',
        projectionTol=1e-8,
        densitySolve=True,
        particleShift='none',
        shiftA=0.5,
        closedPreset=False,
        shifting=True,
        shiftScheme='michel2022',
        shiftProjection='michel2022',
    ),
))


if __name__ == '__main__':
    caseMain(periodicChannelCase)
