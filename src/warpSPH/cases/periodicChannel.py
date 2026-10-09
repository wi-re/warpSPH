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
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils import buildDomainDescription
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, fluidRegion, setupTimestep, shapeSdf, weaklyCompressibleDiagnostics)

__all__ = ['periodicChannelCase']

XI = {'Wendland2': 2.8213846683502197, 'Wendland4': 3.56734561920166}      # warpSPHCore sphKernel_xi(kernel, 2D)


def _plateBody(yCenter: float, bodyId: int):
    from warpSPHBoundaries.scene import Body
    from warpSPHBoundaries.scene.scene import BoxRep
    return Body(bodyId=bodyId, center=(0.5, yCenter), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])


def configureScheme(ctx: RunContext) -> None:
    Lx = ctx.spec.L
    dx = Lx / ctx.spec.nx
    W = ctx.param('W')
    ny = int(round(W / dx))
    ctx.scratch['W'] = ny * dx                                  # the lattice height: the channel is a whole number of rows
    h = ctx.spec.n_h * dx
    pad = 2.0 * h
    shear = ctx.param('shearWave')
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


def initialConditions(ctx: RunContext, system) -> None:
    st = system.state
    st.velocities[:] = 0.0
    if ctx.param('shearWave'):
        W = ctx.scratch['W']
        fluid = st.kinds == 0
        st.velocities[fluid, 0] = ctx.param('u0') * torch.sin(2.0 * math.pi * st.positions[fluid, 1] / W)
    ctx.scratch['nuNominal'] = ctx.param('alpha') * ctx.param('soundSpeed') * float(ctx.spec.n_h) * ctx.config.dx / (8.0 * XI[ctx.spec.kernel])
    setupTimestep(ctx, system)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    st = state.state
    f = st.kinds == 0
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
        bodyForceAtWall=True,
        shearWave=False,
        fShear=0.0,                  # the body force of the plate-free box (a uniform force on a periodic fluid: the mean velocity grows as f t)
        u0=0.05,
        nuReference=None,
        shifting=True,
        shiftScheme='michel2022',
        shiftProjection='michel2022',
    ),
))


if __name__ == '__main__':
    caseMain(periodicChannelCase)
