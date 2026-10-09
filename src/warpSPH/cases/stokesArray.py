"""Stokes flow through a square periodic array of cylinders (or squares), driven by a uniform body force (rungs 2 and 4 of the boundaries repo's validation ladder; stage 3 E6).

    unit cell [0, 1]^2, fully periodic, one analytic body at the centre (a disk of radius R, or a grid-aligned square of side a), the body force `f` on the fluid only

Steady state: the load of the fluid on the body (pressure + viscous) equals the body force on the fluid, `F = rho f (1 - c) L^2`, and the drag coefficient `K = F / (mu U)`, `U` the superficial velocity (the mean
over the whole cell, the solid counting zero), is compared with the Sangani-Acrivos / Hasimoto expansion of the square array `K_SA = 4 pi / (-1/2 ln c - 0.738 + c - 0.887 c^2 + 2.038 c^3)`, `c = pi R^2 / L^2`,
times `(1 - c)` because the body force acts on the fluid only (a mean pressure gradient would act on the cell). Square: the Fourier-penalisation Stokes solve of the boundaries repo, `K_ref = 25.91` for a = 1/3.
`mu = rho nu`, `nu` the viscosity of the discretisation (`nuReference`, from the shear-wave decay of `periodicChannel`'s `shearWave` mode) or the physical one of the incompressible loops.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from ..configurations.region import BCType
from ..enumTypes import isIncompressibleScheme
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils import buildDomainDescription
from .periodicChannel import XI, configureFlow
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, fluidRegion, setupTimestep, shapeSdf, weaklyCompressibleDiagnostics)

__all__ = ['stokesArrayCase', 'sanganiAcrivos']

SQUARE_REF = {round(1.0 / 3.0, 6): 25.91}


def sanganiAcrivos(c: float) -> float:
    return 4.0 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c ** 2 + 2.038 * c ** 3)


def _body(ctx: RunContext):
    from warpSPHBoundaries.scene import Body
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import BoxRep, ImplicitRep
    if ctx.param('shape') == 'square':
        a = ctx.param('a')
        return Body(bodyId=0, center=(0.5, 0.5), reps=[BoxRep((-0.5 * a, -0.5 * a), (0.5 * a, 0.5 * a), solid='inside')])
    return Body(bodyId=0, center=(0.5, 0.5), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=ctx.param('R')))])


def _bodySdf(ctx: RunContext):
    if ctx.param('shape') == 'square':
        a = ctx.param('a')
        return shapeSdf('box', args=[[0.5 * a, 0.5 * a]], offset=[0.5, 0.5])
    return shapeSdf('circle', size=ctx.param('R'), offset=[0.5, 0.5])


def configureScheme(ctx: RunContext) -> None:
    dx = ctx.spec.L / ctx.spec.nx
    domain = buildDomainDescription(ctx.spec.L, ctx.spec.dim, True, ctx.device, ctx.dtype)
    domain.min = torch.zeros(2, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.full((2,), ctx.spec.L, device=ctx.device, dtype=ctx.dtype)
    domain.periodic[:] = True
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx
    ctx.scratch['W'] = ctx.spec.L
    configureFlow(ctx, dx, shear=False)
    ctx.schemeConfig.analyticPeriodicWalls = True


def buildSystem(ctx: RunContext):
    L = ctx.spec.L
    regions = [fluidRegion(ctx, shapeSdf('box', args=[[0.5 * L, 0.5 * L]], offset=[0.5 * L, 0.5 * L])),
               boundaryRegion(ctx, _bodySdf(ctx), kind=BCType[ctx.param('wallBC')], representation=_body(ctx))]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    st = system.state
    st.velocities[:] = 0.0
    sc = ctx.schemeConfig
    dx = ctx.config.dx
    ctx.scratch['solidFraction'] = ctx.param('a') ** 2 if ctx.param('shape') == 'square' else math.pi * ctx.param('R') ** 2
    if isIncompressibleScheme(ctx.scheme):
        from ..modules.analyticBoundary import latticeCalibration
        fl = st.kinds == 0
        cal = latticeCalibration(dx, dx, float(st.supports[fl].max()), ctx.config.kernel)
        st.masses = torch.where(fl, torch.full_like(st.masses, sc.fluid.restDensity * cal['V']), st.masses)      # the bulk lattice at exactly rho0; the wall continuum `mu` of the flat-wall calibration (no plane offset: the body is cut from the lattice)
        sc.analyticWallMass = cal['mu']
        ctx.scratch['nuNominal'] = ctx.param('nuPhysical')
        if ctx.config.dt is None:
            ctx.config.dt = ctx.spec.dt if ctx.spec.dt is not None else 2e-3
        return
    ctx.scratch['nuNominal'] = ctx.param('alpha') * ctx.param('soundSpeed') * float(ctx.spec.n_h) * dx / (8.0 * XI[ctx.spec.kernel])
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
    rho0 = ctx.schemeConfig.fluid.restDensity
    V = st.masses[f].double() / rho0
    U = float((V * st.velocities[f, 0].double()).sum()) / ctx.spec.L ** 2                      # the superficial velocity: the whole cell, the solid counting zero
    nu = ctx.param('nuReference') or ctx.scratch['nuNominal']
    out['superficialVelocity'] = U
    out['fluidMass'] = float(st.masses[f].sum())
    loads = [rb.load[:, 0].sum() for rb in ctx.schemeConfig.boundaryProvider.rigidBodies if getattr(rb, 'load', None) is not None]
    if loads:
        F = float(sum(loads))
        out['bodyLoad'] = F
        out['dragCoefficient'] = F / (nu * U) if U > 0 else 0.0
    return out


def extraData(ctx: RunContext, state) -> Dict:
    return {k: ctx.param(k) for k in stokesArrayCase.params}


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)

stokesArrayCase = registerCase(Case(
    name='stokesArray',
    scheme='deltaSPH',
    description='Stokes flow through a periodic array of cylinders / squares driven by a body force (drag coefficient against Sangani-Acrivos), analytic bodies.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='stokesArray',
        nx=48,
        L=1.0,
        n_h=4.0,
        kernel='Wendland2',
        integrationScheme='rungeKutta2',
        periodic=True,
        tLimit=20.0,
        plotInterval=200,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        alpha=0.5,
        soundSpeed=10.0,
        targetDt=2e-4,
        W=1.0,
        shape='disk',
        R=0.2,
        a=1.0 / 3.0,
        f=0.03,
        wallBC='noSlip',
        fluidViscosity='alpha',
        wallViscosityClosure='mirror',
        complementMoments=True,
        bodyForceAtWall=True,
        shearWave=False,
        fShear=0.0,
        u0=0.05,
        nuReference=None,
        nuPhysical=0.0185,
        calibratedLattice=False,
        projection='jacobi',
        projectionTol=1e-8,
        densitySolve=True,
        particleShift='none',
        shiftA=0.5,
        closedPreset=False,
        shifting=True,
        shiftScheme='michel2022',
        shiftProjection='michel2022',
        pressureConsistent=False,
        wallPressureViscous=False,
    ),
))


if __name__ == '__main__':
    caseMain(stokesArrayCase)
