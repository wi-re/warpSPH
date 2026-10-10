"""Taylor-Couette flow: a rotating inner cylinder (a convex no-slip wall, fluid outside) in a fixed outer cylinder (a concave wall, fluid inside), low Re (rung of the boundaries repo's validation ladder, stage 3 E6).

    u_theta(r) = omega r1^2 (r2^2 / r - r) / (r2^2 - r1^2) ,   torque of the fluid on the inner cylinder (per unit depth)  T = -4 pi mu omega r1^2 r2^2 / (r2^2 - r1^2) ,   mu = rho nu .

Two analytic bodies (a disk `r1`, rotating at `omega` as a prescribed rigid body, and the cavity `r2`), the annulus sampled on the square lattice cut at the walls. The Morris viscosity is the non-symmetric stress
`mu grad v`: the booked torque of a rotating body misses the traction `-2 mu Omega A` of its transpose part (A = the signed area enclosed by the solid), added here to the inner torque.
Diagnostics: the fitted amplitude of `u_theta` over the exact profile, the torques of both walls, the radial profile.
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

__all__ = ['taylorCouetteCase']


def _bodies(ctx: RunContext):
    from warpSPHBoundaries.scene import Body
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import ImplicitRep
    inner = Body(bodyId=0, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=ctx.param('r1')))])
    outer = Body(bodyId=1, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=ctx.param('r2'), solid='outside'))])
    return inner, outer


def configureScheme(ctx: RunContext) -> None:
    dx = ctx.spec.L / ctx.spec.nx
    h = ctx.spec.n_h * dx
    r2 = ctx.param('r2')
    lim = 0.5 * dx * (2 * math.ceil((r2 + 2.0 * h) / dx) + 1)       # an odd number of spacings across the domain (an even number of lattice nodes: nodes at half-integer multiples of dx, as the oracle): the regular sampler lays its lattice over the domain extent, so this gives exactly dx, cell-centred on the axis
    domain = buildDomainDescription(2.0 * lim, ctx.spec.dim, False, ctx.device, ctx.dtype)
    domain.min = torch.tensor([-lim, -lim], device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([lim, lim], device=ctx.device, dtype=ctx.dtype)
    domain.periodic[:] = False
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx
    ctx.scratch['W'] = r2
    configureFlow(ctx, dx, shear=False)
    ctx.schemeConfig.bodyForce = (0.0, 0.0)
    ctx.schemeConfig.analyticPeriodicWalls = False


def wallCut(ctx: RunContext) -> float:
    """The distance from each wall inside which the square lattice is cut (the first row sits beyond it), `wallCut` x dx; `None` takes the oracle's rule of the scheme: the incompressible loops the calibrated first-row
    distance of the flat wall less 0.01 dx (`DFSPH2D`'s `run_couette`), delta+ half a spacing (`taylor_couette.py`); 0 keeps the whole lattice (particles down to the wall)."""
    dx = ctx.spec.L / ctx.spec.nx
    c = ctx.param('wallCut')
    if c is not None:
        return float(c) * dx
    if isIncompressibleScheme(ctx.scheme):
        from ..modules.analyticBoundary import latticeCalibration
        return latticeCalibration(dx, dx, float(ctx.spec.n_h) * dx)['dwallY'] - 0.01 * dx
    return 0.5 * dx


def buildSystem(ctx: RunContext):
    r1, r2 = ctx.param('r1'), ctx.param('r2')
    inner, outer = _bodies(ctx)
    kind = BCType[ctx.param('wallBC')]
    cut = wallCut(ctx)
    regions = [fluidRegion(ctx, shapeSdf('box', args=[[r2, r2]], offset=[0.0, 0.0])),
               boundaryRegion(ctx, shapeSdf('circle', size=r1 + cut, offset=[0.0, 0.0]), kind=kind, representation=inner),
               boundaryRegion(ctx, shapeSdf('circle', size=r2 - cut, offset=[0.0, 0.0], invert=True), kind=kind, representation=outer)]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    st = system.state
    sc = ctx.schemeConfig
    dx = ctx.config.dx
    st.velocities[:] = 0.0
    inner = sc.boundaryProvider.rigidBodies[0]
    inner.angularVelocity = torch.tensor(float(ctx.param('omega')), dtype=inner.angularVelocity.dtype, device=inner.angularVelocity.device)      # prescribed constant rotation
    sc.analyticSpinAxisymmetric = True                              # a disk spinning about its own centre: its pose never matters, so the step can be replayed from a CUDA graph
    if isIncompressibleScheme(ctx.scheme):
        from ..modules.analyticBoundary import latticeCalibration
        fl = st.kinds == 0
        cal = latticeCalibration(dx, dx, float(st.supports[fl].max()), ctx.config.kernel)
        st.masses = torch.where(fl, torch.full_like(st.masses, sc.fluid.restDensity * cal['V']), st.masses)
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
    r1, r2, om = ctx.param('r1'), ctx.param('r2'), ctx.param('omega')
    x, v = st.positions[f].double(), st.velocities[f].double()
    rr = x.norm(dim=1)
    ut = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rr
    ex = om * r1 ** 2 * (r2 ** 2 / rr - rr) / (r2 ** 2 - r1 ** 2)
    out['profileAmplitude'] = float((ut * ex).sum() / (ex * ex).sum())
    nu = ctx.param('nuReference') or ctx.scratch['nuNominal']
    out['torqueExact'] = -4.0 * math.pi * nu * om * r1 ** 2 * r2 ** 2 / (r2 ** 2 - r1 ** 2)
    rbs = ctx.schemeConfig.boundaryProvider.rigidBodies
    if all(getattr(rb, 'load', None) is not None for rb in rbs):
        T = [float(rb.load[:, 2].sum()) for rb in rbs]
        T[0] -= 2.0 * nu * om * math.pi * r1 ** 2                  # the Morris traction of the rotating inner wall: -2 mu Omega A (A = the enclosed area, + for the obstacle)
        out['torqueInner'], out['torqueOuter'] = T[0], T[1]
    return out


def extraData(ctx: RunContext, state) -> Dict:
    return {k: ctx.param(k) for k in taylorCouetteCase.params}


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)

taylorCouetteCase = registerCase(Case(
    name='taylorCouette',
    scheme='deltaSPH',
    description='Taylor-Couette flow between a rotating inner and a fixed outer cylinder (analytic curved walls; profile and torque against the exact solution).',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='taylorCouette',
        nx=48,                        # particles per unit length
        L=1.0,
        n_h=4.0,
        kernel='Wendland2',
        integrationScheme='rungeKutta2',
        periodic=False,
        tLimit=25.0,
        plotInterval=200,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        alpha=0.5,
        soundSpeed=10.0,
        targetDt=2e-4,
        W=0.5,
        r1=0.2,
        r2=0.5,
        omega=1.0,
        f=0.0,
        wallBC='noSlip',
        fluidViscosity='morris',
        wallViscosityClosure='noslipMoment',
        complementMoments=True,
        bodyForceAtWall=True,
        shearWave=False,
        fShear=0.0,
        u0=0.05,
        nuReference=None,
        nuPhysical=0.0185,
        calibratedLattice=False,
        wallCut=None,
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
    caseMain(taylorCouetteCase)
