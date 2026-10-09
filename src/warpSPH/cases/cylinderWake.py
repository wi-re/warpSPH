"""Flow past one analytic body (a disk or a grid-aligned square) driven by a prescribed-velocity frame in a periodic box (rung 5 of the boundaries repo's validation ladder; stage 3 E5/E6; the `movingObstacle`
setup of the oracle's `scripts/studies/cylinder_wake.py`).

    box [0, Lx] x [0, Ly] (in units of the diameter D), periodic in both directions; the frame is the pinned band `|x| < b` and `|y| < b` around the seams (`modules/boundaryConditions/pinned.py`: the
    velocity is the free stream U there, the particles stay fluid); the body sits at (xb, Ly / 2); the run starts impulsively (the uniform stream outside the body).

`Re = U D / nu`, `nu` the viscosity of the discretisation (the Morris form calibrated by `morrisCalibration`, or the alpha form with the lattice-axis factor `NU_FAC`). The drag and lift coefficients are
`C = 2 F / (rho U^2 D)` of the load of the fluid on the body; the Strouhal number and the recirculation length are in `scripts/probe_cylinderWake.py`.
The frame and the periodic images give a blockage `D / (Ly - 2 b)`, which raises C_D and St by a few percent against the unconfined literature.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from ..configurations.region import BCType
from ..enumTypes import isIncompressibleScheme
from ..modules.boundaryConditions import PinnedBand, pinnedBandBC
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils import buildDomainDescription
from .periodicChannel import XI, configureFlow
from .plotting import particlePlot
from .stokesArray import stokesArrayCase
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS, boundaryRegion, buildRegionSystem, fluidRegion,
                                 setupTimestep, shapeSdf, weaklyCompressibleDiagnostics)

__all__ = ['cylinderWakeCase']

NU_FAC = 0.957             # the alpha form's lattice-axis viscosity factor at H = 4 dx (shear-wave decay along the axes)


def _geometry(ctx: RunContext):
    """(Lx, Ly, xc, yc) of the lattice: the box a whole number of rows, the body centre on the lattice half-spacing lines for the square."""
    dx = ctx.spec.L / ctx.spec.nx
    nx = ctx.spec.nx
    ny = int(round(ctx.param('Ly') / dx))
    xc, yc = ctx.param('xb'), 0.5 * ny * dx
    if ctx.param('shape') == 'square':
        yc, xc = round(yc / dx) * dx, round(xc / dx) * dx
    return nx * dx, ny * dx, xc, yc


def _body(ctx: RunContext, xc: float, yc: float):
    from warpSPHBoundaries.scene import Body
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import BoxRep, ImplicitRep
    D = ctx.param('D')
    if ctx.param('shape') == 'square':
        return Body(bodyId=0, center=(xc, yc), reps=[BoxRep((-0.5 * D, -0.5 * D), (0.5 * D, 0.5 * D), solid='inside')])
    return Body(bodyId=0, center=(xc, yc), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.5 * D))])


def _bodySdf(ctx: RunContext, xc: float, yc: float):
    D = ctx.param('D')
    if ctx.param('shape') == 'square':
        return shapeSdf('box', args=[[0.5 * D, 0.5 * D]], offset=[xc, yc])
    return shapeSdf('circle', size=0.5 * D, offset=[xc, yc])


def configureScheme(ctx: RunContext) -> None:
    Lx, Ly, _, _ = _geometry(ctx)
    dx = ctx.spec.L / ctx.spec.nx
    domain = buildDomainDescription(ctx.spec.L, ctx.spec.dim, True, ctx.device, ctx.dtype)
    domain.min = torch.zeros(2, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([Lx, Ly], device=ctx.device, dtype=ctx.dtype)
    domain.periodic[:] = True
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx
    ctx.scratch['W'] = Ly
    configureFlow(ctx, dx, shear=False)
    sc = ctx.schemeConfig
    sc.analyticPeriodicWalls = True
    sc.bodyForce = None
    # the viscosity of the Reynolds number: nu = U D / Re
    nu = ctx.param('U') * ctx.param('D') / ctx.param('Re')
    h = float(ctx.spec.n_h) * dx
    ctx.scratch['nuNominal'] = nu
    if isIncompressibleScheme(ctx.scheme):
        sc.diffusionParams.viscidNu = nu / ctx.scratch['morrisCal']
    elif ctx.param('fluidViscosity') == 'morris':
        sc.diffusionParams.viscidNu = nu / ctx.scratch['morrisCal']
    else:
        sc.diffusionParams.inviscidAlpha = nu / NU_FAC * 8.0 * XI[ctx.spec.kernel] / (ctx.param('soundSpeed') * h)


def buildSystem(ctx: RunContext):
    Lx, Ly, xc, yc = _geometry(ctx)
    regions = [fluidRegion(ctx, shapeSdf('box', args=[[0.5 * Lx, 0.5 * Ly]], offset=[0.5 * Lx, 0.5 * Ly])),
               boundaryRegion(ctx, _bodySdf(ctx, xc, yc), kind=BCType[ctx.param('wallBC')], representation=_body(ctx, xc, yc))]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    st = system.state
    Lx, Ly, xc, yc = _geometry(ctx)
    U, D = ctx.param('U'), ctx.param('D')
    st.velocities[:] = 0.0
    st.velocities[:, 0] = U
    sc = ctx.schemeConfig
    b = ctx.param('band') * D
    band = PinnedBand(slabs=((0, 0.0, b), (1, 0.0, b)), velocity=(U, 0.0), lo=(0.0, 0.0), hi=(Lx, Ly))
    sc.boundaryConditions = list(sc.boundaryConditions or []) + [pinnedBandBC(band)]
    ctx.scratch['blockage'] = D / (Ly - 2.0 * b)
    ctx.scratch['bodyCentre'] = (xc, yc)
    dx = ctx.config.dx
    if isIncompressibleScheme(ctx.scheme):
        from ..modules.analyticBoundary import latticeCalibration
        fl = st.kinds == 0
        cal = latticeCalibration(dx, dx, float(st.supports[fl].max()), ctx.config.kernel)
        st.masses = torch.where(fl, torch.full_like(st.masses, sc.fluid.restDensity * cal['V']), st.masses)
        sc.analyticWallMass = cal['mu']
        if ctx.config.dt is None:
            ctx.config.dt = ctx.spec.dt if ctx.spec.dt is not None else 2e-3
        return
    setupTimestep(ctx, system)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    st = state.state
    f = st.kinds == 0
    if isIncompressibleScheme(ctx.scheme):
        v = st.velocities[f]
        out = {'maxVelocity': float(torch.linalg.norm(v, dim=-1).max()), 'minDensity': float(st.densities[f].min()), 'maxDensity': float(st.densities[f].max())}
    else:
        out = weaklyCompressibleDiagnostics(ctx, state)
    q = 0.5 * ctx.schemeConfig.fluid.restDensity * ctx.param('U') ** 2 * ctx.param('D')
    loads = [rb.load for rb in ctx.schemeConfig.boundaryProvider.rigidBodies if getattr(rb, 'load', None) is not None]
    if loads:
        out['dragCoefficient'] = float(sum(ld[:, 0].sum() for ld in loads)) / q
        out['liftCoefficient'] = float(sum(ld[:, 1].sum() for ld in loads)) / q
    return out


def extraData(ctx: RunContext, state) -> Dict:
    return {k: ctx.param(k) for k in cylinderWakeCase.params}


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)

cylinderWakeCase = registerCase(Case(
    name='cylinderWake',
    scheme='deltaSPH',
    description='Flow past a disk / square in a periodic box with a pinned-velocity frame (drag, lift, Strouhal, recirculation length), analytic body.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='cylinderWake',
        nx=300,
        L=30.0,
        n_h=4.0,
        kernel='Wendland2',
        integrationScheme='rungeKutta2',
        periodic=True,
        tLimit=150.0,
        plotInterval=400,
    ),
    params=dict(
        stokesArrayCase.params,
        shape='disk',
        D=1.0,
        U=1.0,
        Re=100.0,
        Ly=15.0,
        band=1.5,
        xb=10.0,
        soundSpeed=10.0,
        alpha=0.1,
        f=0.0,
        fluidViscosity='morris',
        nuPhysical=0.01,
        targetDt=1e-3,
    ),
))


if __name__ == '__main__':
    caseMain(cylinderWakeCase)
