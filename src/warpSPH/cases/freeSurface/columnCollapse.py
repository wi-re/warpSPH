"""Water-column collapse in a closed tank using native pressure projection.
A one-shot gravity release, distinct from the driven sloshingTank case.
Wall containment is measured; no legacy no-penetration impulse is implied."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from ...runner import Case, RunContext, caseMain, registerCase
from ..boundaries.hydrostaticColumn import buildSystem as _hcolBuildSystem
from ..boundaries.hydrostaticColumn import configureScheme as _hcolConfigureScheme
from ..periodic.kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from ..common.plotting import Field, particlePlot
from ..common.weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, shapeSdf,
                                 particleDistributionMetrics,
                                 calibrateRestDensity)

__all__ = ['columnCollapseCase']


def _fluidSdf(ctx: RunContext):
    """The initial fluid body: a box of width `blockWidth * L` against the
    left wall, `fillHeight * L` tall, bottom-anchored."""
    L = ctx.spec.L
    w = ctx.param('blockWidth') * L
    h = ctx.param('fillHeight') * L
    centre = [-0.5 * L + 0.5 * w, -0.5 * L + 0.5 * h]
    return shapeSdf('box', args=[[0.5 * w, 0.5 * h]], offset=centre)


def configureScheme(ctx: RunContext) -> None:
    # `hydrostaticColumn`'s configureScheme, then rotate gravity by `tiltDeg`.
    # It reads `fillRatio`; alias it to our `fillHeight` so its (unused here)
    # pressure-fit normalisation does not divide by zero.
    ctx.spec.params.setdefault('fillRatio', ctx.param('fillHeight'))
    _hcolConfigureScheme(ctx)
    tilt = math.radians(ctx.param('tiltDeg'))
    ctx.schemeConfig.gravityConfig.origin = [math.sin(tilt), -math.cos(tilt)]


def buildSystem(ctx: RunContext):
    # Reuse `hydrostaticColumn`'s walled-box builder, but swap its full-width
    # bottom-anchored column for our left-anchored block by monkey-patching the
    # SDF it calls. Cheaper than duplicating `buildRegionSystem` wiring.
    from ..boundaries import hydrostaticColumn as _hcol
    _saved = _hcol.columnSdf
    _hcol.columnSdf = _fluidSdf
    try:
        return _hcolBuildSystem(ctx)
    finally:
        _hcol.columnSdf = _saved


def initialConditions(ctx: RunContext, system) -> None:
    # Declared since Part 33 but never read until now -- see
    # `calibrateRestDensity`.
    if ctx.param('calibrateRestDensity'):
        calibrateRestDensity(ctx, system, verbose=ctx.spec.verbose)
    p = system.state
    p.velocities[:] = 0.0
    if p.pressures is None:
        p.pressures = torch.zeros_like(p.densities)
    else:
        p.pressures[:] = 0.0
    ctx.scratch['initialPositions'] = p.positions.clone()


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    p = state.state
    fluid = p.kinds == 0
    pos = p.positions[fluid]
    vel = p.velocities[fluid]
    m = p.masses[fluid]
    rho = p.densities[fluid]
    L = ctx.spec.L
    dx = ctx.config.dx

    d = {
        'kineticEnergy': (0.5 * m * (vel ** 2).sum(-1)).sum().item(),
        'maxVelocity': torch.linalg.norm(vel, dim=-1).max().item(),
        'minDensity': rho.min().item(),
        'densityP05': torch.quantile(rho.detach().float(), 0.05).cpu().item(),
        'maxDensity': rho.max().item(),
        'densityStd': rho.std().item(),
        'densityP05': torch.quantile(rho, 0.05).item(),
        'comX': (m * pos[:, 0]).sum().item() / m.sum().item(),
    }
    d.update(particleDistributionMetrics(ctx, state))
    x, y = pos[:, 0], pos[:, 1]
    half = 0.5 * L
    nearLeft = x < -half + 3.0 * dx
    nearRight = x > half - 3.0 * dx
    if bool(nearLeft.any()):
        d['leftWallHeight'] = (y[nearLeft].max().item() + half) / L
    if bool(nearRight.any()):
        d['rightWallHeight'] = (y[nearRight].max().item() + half) / L
    d['surgeFront'] = (x.max().item() + half) / L
    # Wall-penetration watch (see module docstring). Fluid particles more than
    # half a spacing past the interior extent of the tank.
    pen = ((x < -half - 0.5 * dx) | (x > half + 0.5 * dx)
           | (y < -half - 0.5 * dx) | (y > half + 0.5 * dx))
    d['nPenetrating'] = int(pen.sum().item())
    d['maxPenetrationDx'] = float(torch.clamp(
        torch.stack([(-half - x).max(), (x - half).max(),
                     (-half - y).max(), (y - half).max()]).max(), min=0.0
    ).item() / dx)
    return d


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm',
          boundary='Visualize'),
    Field('densities', 'densities', colorMap='viridis', boundary='Visualize'),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in columnCollapseCase.params}


columnCollapseCase = registerCase(Case(
    name='columnCollapse',
    scheme='dfsph',
    description='Water-column collapse in a closed tank (2D): a released column '
               'oscillates under gravity.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='08-columnCollapse',
        dim=2,
        nx=64,
        L=1.0,
        tLimit=3.0,
        periodic=True,
        # `kolmogorovIncompressibleTimestep` applies `cflFactor` to the
        # particle diameter (Bender & Koschier's advective form). The falling
        # column's impact is a sharp event, so -- as on `dambreak` (Part 20) --
        # 0.2, not the periodic cases' 0.4.
        cflFactor=0.2,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        dt=1e-3,
        minDt=1e-8,
        maxDt=2e-3,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        # Fluid block: `blockWidth` of the tank wide, `fillHeight` tall, left-
        # anchored. 0.5 x 0.6 in a unit tank settles to a 0.3-deep flat pool.
        blockWidth=0.5,
        fillHeight=0.6,
        tiltDeg=0.0,
        gravityMagnitude=9.81,
        gravityDirection=[0.0, -1.0],
        bandWidth=16.0,
        shifting=False,
        calibrateRestDensity=False,
        wallBC='freeSlip',
        # Opt-in post-solve XSPH velocity smoother (DFSPH_FINDINGS.md 1.16).
        xsphScale=0.0,
        jitter=0.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(columnCollapseCase)
