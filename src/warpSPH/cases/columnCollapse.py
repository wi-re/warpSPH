"""Water-column collapse in a closed tank (2D), divergence-free incompressible SPH.

A tall, narrow block of fluid against the left wall of a closed tank is
released at t=0, collapses to the right, climbs the far wall, and sloshes back
and forth -- a damped free oscillation settling to a flat pool at
`blockWidth * fillHeight` of the tank height. This is the Koshizuka & Oka 1996
"collapse of a water column" test: gravity-driven, one initially-flat free
surface, well-known geometry -- the free-surface companion to
`hydrostaticColumn`.

**Not a sloshing tank.** A sloshing tank proper is a *wide, shallow* basin
under a *laterally oscillating* excitation (realised as a time-varying gravity
direction), which this case does not do -- it is a one-shot release. A real
`sloshingTank` case (wide basin, thin layer, oscillating gravity) is a TODO;
`tiltDeg` here only rotates gravity to a *fixed* angle for a tilted-tank
variant.

`gravityConfig.active` is True, so the Part 47 gate routes this through the
in-step constant-density velocity impulse (`dfsph.INSTEP_CD` auto-on) -- the
same path `hydrostaticColumn` and `dambreak` take; the VD+PS position shift is
off here. `configureScheme`, the walled-box `buildSystem`, `wallBC`, `nu`,
`xsphScale` and the surface-detection bandwidth are all `hydrostaticColumn`'s.

**Wall penetration (was a known issue; resolved Part 49):** the `c637785`
rewrite had commented `computeMdbcNoPenShift`'s call out of `dfsph_step`,
leaving the pressure projection alone for no-penetration, and that let ~6
fluid particles cross the wall band a full spacing at the collapse impact.
`dfsph.NOPEN_SHIFT` re-wires the shift (default on); with it, `nPenetrating`
stays 0-1 through the impact. `diagnostics` still reports `nPenetrating` /
`maxPenetrationDx` each step. See `DFSPH_FINDINGS.md` 1.6 / 2.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .hydrostaticColumn import buildSystem as _hcolBuildSystem
from .hydrostaticColumn import configureScheme as _hcolConfigureScheme
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .plotting import Field, particlePlot
from .weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, shapeSdf,
                                 particleDistributionMetrics,
                                 calibrateRestDensityMasses)

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
    from . import hydrostaticColumn as _hcol
    _saved = _hcol.columnSdf
    _hcol.columnSdf = _fluidSdf
    try:
        return _hcolBuildSystem(ctx)
    finally:
        _hcol.columnSdf = _saved


def initialConditions(ctx: RunContext, system) -> None:
    # Declared since Part 33 but never read until now -- see
    # `calibrateRestDensityMasses`.
    if ctx.param('calibrateRestDensity'):
        calibrateRestDensityMasses(ctx, system, verbose=ctx.spec.verbose)
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
    scheme='divergenceFree',
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
