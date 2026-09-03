"""Rotating square patch (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/03-rotating-square-patch.ipynb` and
`examples/incompressible/03-rotating-square-patch.ipynb` -- the same geometry
under two schemes, which is `--scheme deltaSPH` or `--scheme divergenceFree`
here.

A square of fluid in rigid rotation is not an equilibrium: the free surface has
to deform, and the arms it grows are the thing being compared between schemes.

The square is the benchmark, but the patch is a shape parameter like the impact
case's -- `--shape` takes any key of
:data:`~warpSPH.cases.weaklyCompressible.SHAPE_PRESETS`, sized by `--size` and
`--aspectRatio` and pre-turned by `--rotation`. The corners are what make this
test hard (a circle in rigid rotation *is* an equilibrium), so `--shape circle`
is the null experiment and `--shape triangleIsosceles`/`--shape star5` are
sharper versions of the same one.
"""

from __future__ import annotations

from typing import Dict

import torch

from warpSPHCore import sphKernelScale

from ..modules import setupWeaklyCompressibleTimestep
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils.support import volumeToSupport
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 centredShapeSdf, configureWeaklyCompressible, fluidRegion,
                                 paramExtraData, shapeArgs,
                                 squarePatchAreaMetrics, weaklyCompressibleDiagnostics)

__all__ = ['rotatingSquarePatchCase']


def _setupTimestep(ctx: RunContext, system) -> None:
    """Pick `dt` (and the sound speed) for the patch.

    The shared `setupTimestep` fixes `targetDt` and lets the sound speed follow
    from the acoustic CFL -- but for this case `Umax = ω·R` is fixed while `dx`
    shrinks with resolution, so a fixed `targetDt` drives the Mach number *up*
    with `nx` (Ma ≈ 0.055 at nx 64, ≈ 0.34 at nx 400), well out of the
    weakly-compressible regime. When `mach` is set (the default), pick
    `targetDt` so the sound speed lands at `Umax / mach` instead --
    `setupWeaklyCompressibleTimestep` uses `c0 = K / targetDt` with `K` fixed
    by dx / kernel / neighbour count, so the right `targetDt` is `K·mach/Umax`
    in closed form. `mach=None` restores the old fixed-`targetDt` behaviour.
    """
    mach = ctx.param('mach', None)
    baseDt = ctx.param('targetDt')
    if mach is not None:
        omega = ctx.param('omega')
        pos = system.state.positions[system.state.kinds == 0]
        uMax = float((omega * torch.linalg.norm(pos, dim=-1)).max())
        # `setupWeaklyCompressibleTimestep` sets `c0 = K / targetDt` with `K`
        # fixed by dx / kernel / neighbour count -- so the targetDt that lands
        # `c0 = uMax / mach` is just `K * mach / uMax` (no provisional call, and
        # no spurious Mach warning from one).
        K = 0.3 * volumeToSupport(ctx.config.dx ** 2, ctx.config.targetNeighbors, 2) \
            / float(sphKernelScale(ctx.config.kernel.value, 2))
        baseDt = K * mach / uMax
    ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
        ctx.config, ctx.schemeConfig, system, baseDt, verbose=ctx.spec.verbose)


def buildSystem(ctx: RunContext):
    args = shapeArgs(ctx.param('shape'), ctx.param('size'), ctx.param('aspectRatio'))
    # Centred on the measured shape, not on the primitive's own origin, so the
    # rotation below is about the patch rather than about a point beside it.
    sdf, _ = centredShapeSdf(ctx.param('shape'), args, [0.0, 0.0], ctx.config.domain,
                             rotation=ctx.param('rotation'))
    return buildRegionSystem(ctx, [fluidRegion(ctx, sdf)])


def initialConditions(ctx: RunContext, system) -> None:
    omega = ctx.param('omega')
    positions = system.state.positions
    system.state.velocities[:, 0] = omega * positions[:, 1]
    system.state.velocities[:, 1] = -omega * positions[:, 0]
    _setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    # Area/volume-conservation metrics (`WCSPH_SHIFTING_PLAN.md` step 1) on top
    # of the usual KE / v_max / density-bound health check: the δ⁺-SPH surface
    # shift is not volume-preserving, and this case is the controlled probe of
    # that drift.
    return {**weaklyCompressibleDiagnostics(ctx, state),
            **squarePatchAreaMetrics(ctx, state)}


rotatingSquarePatchCase = registerCase(Case(
    name='squarePatch',
    scheme='deltaSPH',
    description='Rotating square patch of fluid (2D), weakly compressible or incompressible.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='03-rotatingSquarePatch',
        nx=192,
        # 2 * 3 * R: the patch grows arms well past its initial extent, so the
        # box has to be much larger than the patch.
        L=6.0,
        tLimit=1.0,
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,
        # `box` at aspect 1 is the 2x2 square the benchmark is defined on.
        shape='box',
        size=1.0,
        aspectRatio=1.0,
        rotation=0.0,
        omega=4.0,
        # Target Mach number: pick `dt`/`c0` so `Umax/c0 = mach` at every
        # resolution (see `_setupTimestep`). `None` falls back to the fixed
        # `targetDt`, whose Mach number climbs with `nx`. Sun et al. 2019 keep
        # this well below 0.1.
        mach=0.05,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(rotatingSquarePatchCase)
