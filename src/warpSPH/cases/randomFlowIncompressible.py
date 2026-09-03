"""Decaying random flow (2D), divergence-free incompressible SPH (DFSPH).

The incompressible sibling of :mod:`randomFlow` (weakly compressible
`deltaSPH`), added for `DFSPH_IMPROVEMENT_PLAN.md` Part 2 steps 7/8: the
bounded (`--bounded`) variant is what exercises this codebase's mDBC
boundary-particle machinery (`BoundaryPressureMode` and its pressure-solver
masking, `computeMdbcDensity`/`computeMdbcPressure`) end to end for the first
time -- no incompressible case samples `kind==1` boundary particles yet, and
this is also the first case of *any* scheme to actually run mDBC's ghost
particles (`kind==2`, produced generically by
`initializers.weaklyCompressible.addBoundaryGhostParticles` for every scheme)
through a full step loop rather than as a verified no-op.

This is a **separate case module**, not a `--scheme divergenceFree` flag on
`randomFlow` itself, even though `--scheme` is already a generic CLI override
on every case (`CaseSpec.scheme`, read by `runner.run` in preference to
`Case.scheme`) -- see `randomFlow.py`'s own docstring, which already assumed
this worked. It does not, for two independent reasons found while building
this case:

- `Case.timestep` is one hook shared by every scheme a case might run under.
  `randomFlow` leaves it unset, which is correct for its `deltaSPH` default
  (falls through to `modules.timestep.computeTimestep`'s
  `WeaklyCompressibleSystem` branch) but not for `divergenceFree`: an
  `IncompressibleSystem` falls through to the *compressible* branch instead
  (`computeTimestep`'s dispatcher only special-cases
  `WeaklyCompressibleSystem`), which reads `system.state.internalEnergies` --
  an attribute `IncompressibleState` does not have. `--scheme
  divergenceFree` on the existing case crashes with an `AttributeError` on
  the first adaptive-dt step, exactly the failure `kolmogorovIncompressible`
  already worked around with its own CFL-only `timestep` hook (reused here
  as-is -- the formula does not depend on anything Kolmogorov-specific).
- `randomFlow.initialConditions` finishes with `setupTimestep`, which sets a
  *fixed* `dt = targetDt` for the whole run (mechanically harmless for DFSPH,
  since the sound speed it also derives is unused by this scheme, but not
  adaptive) and unconditionally warns above 10% of a synthetic acoustic Mach
  number that means nothing for an incompressible solver.

Everything else `randomFlow` does -- `buildSystem`'s region sampling/mass
handling, `configureScheme`'s domain/band widening, the boundary and obstacle
SDFs, the divergence-free noise seeding, the diagnostics -- turned out to be
scheme-agnostic already (`initializers.weaklyCompressible.initializeSimulation`
branches generically on `SimulationState is IncompressibleState`) and is
reused directly rather than re-implemented.
"""

from __future__ import annotations

from typing import Any, Dict

from ..configurations import BoundaryPressureMode
from ..runner import Case, RunContext, caseMain, registerCase, resolveEnum
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .randomFlow import BOUNDED_BAND, buildSystem, noiseVelocities
from .weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 configureWeaklyCompressible, paramExtraData,
                                 weaklyCompressibleDiagnostics)
from .plotting import particlePlot

__all__ = ['randomFlowIncompressibleCase']


def configureScheme(ctx: RunContext) -> None:
    if ctx.param('bounded') and not ctx.param('band'):
        ctx.spec.params['band'] = BOUNDED_BAND
    configureWeaklyCompressible(ctx)

    # NOTE (`DFSPH_IMPROVEMENT_PLAN.md` "Immediate A1", Part 48): the
    # `--bounded` variant DIVERGES under the post-`c637785` `divergenceFree`
    # scheme at every resolution. A density excursion builds near the wall
    # over many steps (rho spreads [0.98,1.02] -> [0.2,3.4] while KE/|v| stay
    # flat), then the pressure response detonates the velocity; onset time
    # scales with `nx` (nx=32 t~0.025, nx=48 t~0.16, nx=64 t~0.95, nx=128
    # step 4). Isolated to the constant-density `solveIncompressible` inside
    # the VD+PS shift (`systems/incompressible.finalize`) -- the near-singular
    # pure-Neumann near-wall operator of the plan's Active track (Parts
    # 42-45); `iisph` / `omniIncompressible` fail this same case. No knob
    # tried (maxDt, cflFactor, `DIVERGENCE_SOLVER`, `boundaryPressureMode`,
    # `nu`, disabling the shift) fixes it. Left open, folded into the
    # near-wall CD-solve / `band2018pb` work; the regression test xfails it.

    # `configureWeaklyCompressible` wires the shared `inviscid`/`alpha`
    # (artificial-viscosity) knobs `deltaSPH` uses; DFSPH has no such term
    # (see `kolmogorovIncompressible`), so this scheme always runs with a
    # plain physical viscosity instead.
    schemeConfig = ctx.schemeConfig
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    schemeConfig.shiftProperties.active = ctx.param('shifting')
    schemeConfig.solverConfig.boundaryPressureMode = resolveEnum(
        BoundaryPressureMode, ctx.param('boundaryPressureMode'))
    # The surface-detection bandwidth is measured in particle spacings.
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def initialConditions(ctx: RunContext, system) -> None:
    system.state.velocities[:] = noiseVelocities(ctx, system)
    # DFSPH has no acoustic term to derive a fixed `dt` from the way
    # `randomFlow`'s `setupTimestep` does -- seed the run's starting `dt`
    # directly from `targetDt` and let the `timestep` hook below take over
    # adaptively from the first step.
    ctx.config.dt = ctx.param('targetDt')


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


randomFlowIncompressibleCase = registerCase(Case(
    name='randomFlowIncompressible',
    scheme='divergenceFree',
    description='Decaying divergence-free random flow (2D), DFSPH, periodic or bounded.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='06-randomFlowIncompressible',
        # `kolmogorovIncompressibleTimestep` applies `cflFactor` to the
        # particle diameter, so this is Bender & Koschier's published 0.4
        # rather than the h-based 0.3 the weakly-compressible cases share.
        cflFactor=0.4,
        nx=128,
        L=2.0,
        tLimit=10.0,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        dt=1e-3,
        minDt=1e-7,
        maxDt=1e-1,
    ),
    params=dict(
        rho0=1.0,
        nu=0.0,
        targetDt=0.00025,
        band=0,
        freeSurface=False,
        # `--bounded` mirrors `randomFlow`'s 07-notebook variant; `--obstacle`
        # its circular cylinder.
        bounded=False,
        obstacle=False,
        **OBSTACLE_PARAMS,
        bandWidth=16.0,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        tileable=True,
        kind='perlin',
        seed=45906734,
        shifting=False,
        boundaryPressureMode='mdbcDensity',
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(randomFlowIncompressibleCase)
