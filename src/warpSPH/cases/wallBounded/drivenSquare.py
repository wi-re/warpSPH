"""Driven square (2D), weakly compressible.

The script form of this case was `examples/weaklyCompressible/11-driven-square.ipynb`.
A rigid body -- a square by default, `--obstacleShape` takes any key of
:data:`~warpSPH.cases.common.weaklyCompressible.SHAPE_PRESETS` -- **oscillates
sideways**, back and forth along x at `oscillationAmplitude` and
`oscillationPeriod`, starting from rest at `x = +oscillationAmplitude` and
swinging through an otherwise still fluid. That is what "driven" names here:
the body is what is being driven, not the flow. Compare against
`movingObstacle` (case 10, a body that spins in place instead of translating)
and `openFlowCase` (`13-open-flow.py`, a fixed obstacle in a channel with real
walls and inflow).

Three things follow from "back and forth" that did not apply to a body
translating in one direction forever:

- **The velocity is re-imposed every step**, not integrated once at t=0.
  `postStep` sets `RigidBody.linearVelocity` from the analytic
  `d/dt[A sin(2*pi*t/T)]` each step (`kidder.py`'s pattern for a boundary
  condition with a closed-form time dependence), and `initialConditions` calls
  the same formula once for the state the run starts in. `RigidBody.linearVelocity`
  is genuinely read every step by `rigidBody/integrate.py`'s Euler update of
  `centerOfMass` -- it is not a one-shot initial condition, so this is the
  intended way to drive it, not a workaround.
- **The domain has to be sized for the excursion, not just the body.** A body
  that only ever moves in one direction can be dropped into any periodic box;
  one that swings back and forth needs the box wide enough that it never
  approaches the wrap in either direction, or its own wake starts colliding
  with its periodic image before a single period completes. `domainMarginRatio`
  (default 2.0: the domain is at least twice as wide as the body's total
  excursion) is what keeps that true regardless of `obstacleSize`/
  `oscillationAmplitude` -- `configureScheme` below computes the domain's x
  extent from it rather than reusing the shared block's square box.
- **Starting the body at the domain centre moving at its peak speed is a real
  velocity discontinuity**, not just an odd-looking first frame -- see
  `oscillationVelocity`'s docstring for the measured effect on density. The
  body is sampled at `x = +oscillationAmplitude` (`buildSystem`) and released
  from rest instead.

The notebook this replaced had grown into something else entirely: a channel
(`W x L`, walls via `band`, `semiPeriodic`) with a fixed NACA aerofoil in a
freestream -- the dam-break/`channelFlow` machinery under different numbers,
not a driven square, and not oscillatory. This case starts over from
`movingObstacle` (translate instead of spin) with the domain-sizing above added
on top, and stays a still, unwalled box: no freestream, no channel walls.
`enableFreestream` (`movingObstacle`'s mean-flow forcing, layered under the
oscillation) is one real, separate experiment this case still supports; walled
confinement of a *moving* body is not, since that needs a stop-or-reflect
decision this case does not make for you -- `openFlowCase` already covers a
*fixed* obstacle inside real channel walls if that is what's wanted.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ...configurations.region import BCType
from ...runner import Case, RunContext, caseMain, registerCase
from ..common.plotting import particlePlot
from ..common.weaklyCompressible import (OBSTACLE_PARAMS, VELOCITY_DENSITY_FIELDS,
                                 WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS,
                                 boundaryRegion, buildRegionSystem,
                                 configureWeaklyCompressible, domainFluidSdf, fluidRegion,
                                 meanFlowForcingBC, paramExtraData, paramShapeSdf,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['drivenSquareCase', 'oscillationVelocity', 'sweptWidth']


def sweptWidth(ctx: RunContext) -> float:
    """Total x-range the body's own extent sweeps: `2 * (amplitude + size)`.

    `obstacleSize` is a half-extent (`SHAPE_PRESETS`'s convention), so the body
    itself reaches `size` past whichever end of the `[-amplitude, amplitude]`
    excursion it is currently at.
    """
    return 2.0 * (ctx.param('oscillationAmplitude') + ctx.param('obstacleSize'))


def oscillationVelocity(ctx: RunContext, t: float) -> torch.Tensor:
    """`d/dt[A cos(2*pi*t/T)]` as a `[vx, 0]` tensor -- sideways, along x.

    Zero at `t=0`: the body starts at **rest** at the `x = +amplitude` extreme
    of its swing (`buildSystem` places it there), not at the domain centre
    moving at its peak speed. The centre-with-peak-speed phase was the first
    version of this case and it is not a cosmetic choice -- an instantaneous
    jump from 0 to `amplitude*omega` in one step is a velocity discontinuity,
    and it measurably shocked the fluid: `run_sweep`-scale verification found
    density excursions of -7.2%/+5.7% against `movingObstacle`'s -0.3%/+1.8%
    at matched settings, for a case with no other difference from
    `movingObstacle` except this. Starting from rest removes the discontinuity
    (the acceleration is still maximal at `t=0`, but that alone did not
    produce a comparable shock in `movingObstacle`, which also starts its spin
    from a standing start).
    """
    amplitude = ctx.param('oscillationAmplitude')
    omega = 2.0 * math.pi / ctx.param('oscillationPeriod')
    vx = -amplitude * omega * math.sin(omega * t)
    return torch.tensor([vx, 0.0], device=ctx.device, dtype=ctx.dtype)


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)
    # The shared block builds a square box sized off `spec.L`; widen it in x to
    # fit the oscillation with margin, matching the y extent (`spec.L`) it
    # already gave the domain -- `sampleRegularParticles` derives `dx` from the
    # shorter edge, so this only adds particles along x, at the same spacing.
    width = max(float(ctx.config.domain.max[0] - ctx.config.domain.min[0]),
               ctx.param('domainMarginRatio') * sweptWidth(ctx))
    ctx.config.domain.min[0] = -width / 2
    ctx.config.domain.max[0] = width / 2


def buildSystem(ctx: RunContext):
    # Sample the body at x = +amplitude (plus whatever obstacleOffset asks for
    # on top), matching where oscillationVelocity's zero-at-t=0 phase expects
    # it to start -- see that function's docstring for why this matters.
    offset = list(ctx.param('obstacleOffset'))
    offset[0] += ctx.param('oscillationAmplitude')
    ctx.spec.params['obstacleOffset'] = offset

    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    return buildRegionSystem(ctx, [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, paramShapeSdf(ctx), kind=BCType.constant),
    ])


def initialConditions(ctx: RunContext, system) -> None:
    setupTimestep(ctx, system)

    ctx.config.rigidBodies[0].linearVelocity = oscillationVelocity(ctx, 0.0)
    ctx.schemeConfig.rigidBodies = ctx.config.rigidBodies

    if ctx.param('enableFreestream'):
        ctx.schemeConfig.boundaryConditions = [meanFlowForcingBC(
            ctx.scratch['fluidSdf'], ctx.param('U_target'), ctx.param('forcingTau'))]


def postStep(ctx: RunContext, state, step: int) -> None:
    """Re-impose the analytic oscillation velocity every step.

    `RigidBody.linearVelocity` is what `rigidBody/integrate.py` Euler-integrates
    `centerOfMass` from every step, held constant unless something sets it
    again -- so a one-shot assignment in `initialConditions` would leave the
    body translating in a straight line rather than oscillating.
    """
    ctx.config.rigidBodies[0].linearVelocity = oscillationVelocity(ctx, float(state.t))


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


drivenSquareCase = registerCase(Case(
    name='drivenSquare',
    scheme='deltaSPH',
    description='Square rigid body driven back and forth through fluid (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    postStep=postStep,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='11-drivenSquare',
        nx=128,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        # Shape/size/aspect/rotation/offset; a square (box, aspect 1) is what
        # this case is named for, any other key of SHAPE_PRESETS works.
        **dict(OBSTACLE_PARAMS, obstacleShape='box'),
        # How it moves: back and forth along x, x(t) = amplitude * cos(2*pi*t/period)
        # -- starting from rest at x = +amplitude, not from the centre at peak speed.
        oscillationAmplitude=0.5,
        oscillationPeriod=4.0,
        # The domain's x extent is at least this many times the body's total
        # excursion (2 * (amplitude + obstacleSize)) -- see the module docstring.
        domainMarginRatio=2.0,
        # Off by default -- see the module docstring. When on, the body
        # oscillates through a driven current rather than through still fluid.
        enableFreestream=False,
        U_target=1.0,
        forcingTau=0.5,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(drivenSquareCase)
