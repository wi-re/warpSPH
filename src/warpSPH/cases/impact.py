"""Fluid bodies colliding (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/01-impact_spheres.ipynb` and
`02-impact-squares.ipynb`: two free-surface bodies given equal and opposite
velocities, differing only in the shape (spheres met along x, squares along y).
That is one case with the shape as a parameter -- and once the shape is a
parameter there is no reason for the *arrangement* not to be one too, so this
case covers the whole family the two notebooks were two points of:

- **any closed primitive** as the body shape (`--shape`, anything in
  :data:`~warpSPH.cases.weaklyCompressible.SHAPE_PRESETS` -- circle, box,
  hexagon, star5, ...), sized by one characteristic half-size and squashed by
  one aspect ratio, and turned on the spot by `--rotation`;
- **head-on, oblique or glancing**: `--impactAngle` turns both velocity
  vectors off the line joining the bodies, `--lateralOffset` slides the bodies
  off that line instead, and `--spin` gives each body a solid-body rotation;
- **two bodies or N**: `--arrangement ring --nBodies 5` puts them on a circle
  all moving inwards, which is the same experiment with more of it;
- **or anything else**, via the `bodies` parameter: an explicit list of body
  dicts, one per body, bypassing the arrangement entirely (config file or
  notebook only -- a list has no sensible command-line form).

Momentum is zero in every arrangement above -- the bodies are placed and given
their velocities as a mirrored pair (or an N-fold symmetric ring) -- so the
collision stays in the middle of the box, and what the run shows is the
free-surface deformation rather than a drifting blob.

Running it under the incompressible scheme
------------------------------------------

`--scheme divergenceFree` is supported, as baseline case 2 of
`DFSPH_IMPROVEMENT_PLAN.md`'s "what is left" item 2: the collision should
reproduce the weakly-compressible outcome -- the bodies meet in the middle and
merge into one central blob -- not pass through each other or explode. Two
things the `deltaSPH` path does not need:

- **Pass `--integrationScheme semiImplicitEuler`.** This case defaults to
  `rungeKutta2`, and the pressure-projection derivation is specific to
  semi-implicit Euler: a multi-stage integrator solves each stage as if it were
  final and then blends, so the blended velocity is not divergence-free.
  Nothing in the code enforces this yet (same as `dambreak`).
- **`impactTimestep` gives `--scheme divergenceFree` Bender & Koschier's
  advective CFL** instead of inheriting the weakly-compressible acoustic `dt`
  that `setupTimestep` fixes once at setup (DFSPH has no acoustic term).
  `deltaSPH` runs are untouched -- the hook returns `config.dt` unchanged for
  every other scheme, the same pattern `dambreakTimestep` uses.

The collision itself is scheme-agnostic: `gap` (nearest particle-to-particle
distance between the two bodies, by `materials`) and `comDrift` (the fluid
centre of mass away from its initial value) are reported by `diagnostics`
under both schemes, which is what makes the WC-vs-incompressible comparison
the same two columns in two runs.

Parameters, in the order they are applied:

===================  ======================================================
`shape`              body shape, one of `SHAPE_PRESETS`
`size`               characteristic half-size of a body
`aspectRatio`        squashes the shape in its second direction
`rotation`           degrees CCW, each body about its own centre
`arrangement`        `'pair'` (mirrored, along `impactAxis`) or `'ring'`
`nBodies`            ring only: how many bodies on the circle
`impactAxis`         pair only: 0 collides along x, 1 along y
`separation`         distance from the origin to each body's centre
`touching`           ignore `separation`; start the bodies `gap` apart
`gap`                particle spacings between the bodies when `touching`
`lateralOffset`      pair only: perpendicular offset -> off-centre impact
`impactVelocity`     speed of each body, towards the origin
`impactAngle`        degrees CCW, turns the velocities off that line
`spin`               rad/s of solid-body rotation, opposite on the pair
`bodies`             explicit body list, overriding all of the above
===================  ======================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from ..enumTypes import isIncompressibleScheme
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .plotting import Field, particlePlot
from .weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 centredShapeSdf, configureWeaklyCompressible, fluidRegion,
                                 paramExtraData, setupTimestep, shapeArgs,
                                 weaklyCompressibleDiagnostics)

__all__ = ['impactCase', 'IMPACT_FIELDS', 'bodySpecs', 'impactTimestep']

#: The two panels both impact notebooks plotted. `flare` rather than
#: `VELOCITY_DENSITY_FIELDS`' `RdBu`, which is what they had.
IMPACT_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='flare', flip=True, midPoint=1.0,
          vMin=0.99, vMax=1.01),
]


def _unit(angle: float) -> List[float]:
    return [math.cos(angle), math.sin(angle)]


def _rotated(vector, degrees: float) -> List[float]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]]


def bodySpecs(ctx: RunContext) -> List[Dict[str, Any]]:
    """The bodies to sample, as fully resolved dicts.

    Every key is explicit here rather than being re-read from the spec further
    down, so a notebook can print this, edit an entry, and hand the result back
    as the `bodies` parameter -- which is also the form the arrangement
    presets below produce.
    """
    explicit = ctx.param('bodies')
    if explicit:
        return [dict(defaultBody(ctx), **body) for body in explicit]

    arrangement = ctx.param('arrangement')
    speed = ctx.param('impactVelocity')
    angle = ctx.param('impactAngle')
    spin = ctx.param('spin')

    if arrangement == 'pair':
        axis = int(ctx.param('impactAxis'))
        along = [0.0, 0.0]
        across = [0.0, 0.0]
        along[axis], across[1 - axis] = 1.0, 1.0
        lateral = ctx.param('lateralOffset')
        bodies = []
        for sign in (-1.0, +1.0):
            outward = [sign * a for a in along]
            bodies.append(dict(
                defaultBody(ctx),
                direction=outward,
                # Opposite sideways offsets, so the pair misses head-on by
                # `2 * lateralOffset` and the collision carries angular
                # momentum instead of none.
                offset=[sign * lateral * a for a in across],
                velocity=[-speed * v for v in _rotated(outward, angle)],
                # Counter-rotating, like two gears; a common sense of rotation
                # would need the explicit `bodies` list.
                spin=sign * spin,
            ))
        return bodies

    if arrangement == 'ring':
        count = int(ctx.param('nBodies'))
        if count < 1:
            raise ValueError(f'nBodies must be at least 1, got {count}.')
        bodies = []
        for i in range(count):
            theta = 2.0 * math.pi * i / count + math.radians(ctx.param('ringPhase'))
            outward = _unit(theta)
            bodies.append(dict(defaultBody(ctx), direction=outward,
                               velocity=[-speed * v for v in _rotated(outward, angle)],
                               spin=spin))
        return bodies

    raise ValueError(f"Unknown arrangement {ctx.param('arrangement')!r}. "
                     "Known: 'pair', 'ring'.")


def defaultBody(ctx: RunContext) -> Dict[str, Any]:
    """The per-body defaults an arrangement (or the `bodies` list) fills in.

    A body's centre is `separation * direction + offset`. `direction` is a
    unit vector, because the separation is not known until the shape has been
    measured -- `touching` resolves it from the bodies' own extent -- while
    `offset` is an absolute displacement applied afterwards, which is what
    `lateralOffset` and any hand-placed body in the `bodies` list use.
    """
    return dict(shape=ctx.param('shape'), size=ctx.param('size'),
                aspect=ctx.param('aspectRatio'), rotation=ctx.param('rotation'),
                args=None, direction=[0.0, 0.0], offset=[0.0, 0.0],
                velocity=[0.0, 0.0], spin=0.0)


def _velocityField(velocity, spin: float):
    """`positions -> velocities` for a body translating and spinning rigidly.

    The spin is taken about the body's **sampled** centroid rather than the
    centre it was placed at: the particles all carry the same mass, so that is
    the centre of mass, and spinning about anything else would add a net
    translation to the body -- visible as a momentum imbalance in a case whose
    whole point is that the bodies meet in the middle.
    """
    def field(positions: torch.Tensor) -> torch.Tensor:
        base = torch.as_tensor(velocity).to(device=positions.device, dtype=positions.dtype)
        values = base.expand_as(positions).clone()
        if spin:
            radius = positions - positions.mean(dim=0, keepdim=True)
            # omega x r, in 2D: (-omega * ry, omega * rx).
            values[:, 0] -= spin * radius[:, 1]
            values[:, 1] += spin * radius[:, 0]
        return values
    return field


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)
    if not isIncompressibleScheme(ctx.scheme):
        return
    # `configureWeaklyCompressible` wires the shared `inviscid`/`alpha`
    # (artificial-viscosity) knobs `deltaSPH` uses; DFSPH has no such term
    # (see `kolmogorovIncompressible`), so with the WC default `inviscid=True`
    # the Monaghan-style term would run scaled by the WC sound speed
    # `setupTimestep` derives -- a dissipation this scheme was not given.
    # Same override `randomFlowIncompressible` makes.
    schemeConfig = ctx.schemeConfig
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    bodies = bodySpecs(ctx)
    domain = ctx.config.domain

    # Pass 1: measure every body at the origin, purely to learn how far it
    # reaches -- `touching` cannot say where the bodies go until it knows.
    for body in bodies:
        if body.get('args') is None:
            body['args'] = shapeArgs(body['shape'], body['size'], body['aspect'])
        _, halfExtent = centredShapeSdf(body['shape'], body['args'], [0.0, 0.0], domain,
                                        rotation=body['rotation'])
        body['halfExtent'] = [float(v) for v in halfExtent]

    dx = ctx.config.dx
    if ctx.param('touching'):
        # Each body's reach along its own placement direction; `lateralOffset`
        # deliberately does not count, it slides the bodies past each other
        # rather than apart. One `gap` of particle spacing between the two
        # surfaces is what stops them sampling particles on top of each other,
        # so round the separation *up* to keep it.
        reach = max(sum(abs(e * d) for e, d in zip(body['halfExtent'], body['direction']))
                    for body in bodies)
        separation = math.ceil((reach + ctx.param('gap') * dx) / dx) * dx
    else:
        separation = ctx.param('separation')
    ctx.scratch['separation'] = separation

    # Pass 2: the real placement, at the separation just resolved.
    regions = []
    for body in bodies:
        # Snapped to the particle lattice, as `dambreak` snaps its obstacle:
        # a mirrored pair displaced by a whole number of spacings samples two
        # congruent particle sets, so the momentum it starts with is exactly
        # zero rather than one stray particle's worth of it.
        body['centre'] = [round((separation * d + o) / dx) * dx
                          for d, o in zip(body['direction'], body['offset'])]
        sdf, _ = centredShapeSdf(body['shape'], body['args'], body['centre'], domain,
                                 rotation=body['rotation'])
        regions.append(fluidRegion(ctx, sdf, initialConditions={
            'velocities': _velocityField(body['velocity'], body['spin']),
        }))

    ctx.scratch['bodies'] = bodies
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    """Only the timestep: the velocities are the regions' own initial state.

    Each body carries its velocity (and its spin) as a `initialConditions`
    callable on its region, which is what `initializeState` evaluates per
    region -- so the two bodies are never told apart by the sign of a
    coordinate, and `arrangement='ring'`, overlapping bounding boxes and
    off-centre placements all work without a special case.
    """
    setupTimestep(ctx, system)

    # The `comDrift` figure of merit is measured from the centre of mass the
    # mirrored placement actually produces (exactly zero up to one stray
    # particle's worth of sampling error), not from the nominal origin.
    particles = system.state
    fluid = particles.kinds == 0
    ctx.scratch['initialCenterOfMass'] = (
        particles.positions[fluid] * particles.masses[fluid, None]).sum(dim=0) \
        / particles.masses[fluid].sum()


def impactTimestep(ctx: RunContext, state) -> float:
    """Bender & Koschier's advective CFL, but only under `--scheme
    divergenceFree`.

    `deltaSPH`'s own dt is fixed once, at setup, by `setupTimestep` in
    `initialConditions` above -- nothing in the run loop revisits `config.dt`
    for a case without a `timestep` hook, so returning it unchanged here
    reproduces that path exactly. `divergenceFree` has no acoustic term and
    needs a real advective dt instead, which is
    `kolmogorovIncompressibleTimestep`'s formula, reused as-is the same way
    `dambreakTimestep` reuses it.
    """
    if not isIncompressibleScheme(ctx.scheme):
        return ctx.config.dt
    return kolmogorovIncompressibleTimestep(ctx, state)


setupPlot, updatePlot = particlePlot(IMPACT_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    d = weaklyCompressibleDiagnostics(ctx, state)
    particles = state.state
    fluid = particles.kinds == 0
    if fluid.sum() == 0:
        return d

    # `materials` is the region's index in its type's list, assigned by
    # `initializeState` -- so for the `pair` arrangement (two fluid regions)
    # it labels the bodies. The `gap` is the nearest particle-to-particle
    # distance between them: it starts at the surface gap, closes to ~`dx` at
    # contact, and hovers there afterwards once the bodies have merged, which
    # is exactly the collision signature the case exists to grade.
    materials = particles.materials[fluid]
    if materials.unique().numel() == 2:
        posA = particles.positions[fluid & (particles.materials == 0)]
        posB = particles.positions[fluid & (particles.materials == 1)]
        d['gap'] = torch.cdist(posA, posB).min().detach().cpu().item()

    com0 = ctx.scratch.get('initialCenterOfMass')
    if com0 is not None:
        com = (particles.positions[fluid]
               * particles.masses[fluid, None]).sum(dim=0) \
            / particles.masses[fluid].sum()
        d['comDrift'] = torch.linalg.norm(com - com0).detach().cpu().item()
    return d


impactCase = registerCase(Case(
    name='impact',
    scheme='deltaSPH',
    description='Impact of two or more fluid bodies (2D), weakly compressible deltaSPH '
                '(also runnable under `--scheme divergenceFree`).',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=impactTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='01-impact',
        nx=256,
        L=4.0,
        tLimit=10.0,
        # The spheres notebook integrated with symplectic Euler; the squares
        # one used RK2. RK2 is the shared default and works for both.
        integrationScheme='rungeKutta2',
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,

        # -- the bodies ------------------------------------------------------
        shape='circle',
        size=0.5,
        aspectRatio=1.0,
        rotation=0.0,

        # -- where they start ------------------------------------------------
        arrangement='pair',
        nBodies=2,
        ringPhase=0.0,
        impactAxis=0,
        separation=0.75,
        touching=False,
        gap=1.0,
        lateralOffset=0.0,

        # -- how they move ---------------------------------------------------
        impactVelocity=0.5,
        impactAngle=0.0,
        spin=0.0,

        # Surface-detection bandwidth in particle spacings; read only under
        # `--scheme divergenceFree` (the WC path leaves the scheme default).
        bandWidth=16.0,

        # -- or: the bodies spelled out, one dict each ------------------------
        # Keys are `defaultBody`'s, plus an optional `args` overriding the
        # shape's preset argument list. `centre` is in units of `separation`.
        bodies=[],
    ),
))


if __name__ == '__main__':
    caseMain(impactCase)
