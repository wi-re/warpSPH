"""What every weakly compressible example case does the same way.

The `examples/weaklyCompressible/*.ipynb` notebooks all open with the same four
moves: widen the domain by `band` particle layers, describe the fluid and the
walls as SDF regions, sample them, and then pick the sound speed and `dt`
*together* from a target timestep. Only the SDFs and the initial velocity field
differ between them.

That common part is here. A case module supplies :func:`regions`-worth of SDFs
and an initial velocity field, and gets the rest.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..configurations import BoundaryCondition, BoundaryConditionType
from ..configurations.moduleConfigurations.gravity import GravityType
from ..configurations.region import BCType, RegionType
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..regions import buildRegion, filterRegion, sampleDomainSDF
from ..runner import RunContext, resolveEnum
from ..utils import buildDomainDescription
from ..geometry import getSDF, operatorDict, sampleSDF
from .plotting import Field

__all__ = [
    'WEAKLY_COMPRESSIBLE_DEFAULTS', 'WEAKLY_COMPRESSIBLE_PARAMS',
    'configureWeaklyCompressible', 'domainFluidSdf', 'domainBoundarySdf', 'shapeSdf',
    'SHAPE_PRESETS', 'shapeArgs', 'sdfBounds', 'centredShapeSdf',
    'OBSTACLE_PARAMS', 'paramShapeSdf',
    'buildRegionSystem', 'fluidRegion', 'boundaryRegion', 'meanFlowForcingBC',
    'setupTimestep', 'weaklyCompressibleDiagnostics', 'squarePatchAreaMetrics',
    'VELOCITY_DENSITY_FIELDS', 'VELOCITY_UID_FIELDS', 'paramExtraData',
]


#: `CaseSpec` fields shared by the weakly compressible examples.
WEAKLY_COMPRESSIBLE_DEFAULTS = dict(
    dim=2,
    kernel='Wendland4',
    integrationScheme='rungeKutta2',
    supportMode='KernelMeanSymmetric',
    gradientMode='Difference',
    laplacianMode='Brookshaw',
    samplingScheme='regular',
    periodic=True,
    n_h=4.0,
    # dt is set by `setupTimestep` from `targetDt` and the sound speed.
    dt=None,
    adaptiveDt=True,
    cflFactor=0.3,
    minDt=1e-8,
    plotInterval=60,
    storeInterval=500,
)

#: Case parameters shared by the weakly compressible examples.
WEAKLY_COMPRESSIBLE_PARAMS = dict(
    rho0=1.0,
    targetDt=0.0005,
    band=0,
    freeSurface=False,
    inviscid=True,
    nu=0.0,
    # The artificial-viscosity coefficient, i.e. the dissipation a run gets when
    # `inviscid` leaves it as the only one. It is the other half of the `nu`
    # knob rather than a separate mechanism -- the two are interconvertible
    # through `alphaToNu`/`nuToAlpha` given the sound speed and the support
    # radius -- and 0.01 is both the scheme default and the value below which
    # runs stop being reliably stable.
    alpha=0.01,
    # Uniform background pressure added to the EOS (WCSPH_SHIFTING_PLAN.md 2a).
    # 0.0 = off; a small positive value opposes the δ⁺ shift's outward
    # free-surface drift / the tensile instability, at the cost of rounding
    # sharp free-surface features.
    backgroundPressure=0.0,
    markerSize=4,
)

#: The two panels almost every weakly compressible notebook plotted.
VELOCITY_DENSITY_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0, vMin=0.99, vMax=1.01),
]

#: The same velocity panel, but coloured by particle UID instead of density --
#: the UIDs are handed out in sampling order, so the second panel is a dye trace
#: of where each particle started and reads as mixing rather than as a state
#: variable. A cyclic map keeps neighbouring UIDs distinguishable everywhere.
VELOCITY_UID_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('UIDs', 'particle UID', colorMap='twilight', colorMapKind='cyclic',
          midPoint=None),
]


def configureWeaklyCompressible(ctx: RunContext) -> None:
    """Domain, resolution and the scheme knobs the examples share.

    The simulated box is `band` particle layers wider than the *interior*
    domain on every side; the interior is what the walls are cut from, and it
    is stashed on `ctx.scratch` because the boundary SDFs need it.
    """
    dx = ctx.spec.L / ctx.spec.nx
    band = ctx.param('band')

    domain = buildDomainDescription(ctx.spec.L + dx * band * 2, ctx.spec.dim,
                                    ctx.spec.periodic, ctx.device, ctx.dtype)
    interior = buildDomainDescription(ctx.spec.L, ctx.spec.dim, False,
                                      ctx.device, ctx.dtype)
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx + band * 2
    ctx.scratch['interiorDomain'] = interior

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = ctx.param('freeSurface')
    if hasattr(schemeConfig.fluid, 'backgroundPressure'):
        schemeConfig.fluid.backgroundPressure = ctx.param('backgroundPressure', 0.0)
    schemeConfig.diffusionParams.inviscid = ctx.param('inviscid')
    schemeConfig.diffusionParams.inviscidAlpha = ctx.param(
        'alpha', schemeConfig.diffusionParams.inviscidAlpha)
    if not ctx.param('inviscid'):
        schemeConfig.diffusionParams.viscidNu = ctx.param('nu')

    if ctx.param('gravity', False):
        schemeConfig.gravityConfig.active = True
        schemeConfig.gravityConfig.type = resolveEnum(GravityType,
                                                      ctx.param('gravityType'))
        schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
        schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
    else:
        schemeConfig.gravityConfig.active = False


# -- SDF helpers -------------------------------------------------------------
# These are the three shapes the notebooks wrote inline, once each per file, as
# multi-line lambdas over `getSDF`/`operatorDict`.

def domainFluidSdf(ctx: RunContext) -> Callable:
    """Fluid filling the whole (band-widened) domain."""
    domain = ctx.config.domain
    return lambda x: sampleDomainSDF(x, domain, invert=True)


def domainBoundarySdf(ctx: RunContext) -> Callable:
    """Walls: everything outside the interior domain."""
    interior = ctx.scratch['interiorDomain']
    return lambda x: sampleDomainSDF(x, interior, invert=False)


def shapeSdf(name: str, size=None, offset=None, invert: bool = False, *,
             args: Optional[Sequence] = None, rotation: float = 0.0) -> Callable:
    """One of the built-in implicit shapes, optionally rotated and translated.

    `size` is whatever that shape's *single* argument is -- a radius for
    `circle`, half-extents for `box`. Shapes taking more than one argument
    (`trapezoid`, `star5`, `vesica`, ...) are given the whole list as `args`
    instead; :data:`SHAPE_PRESETS` builds that list from one characteristic
    size, so a case exposing `--shape` does not have to know each signature.

    `rotation` is in **degrees, counter-clockwise**, about the shape's own
    origin -- which is not the shape's centre for every primitive (see
    :func:`sdfBounds`) -- and is applied before `offset` translates it.
    """
    if args is None:
        if size is None:
            raise ValueError(f'shapeSdf({name!r}) needs either `size` or `args`.')
        args = [size]

    def sdf(points):
        device, dtype = points.device, points.dtype
        cast = [torch.as_tensor(a).to(device=device, dtype=dtype) for a in args]
        shape = lambda x: getSDF(name)['function'](x, *cast)
        if rotation:
            # op_rotate turns the *query point*, so the shape turns the other
            # way; negate to make a positive angle counter-clockwise.
            shape = operatorDict['rotate'](shape, -math.radians(rotation))
        if offset is not None:
            shape = operatorDict['translate'](
                shape, torch.as_tensor(offset).to(device=device, dtype=dtype))
        return sampleSDF(points, shape, invert=invert)
    return sdf


#: Closed 2D primitives usable as a fluid body or an obstacle, as
#: ``name -> (size, aspect) -> argument list`` for :func:`shapeSdf`.
#:
#: `size` is the shape's characteristic half-size and `aspect` squashes it in
#: its second direction (or picks the secondary radius, for the shapes built
#: from two of them), so `--shape hexagon --size 0.5` and
#: `--shape box --size 0.5 --aspectRatio 0.5` are both meaningful without
#: knowing that `sdBox` takes half-extents and `sdHexagon` takes a radius.
#:
#: These are the primitives from :data:`warpSPH.geometry.sdfFunctions` that
#: enclose a single connected area around their own origin. `segment` and
#: `polygon`/`star`/`ring` are left out: the first encloses nothing, and the
#: other three raise from `getSDF` as shipped.
SHAPE_PRESETS: Dict[str, Callable[[float, float], List[Any]]] = {
    'circle':              lambda size, aspect: [size],
    'box':                 lambda size, aspect: [[size, size * aspect]],
    'roundedBox':          lambda size, aspect: [[size, size * aspect],
                                                 [0.25 * size * aspect] * 4],
    'rhombus':             lambda size, aspect: [[size, size * aspect]],
    'trapezoid':           lambda size, aspect: [size, size * aspect, size],
    'parallelogram':       lambda size, aspect: [size, size * aspect, size * 0.5],
    'equilateralTriangle': lambda size, aspect: [size],
    'triangleIsosceles':   lambda size, aspect: [[size, 2.0 * size * aspect]],
    'pentagon':            lambda size, aspect: [size],
    'hexagon':             lambda size, aspect: [size],
    'octogon':             lambda size, aspect: [size],
    'hexagram':            lambda size, aspect: [size * 0.6],
    'star5':               lambda size, aspect: [size, 0.45 * aspect],
    'vesica':              lambda size, aspect: [size, size * aspect],
    'cutDisk':             lambda size, aspect: [size, -size * aspect],
    'unevenCapsule':       lambda size, aspect: [size * aspect, size * aspect * 0.5, size],
    'moon':                lambda size, aspect: [size * aspect, size, size * 0.75],
}


def shapeArgs(name: str, size: float, aspect: float = 1.0) -> List[Any]:
    """The :func:`shapeSdf` argument list for one of :data:`SHAPE_PRESETS`."""
    preset = SHAPE_PRESETS.get(name)
    if preset is None:
        raise ValueError(f'Unknown shape {name!r}. Known: {sorted(SHAPE_PRESETS)}.')
    return preset(size, aspect)


def sdfBounds(sdf: Callable, domain, resolution: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(centre, halfExtent)` of what `sdf` encloses, measured on a grid.

    The primitives are not all centred on their own origin -- `sdEquilateralTriangle`
    sits on its incircle, `sdTriangleIsosceles` grows upwards from the origin,
    `moon` and `cutDisk` are cut off-centre -- and a rotated shape is not
    centred even when the unrotated one is. Rather than tabulate a bounding box
    per primitive, measure it: one `resolution**dim` evaluation of the SDF,
    which costs nothing next to sampling the particles from it.

    Accurate to one grid cell, so it is for *placing* bodies (recentring,
    "start them just touching"), not for anything that needs the exact surface.
    """
    device, dtype = domain.min.device, domain.min.dtype
    axes = [torch.linspace(float(domain.min[i]), float(domain.max[i]), resolution,
                           device=device, dtype=dtype)
            for i in range(len(domain.min))]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, len(axes))
    distance, _ = sdf(grid)
    inside = grid[distance.detach() < 0]
    if inside.numel() == 0:
        raise ValueError('sdfBounds: the shape encloses no point of the domain -- '
                         'it is either empty, or larger than / outside the domain.')
    low, high = inside.min(dim=0).values, inside.max(dim=0).values
    return (low + high) / 2, (high - low) / 2


def centredShapeSdf(name: str, args: Sequence, centre, domain, rotation: float = 0.0,
                    invert: bool = False) -> Tuple[Callable, torch.Tensor]:
    """`(sdf, halfExtent)` for a shape whose *measured* centre lands on `centre`.

    Placing by the primitive's own origin puts, say, a triangle visibly off the
    mark it was given; this measures the shape where it lands
    (:func:`sdfBounds`) and translates by the residual instead.
    """
    atOrigin = shapeSdf(name, args=args, rotation=rotation, invert=invert)
    measured, halfExtent = sdfBounds(atOrigin, domain)
    target = torch.as_tensor(centre).to(device=measured.device, dtype=measured.dtype)
    return shapeSdf(name, args=args, rotation=rotation, invert=invert,
                    offset=(target - measured)), halfExtent


#: The five parameters a case declares to make one of its shapes selectable.
#: `paramShapeSdf` reads them; every case that has an obstacle uses the same
#: names, so `--obstacleShape star5` means the same thing in all of them.
OBSTACLE_PARAMS = dict(
    obstacleShape='circle',
    obstacleSize=0.25,
    obstacleAspect=1.0,
    obstacleRotation=0.0,
    obstacleOffset=[0.0, 0.0],
)


def paramShapeSdf(ctx: RunContext, prefix: str = 'obstacle') -> Callable:
    """SDF for the shape described by `<prefix>Shape/Size/Aspect/Rotation/Offset`.

    The placement is :func:`centredShapeSdf`'s: `Offset` is where the shape's
    *measured* centre goes, so an off-centre primitive (a triangle, a `moon`)
    lands where it was asked to rather than beside it.
    """
    name = ctx.param(f'{prefix}Shape')
    args = shapeArgs(name, ctx.param(f'{prefix}Size'), ctx.param(f'{prefix}Aspect', 1.0))
    sdf, _ = centredShapeSdf(name, args, ctx.param(f'{prefix}Offset', [0.0, 0.0]),
                             ctx.config.domain,
                             rotation=ctx.param(f'{prefix}Rotation', 0.0))
    return sdf


def buildRegionSystem(ctx: RunContext, regions: Sequence) -> Any:
    """Filter overlapping regions, register them, and sample the particles.

    `filterRegion` is what stops two touching fluid bodies from sampling
    particles on top of each other, so it has to run before the system is
    initialised, not after.
    """
    regions = list(regions)
    for region in regions:
        region = filterRegion(region, regions)
    ctx.config.regions = ctx.schemeConfig.regions = regions
    ctx.scratch['regions'] = regions

    return initializeWeaklyCompressibleSimulation(
        regions, ctx.config, ctx.schemeConfig,
        ctx.SimulationSystem, ctx.SimulationState, verbose=ctx.spec.verbose)


def fluidRegion(ctx: RunContext, sdf: Callable, **kwargs):
    """A fluid region. `initialConditions={'velocities': fn}` sets a field on it.

    Per-region initial conditions are how a case gives *this* body a velocity
    without having to tell it apart from the others afterwards by the sign of a
    coordinate: `initializeState` evaluates each callable over that region's
    own sampled positions. See `impact.py`.
    """
    kwargs.setdefault('initialConditions', {})
    return buildRegion(ctx.config, ctx.schemeConfig, sdf, RegionType.Fluid, **kwargs)


def boundaryRegion(ctx: RunContext, sdf: Callable, kind: BCType = BCType.freeSlip, **kwargs):
    kwargs.setdefault('initialConditions', {})
    return buildRegion(ctx.config, ctx.schemeConfig, sdf, RegionType.Boundary,
                       kind=kind, **kwargs)


def meanFlowForcingBC(fluidSdf: Callable, target: float, tau: float) -> BoundaryCondition:
    """Drive the domain-*mean* fluid velocity towards `(target, 0)` over `tau`.

    Forcing every particle towards the target individually would damp the
    fluctuations (a wake, a shed vortex) that are usually the point of the case
    this is used in -- correcting only the mean leaves them alone. Shared by
    `movingObstacle` (a spinning body in a driven current) and `drivenSquare`
    (a translating body, optionally in one).
    """

    def meanFlowForcing(state, config, schemeConfig, positions, d, n, t, dt):
        force = torch.zeros_like(state.positions)
        fluid = state.kinds == 0
        if torch.count_nonzero(fluid) == 0:
            return force
        mean = state.velocities[fluid].mean(dim=0)
        force[fluid, 0] = state.masses[fluid] * (target - mean[0]) / tau
        force[fluid, 1] = state.masses[fluid] * (-mean[1]) / tau
        return force

    return BoundaryCondition(type=BoundaryConditionType.dynamic, sdf=fluidSdf,
                             forcingFunctions=[meanFlowForcing])


def setupTimestep(ctx: RunContext, system) -> None:
    """Pick the sound speed and `dt` together, from `targetDt`.

    Weakly compressible SPH is free to choose its own stiffness, so rather than
    a sound speed being given and dt following, the notebooks fix the timestep
    they want and let the sound speed follow from the acoustic CFL. This is the
    call that finally sets `config.dt` for the run.
    """
    ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
        ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'),
        verbose=ctx.spec.verbose)


def weaklyCompressibleDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Kinetic energy, peak speed and the density bounds, over fluid only.

    Density bounds are the weakly compressible health check: the whole scheme
    rests on the fluid staying within about a percent of `rho0`.
    """
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    velocities = particles.velocities[fluid]
    densities = particles.densities[fluid]
    return {
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
    }


def _convexHullArea(points: 'Any') -> float:
    """Area of the 2D convex hull of `points` (N x 2 array-like), monotone
    chain + shoelace. `scipy` is not a dependency here, so this is hand-rolled;
    it is O(N log N) and called once per step, which is negligible next to a
    kernel sweep. Returns 0.0 for degenerate (< 3 distinct) inputs.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _halfHull(ordered):
        hull: List[Any] = []
        for p in ordered:
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                # cross((b - a), (p - a)) <= 0  ->  right turn / collinear
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)
        return hull[:-1]

    hull = np.array(_halfHull(pts) + _halfHull(pts[::-1]))
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def squarePatchAreaMetrics(ctx: RunContext, state) -> Dict[str, float]:
    """Area / volume-conservation diagnostics for the rotating square patch
    (`WCSPH_SHIFTING_PLAN.md` step 1). The δ⁺-SPH surface shift is not
    volume-preserving; these are what make its outward drift measurable.

    - ``sphVolume``      -- ``Σ_i m_i / ρ_i`` over fluid. The SPH volume; moves
      only with density error, so it is the cleanest volume-drift signal.
    - ``hullArea``       -- convex-hull area of the fluid point cloud. The
      "inflation" number: grows with the physical arms *and* with any outward
      surface drift, so read it against ``sphVolume`` and the ``--circle`` null.
    - ``rmsRadius``      -- mass-weighted RMS distance from the centre of mass.
      Grows if the patch spreads.
    - ``surfaceFraction``-- fraction of fluid particles flagged as free surface
      (``surfaceIndicators``); grows if the surface frays. ``nan`` when surface
      detection is off.
    - ``cornerRetention``-- current fluid extent along the initial corner
      diagonals ÷ its t=0 value. ``~1`` until the arms form; ``< 1`` is corner
      erosion. Baseline is cached in ``ctx.scratch`` on the first call.
    """
    import numpy as np

    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    positions = particles.positions[fluid]
    masses = particles.masses[fluid]
    densities = particles.densities[fluid]

    totalMass = masses.sum()
    centreOfMass = (masses.view(-1, 1) * positions).sum(dim=0) / totalMass
    offsets = positions - centreOfMass
    rmsRadius = torch.sqrt((masses * (offsets ** 2).sum(dim=-1)).sum() / totalMass)

    rotation = float(ctx.param('rotation')) if 'rotation' in ctx.spec.params else 0.0
    angles = rotation + np.deg2rad([45.0, 135.0, 225.0, 315.0])
    diagonals = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], axis=1),
                             dtype=positions.dtype, device=positions.device)
    # Signed extent of the cloud towards each initial corner, averaged.
    cornerExtent = (offsets @ diagonals.T).amax(dim=0).mean().detach().cpu().item()
    cornerExtent0 = ctx.scratch.setdefault('squarePatchCornerExtent0', cornerExtent)

    surfaceFraction = float('nan')
    indicators = getattr(particles, 'surfaceIndicators', None)
    if indicators is not None:
        indicators = indicators[fluid]
        surfaceFraction = (indicators == 1).sum().item() / max(indicators.shape[0], 1)

    return {
        'sphVolume': (masses / densities).sum().detach().cpu().item(),
        'hullArea': _convexHullArea(positions.detach().cpu().numpy()),
        'rmsRadius': rmsRadius.detach().cpu().item(),
        'surfaceFraction': surfaceFraction,
        'cornerRetention': cornerExtent / cornerExtent0 if cornerExtent0 else float('nan'),
    }


def paramExtraData(ctx: RunContext, state) -> Dict[str, Any]:
    """Record the case's scalar parameters on every exported frame."""
    data = {k: v for k, v in ctx.spec.params.items()
            if not isinstance(v, (list, dict))}
    data.update(nx=ctx.spec.nx, L=ctx.spec.L, n_h=ctx.spec.n_h,
                dx=ctx.config.dx, timeLimit=ctx.spec.tLimit)
    return data
