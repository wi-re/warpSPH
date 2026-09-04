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

import numpy as np
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
    'particleDistributionMetrics', 'calibrateRestDensity', 'calibrateRestDensityMasses',
    'achievedLatticeSpacing', 'latticeDensityDecomposition',
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


def achievedLatticeSpacing(positions: torch.Tensor, dim: int) -> List[float]:
    """The lattice spacing the sampler actually laid down, per axis.

    NOT `config.dx`. The samplers fit the fluid block to its region and pick
    their own spacing, independently in each direction, so the lattice comes
    out slightly anisotropic (`sloshingTank` nx=60: sx 0.009375 vs sy 0.009400)
    and generally different from the nominal `dx` the particle mass was
    computed from -- which is the larger, resolution-dependent half of the
    initial density offset. Measured off the unique coordinates per axis, so it
    is meaningful only for an axis-aligned sampling, i.e. at initialisation and
    before any relaxation. Entries are `nan` for an axis with one layer.
    """
    pos = positions.detach().cpu().numpy()
    spacings = []
    for axis in range(dim):
        unique = np.unique(np.round(pos[:, axis], 9))
        spacings.append(float(np.median(np.diff(unique))) if len(unique) > 1
                        else float('nan'))
    return spacings


def latticeDensityDecomposition(ctx: RunContext, system,
                                measured: Optional[float] = None) -> Dict[str, float]:
    """Split the initial density offset into its two independent causes.

    `calibrateRestDensity` corrects a single measured number, but that
    number is a product of two unrelated errors, and only one of them is a
    sampling defect:

    * `latticeFactor` (`L`) -- the ideal-lattice quadrature offset. A perfect
      lattice at this `h/s` reads `rho0 * L` no matter how well it was sampled,
      because the kernel normalises its integral and not its lattice sum.
      Computed in closed form by `warpSPHCore.util.latticeDensity` from the
      kernel, the dimension and the achieved `n_h = h/s` alone -- no particles
      involved. For the Wendland family `L > 1` strictly at every `h`
      (`latticeDensityIsStrictlyAbove1`), so this term cannot be tuned away by
      widening the support; only bought down, as `n_h^-(d+2k+1)`.
    * `massRatio` -- `m / (rho0 * prod(s_i))`, the sampler's block fit. This is
      the resolution lottery: 1.0004 at `sloshingTank` nx=200 against **1.0142**
      at nx=100, and it is a genuine inconsistency between the mass a particle
      carries and the cell it actually occupies.

    Their product predicts the measured summation density (validated against
    live runs by `scripts/lattice_density_offset.py --validate`), so the
    residual `predicted / measured` is the check that the model is complete.

    Returns `{}` when the geometry cannot be read (non-lattice sampling, one
    layer per axis, fewer than 8 fluid particles).
    """
    from warpSPHCore.util import latticeDensity, latticeDensityIsStrictlyAbove1

    particles = system.state
    fluid = particles.kinds == 0
    if int(fluid.sum()) < 8:
        return {}
    dim = int(ctx.config.dim)
    spacings = achievedLatticeSpacing(particles.positions[fluid], dim)
    if not all(math.isfinite(s) and s > 0.0 for s in spacings):
        return {}
    cell = float(np.prod(spacings))
    # Geometric mean is the isotropic-equivalent spacing for `h/s`; the mass
    # ratio needs the true cell volume `prod(s_i)`, not `s**dim` from it --
    # using the latter mispredicts by exactly the anisotropy (2.7e-3 at nx=60).
    s = cell ** (1.0 / dim)
    h = float(torch.median(particles.supports[fluid].detach().float()))
    m = float(torch.median(particles.masses[fluid].detach().float()))
    rho0 = float(ctx.schemeConfig.fluid.restDensity)
    n_h = h / s
    L = latticeDensity(ctx.config.kernel, n_h, dim)
    massRatio = m / (rho0 * cell)
    # If the kernel already carries 1/L (config.calibrateNormalization, see
    # LATTICE_DENSITY_PLAN.md), the lattice term has been removed from the
    # measurement and only the sampler's block fit is left to predict. Reported
    # either way so the two routes stay comparable, but `predicted` has to
    # follow whichever one is live or the residual is meaningless.
    corrected = bool(getattr(ctx.config, 'calibrateNormalization', False))
    out = {
        'n_h': n_h,
        'latticeFactor': L,
        'massRatio': massRatio,
        'kernelCorrected': corrected,
        'predicted': massRatio if corrected else massRatio * L,
        'spacing': s,
        'irreducible': latticeDensityIsStrictlyAbove1(ctx.config.kernel),
    }
    if measured:
        out['measured'] = measured
        out['modelResidual'] = out['predicted'] / measured - 1.0
    return out


def calibrateRestDensity(ctx: RunContext, system, *,
                         quantile: float = 0.75,
                         tolerance: float = 1e-3,
                         onResidual: str = 'raise',
                         verbose: bool = False) -> float:
    """Make an at-rest sampling measure `rho0`, and catch it if it doesn't.

    Formerly `calibrateRestDensityMasses`: it scaled particle mass to paper
    over TWO separate offsets in the initial summation density --
    `massRatio` (the sampler assigning a particle a mass that did not match
    the cell it was actually placed in) and `latticeFactor` (`L`, the kernel's
    lattice-quadrature offset: `int W dV = 1` says nothing about `sum_j m_j
    W_ij`, so a defect-free lattice reads `rho0 * L(h/s) != rho0` no matter how
    well it was sampled). Both are still computed and reported here
    (`latticeDensityDecomposition`, recorded on
    `ctx.scratch['latticeDensitySplit']`), but they are no longer handled the
    same way:

    * `massRatio` is now `sample/regular.py`'s job, not this function's. It
      used to be the larger, resolution-dependent term (1.0004 at
      `sloshingTank` nx=200 vs **1.0142 at nx=100** -- the sampler's own block
      fit, root-caused to `buildPointCloud` computing mass from the nominal
      pre-snap spacing while placing particles at the achieved, per-axis
      post-snap one). Fixed at the source, it now measures `1 +/- ~1e-6` for
      any plain, unoptimized regular lattice -- see the **residual check**
      below.
    * `latticeFactor` is a *kernel normalisation* property, not a mass one --
      `C_d` normalises the kernel's integral, and what a lattice SUM needs is a
      slightly different constant. Folding it into mass rescaled every
      operator that touches `m` (momentum, continuity, every force) exactly as
      if `C_d` had been rescaled, while silently changing the fluid's total
      mass by ~0.1 %. The two routes happen to agree for the density sum and
      nowhere else. `LATTICE_DENSITY_PLAN.md` is that design; `calibrateRest`
      is what actually flips `ctx.config.calibrateNormalization` on, which
      applies `1/L` at the kernel level for the rest of the run -- so this
      function still calibrates a rest density, just via the one lever that
      is actually a density-normalisation constant.

    **The residual check.** With `massRatio` fixed at the sampler, ANY
    deviation of it from 1 -- for a plain regular, unjittered lattice -- is no
    longer expected sampling noise; it means something corrupted the initial
    state (overlapping regions, an SDF that clipped the lattice unevenly, an
    unwanted per-particle mass override elsewhere). `onResidual` controls what
    happens when `|massRatio - 1| > tolerance` (default `1e-3`, three orders
    above the ~1e-6 float32 noise measured across `nx` in [30, 333], four
    below the old bug): `'raise'` (default) stops the run with the offending
    numbers; `'warn'` prints and continues; `'ignore'` skips reporting. The
    check itself is skipped -- not raised, not warned -- whenever the sampling
    is not "regular and not optimized" (`ctx.spec.samplingScheme != 'regular'`
    or a nonzero `jitter` param), since neither this function nor the sampler
    fix promises anything about `massRatio` there.

    Call from `initialConditions`, **before** any analytic pressure/velocity
    seeding, and only for an at-rest sampling -- on a genuinely compressed
    initial state this would calibrate the physics away. Returns the measured
    reference density (1.0 means nothing needed doing).
    """
    from ..modules import computeDensities

    particles = system.state
    fluid = particles.kinds == 0
    if int(fluid.sum()) < 8:
        return 1.0
    density = computeDensities(particles, ctx.config, ctx.schemeConfig, None)
    reference = float(torch.quantile(density[fluid].detach().float(), quantile))
    if not (reference > 0.0):
        return 1.0
    rho0 = ctx.schemeConfig.fluid.restDensity

    split = latticeDensityDecomposition(ctx, system, measured=reference / rho0)
    ctx.scratch['latticeDensitySplit'] = split

    isRegular = (getattr(ctx.spec, 'samplingScheme', 'regular') == 'regular'
                and not ctx.param('jitter', 0.0))
    if split and isRegular and onResidual != 'ignore':
        residual = abs(split['massRatio'] - 1.0)
        if residual > tolerance:
            message = (
                f'[restDensityCalibration] massRatio = {split["massRatio"]:.6f} '
                f'(|residual| {residual:.1e} > tolerance {tolerance:.1e}) on a '
                f"regular, unjittered sampling -- sample/regular.py's mass fix "
                f'should make this ~1 exactly. This means the initial state is '
                f'not the clean lattice it is supposed to be (overlapping '
                f'regions, an uneven SDF clip, or a per-particle mass override '
                f'elsewhere) -- not something to calibrate away.')
            if onResidual == 'raise':
                raise ValueError(message)
            if onResidual == 'warn':
                import warnings
                warnings.warn(message)
            else:
                raise ValueError(f'onResidual must be raise/warn/ignore, got {onResidual!r}')

    if not ctx.config.calibrateNormalization:
        ctx.config.calibrateNormalization = True
        if verbose:
            check = computeDensities(particles, ctx.config, ctx.schemeConfig, None)
            now = float(torch.quantile(check[fluid].detach().float(), quantile))
            print(f'[restDensityCalibration] interior q{quantile:g} '
                  f'{reference:.6f} -> {now:.6f} (rho0={rho0}, '
                  f'calibrateNormalization enabled, mass untouched)')
    elif verbose:
        print(f'[restDensityCalibration] interior q{quantile:g} {reference:.6f} '
              f'(calibrateNormalization already on, mass untouched)')
    if verbose and split:
        print(f'[restDensityCalibration]   massRatio {split["massRatio"]:.6f} '
              f'(sampler) x latticeFactor {split["latticeFactor"]:.6f} at '
              f'n_h={split["n_h"]:.4f} (kernel)')
    return reference


#: Retired name -- see `calibrateRestDensity`'s docstring for what changed.
calibrateRestDensityMasses = calibrateRestDensity


def particleDistributionMetrics(ctx: RunContext, state) -> Dict[str, float]:
    """Is the particle *arrangement* physical? -- the blind spot of every
    field-valued metric in this file.

    Density is a kernel sum, so it is nearly blind to how the particles are
    actually laid out. Two particles sitting on top of each other still read
    `rho ~ rho0` because the neighbourhood compensates; a delaminated column of
    dense sheets separated by voids reads `rho ~ rho0` in every sheet. Measured
    on `band2018pb`: `hydrostaticColumn` nx=64 holds `rho` in [0.997, 1.007]
    -- a perfect-looking band -- while **7.2 % of its fluid particles are
    paired** at under half a spacing, and `impact` passes a `maxDensity` check
    while 27.9 % of particles are paired, 9.9 % have no neighbour within 2 dx,
    and the blob has hollowed out to a median density of 0.64.

    So these are geometric, not field, quantities:

    * `nnDistP01`      -- 1st percentile of nearest-neighbour distance, in units
                          of the run's OWN median spacing (self-normalising).
                          Collapses toward 0 under the pairing (tensile)
                          instability; ~1 for a healthy lattice.
    * `pairedFraction` -- fraction of fluid particles whose nearest neighbour
                          is closer than `dx/2`. The clumping signature.
    * `voidFraction`   -- fraction whose nearest neighbour is further than
                          `1.5 dx`. The void / de-densification signature.
    * `neighbourCountCV` -- coefficient of variation of the neighbour count.
                          Uniform sampling is ~0.1; layering and voids push it
                          up because sheets are over- and gaps under-populated.
    * `densityMedian`  -- the *bulk* density. `minDensity` at a free surface is
                          legitimately low (Part 33 spray, one-sided support),
                          but the median going low means the body itself is
                          de-densifying, which is never legitimate.

    Returns `{}` when no neighbour list is reachable (`system.adjacency` is set
    by the schemes), so this is safe to call from any case.
    """
    adjacency = getattr(state, 'adjacency', None)
    particles = state.state
    if adjacency is None or not hasattr(adjacency, 'i'):
        return {}
    fluid = particles.kinds == 0
    if int(fluid.sum()) == 0:
        return {}
    pos = particles.positions
    i, j = adjacency.i.long(), adjacency.j.long()
    # Fluid-FLUID pairs only. A wall is sampled as its own particle band that
    # can sit closer than dx/2 to the fluid it supports (a five-layer Akinci
    # band at a different effective spacing), so including fluid-boundary edges
    # reports a large constant "pairing" that is just the wall: measured, it
    # put `hydrostaticColumn` at 0.29 paired *at step 0*, before any physics,
    # while the wall-free `staticBlob` and `impact` read exactly 0.0. The
    # instability being measured is fluid particles collapsing onto each other.
    keep = (i != j) & fluid[i] & fluid[j]
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return {}
    delta = pos[i] - pos[j]
    domain = getattr(ctx.config, 'domain', None)
    if domain is not None and getattr(domain, 'periodic', None) is not None:
        span = (domain.max - domain.min).to(delta.dtype)
        per = domain.periodic.to(pos.device)
        wrapped = delta - span * torch.round(delta / span)
        delta = torch.where(per.unsqueeze(0), wrapped, delta)
    dist = torch.linalg.norm(delta, dim=-1)

    n = pos.shape[0]
    nn = torch.full((n,), float('inf'), device=pos.device, dtype=dist.dtype)
    nn.scatter_reduce_(0, i, dist, reduce='amin', include_self=True)
    counts = torch.zeros(n, device=pos.device, dtype=dist.dtype)
    counts.scatter_add_(0, i, torch.ones_like(dist))

    # Normalise by the MEDIAN nearest-neighbour distance, not `config.dx`.
    # The samplers do not lay particles down at `config.dx`: measured on
    # `sloshingTank`, the achieved lattice spacing is ~0.6 dx at every
    # resolution (nx=100: s 0.0054 vs dx 0.009). Dividing by `dx` therefore put
    # a healthy lattice at nn/dx ~ 0.6, so a "< 0.5 dx" pairing test was really
    # firing at 0.83 of the true spacing -- common and harmless -- and reported
    # a spurious ~6% "pairing" on runs that are known good. Self-normalising by
    # the median makes the ratio mean what it says on any sampling.
    nnF = nn[fluid]
    nnF = nnF[torch.isfinite(nnF)]
    if nnF.numel() == 0:
        return {}
    nnScale = torch.quantile(nnF.float(), 0.5).clamp_min(1e-30)
    nnF = nnF / nnScale
    cF = counts[fluid]
    rhoF = particles.densities[fluid].detach().float()
    return {
        'nnDistP01': torch.quantile(nnF.float(), 0.01).cpu().item(),
        'nnDistMedian': torch.quantile(nnF.float(), 0.5).cpu().item(),
        'pairedFraction': (nnF < 0.5).float().mean().cpu().item(),
        'voidFraction': (nnF > 1.5).float().mean().cpu().item(),
        'neighbourCountCV': (cF.std() / cF.mean().clamp_min(1e-9)).cpu().item(),
        'densityMedian': torch.quantile(rhoF, 0.5).cpu().item(),
    }


def weaklyCompressibleDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Kinetic energy, peak speed and the density bounds, over fluid only.

    Density bounds are the weakly compressible health check: the whole scheme
    rests on the fluid staying within about a percent of `rho0`.

    `minDensity` is reported but is a **poor figure of merit at a free
    surface**: it reads whichever one or two particles have been thrown clear
    of the bulk, whose density is low purely by kernel deficiency and which
    fall back a few steps later (DFSPH_FINDINGS.md Part 33 / Sec. 1.1 -- the
    finding that a "late-time degradation" was cosmetic ballistic spray, not
    structural loss). `densityP05`, the 5th percentile, is the spray-robust
    companion: it moves only when a real fraction of the fluid de-densifies.
    Grade free-surface runs on `densityP05`; keep `minDensity` for the
    periodic/confined cases, where it means what it says.
    """
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    velocities = particles.velocities[fluid]
    densities = particles.densities[fluid]
    out = {
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
        'densityP05': torch.quantile(
            densities.detach().float(), 0.05).cpu().item()
        if densities.numel() else float('nan'),
    }
    out.update(particleDistributionMetrics(ctx, state))
    return out


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
