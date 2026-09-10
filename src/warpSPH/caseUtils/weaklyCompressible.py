"""Domain, obstacle, and forcing helpers for weakly-compressible cases.

Builds the periodic/semi-periodic/fully-periodic domain and interior-domain
pair (`buildDomain`), assembles fluid/boundary regions from SDFs including a
library of preset obstacle shapes (`buildPresetObstacles`, `buildObstacleSDF`,
`build_sdfs`, `buildRegions`), and provides optional post-init setup hooks for
divergence-free noise seeding, a freestream inflow ramp, and Kolmogorov
forcing (`sampleNoise`, `setupFreestream`, `setupKolmogorov`). All obstacle
geometry here is 2D (SDFs take `[..., 2]` points).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..configurations import BCType, BoundaryCondition, BoundaryConditionType, RegionType
from ..modules.noise.sampleDivergenceFree import (
    generateNoiseInterpolator,
    sampleDivergenceFreeNoise,
)
from ..regions.domainSDF import domainSDF, sampleDomainSDF
from ..regions.filter import filterRegion
from ..regions.region import buildRegion
from ..utils import buildDomainDescription
from ..math import getPeriodicPositions
from ..geometry.sdf import getSDF, operatorDict, sampleSDF


@dataclass
class SimulationProperties:
    device: torch.device
    dtype: torch.dtype

    nx: int
    dim: int
    L: float
    W: float

    dx: float
    band: int
    n_h: float
    targetDt: float
    freeSurface: bool
    semiPeriodic: bool
    fullyPeriodic: bool


def buildPresetObstacles(maxExtent: float, offsetX: float, L: float, fillRatio: float, angle: float):
    domainL = L / 2
    fillHeight = fillRatio * L

    obstacles = {
        "equilateralBottom": {
            "maxExtent": maxExtent,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 4,
            "aspectRatio": 2.0,
            "obstacleType": "equilateralTriangle",
            "aoa": 0.0,
        },
        "equilateralMiddle": {
            "maxExtent": maxExtent,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 2 + fillHeight / 2 - maxExtent / 2,
            "aspectRatio": 2.0,
            "obstacleType": "equilateralTriangle",
            "aoa": angle,
        },
        "equilateralTop": {
            "maxExtent": maxExtent,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 2.0,
            "obstacleType": "equilateralTriangle",
            "aoa": angle,
        },
        "triangleBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 8,
            "aspectRatio": 1.0,
            "obstacleType": "equilateralTriangle",
            "aoa": 0.0,
        },
        "triangleMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 8 + fillHeight / 2 - maxExtent / 4,
            "aspectRatio": 1.0,
            "obstacleType": "equilateralTriangle",
            "aoa": angle,
        },
        "triangleTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "equilateralTriangle",
            "aoa": angle,
        },
        "circleBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL,
            "aspectRatio": 1.0,
            "obstacleType": "circle",
            "aoa": 0.0,
        },
        "circleMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "circle",
            "aoa": 0.0,
        },
        "circleTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "circle",
            "aoa": 0.0,
        },
        "ellipsoidBottom": {
            "maxExtent": maxExtent * 0.5,
            "offsetX": offsetX,
            "offsetY": -domainL,
            "aspectRatio": 2.0,
            "obstacleType": "ellipse",
            "aoa": 0.0,
        },
        "ellipsoidMiddle": {
            "maxExtent": maxExtent * 0.5,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 2.0,
            "obstacleType": "ellipse",
            "aoa": angle,
        },
        "ellipsoidTop": {
            "maxExtent": maxExtent * 0.5,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 2.0,
            "obstacleType": "ellipse",
            "aoa": angle,
        },
        "squareBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 3,
            "aspectRatio": 1.0,
            "obstacleType": "box",
            "aoa": 0.0,
        },
        "squareMiddle": {
            "maxExtent": maxExtent / 3,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "box",
            "aoa": angle,
        },
        "squareTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "box",
            "aoa": angle,
        },
        "wallBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 2 - maxExtent / 3.0 * np.sin(np.abs(angle) * np.pi / 180),
            "aspectRatio": 3.0,
            "obstacleType": "box",
            "aoa": 90.0 + angle,
        },
        "wallMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 3.0,
            "obstacleType": "box",
            "aoa": 90.0 + angle,
        },
        "wallTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 3.0,
            "obstacleType": "box",
            "aoa": 90.0 + angle,
        },
        # `buildObstacleSDF` has always supported these four shapes, but no
        # preset ever exposed them, so no sweep matrix ever swept them either.
        "roundedBoxBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + maxExtent / 3,
            "aspectRatio": 1.0,
            "obstacleType": "roundedBox",
            "aoa": 0.0,
        },
        "roundedBoxMiddle": {
            "maxExtent": maxExtent / 3,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "roundedBox",
            "aoa": angle,
        },
        "roundedBoxTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "roundedBox",
            "aoa": angle,
        },
        "hexagonBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL,
            "aspectRatio": 1.0,
            "obstacleType": "hexagon",
            "aoa": 0.0,
        },
        "hexagonMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "hexagon",
            "aoa": angle,
        },
        "hexagonTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "hexagon",
            "aoa": angle,
        },
        "starBottom": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL,
            "aspectRatio": 1.0,
            "obstacleType": "star",
            "aoa": 0.0,
        },
        "starMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "star",
            "aoa": angle,
        },
        "starTop": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight,
            "aspectRatio": 1.0,
            "obstacleType": "star",
            "aoa": angle,
        },
        # Concave and orientation-dependent (the aperture faces `aoa`), unlike
        # the other, convex shapes -- Bottom/Top presets would just bury the
        # opening in the floor/ceiling, so only the submerged preset is useful.
        "horseshoeMiddle": {
            "maxExtent": maxExtent / 2,
            "offsetX": offsetX,
            "offsetY": -domainL + fillHeight / 2,
            "aspectRatio": 1.0,
            "obstacleType": "horseshoe",
            "aoa": angle,
        },
        # Marrone et al. 2011 Fig. 19 -- dam break against a sharp-edged
        # obstacle, with a concave quarter-circle fillet rounding the tank's
        # downstream bottom corner. Geometry is fixed by the figure (all of it
        # in units of the obstacle height H = W/10, see `_marroneSharpEdgeSDF`),
        # so `maxExtent` / `offsetX` / `aspectRatio` / `aoa` are ignored -- the
        # SDF is built straight from the domain `L`, `W`. The preset entry only
        # has to exist so `buildSystem`'s `presets.get(...)` succeeds.
        # `DELTASPH_VALIDATION_PLAN.md` Sec. 5.2.2.
        "marroneSharpEdge": {
            "maxExtent": maxExtent,
            "offsetX": 0.0,
            "offsetY": 0.0,
            "aspectRatio": 1.0,
            "obstacleType": "marroneSharpEdge",
            "aoa": 0.0,
        },
        # Same tank + rounded downstream corner as `marroneSharpEdge`, but with
        # the sharp-edged floor obstacle removed -- an isolated test of the
        # concave quarter-circle fillet: the dam-break surge runs the full 10 H
        # and impacts the smoothed corner directly.
        "marroneRoundedCorner": {
            "maxExtent": maxExtent,
            "offsetX": 0.0,
            "offsetY": 0.0,
            "aspectRatio": 1.0,
            "obstacleType": "marroneRoundedCorner",
            "aoa": 0.0,
        },
    }
    return obstacles


#: Marrone et al. 2011 Fig. 19, all lengths in obstacle heights H. The tank is
#: `10 H x 8 H`; the sharp-edged obstacle sits on the floor with its 45deg edge
#: apex at `x = 5 H` (measured from the upstream wall), toe at `6 H`, vertical
#: back face at `7 H`; the concave fillet (radius H) runs from `x = 9 H` on the
#: floor to `x = 10 H` at height H where it meets the downstream wall.
_MARRONE_SE = dict(apex_x=5.0, toe_x=6.0, back_x=7.0, fillet_x=9.0)


def _marroneSharpEdgeSDF(L: float, W: float, obstacle: bool = True):
    """`x -> signed distance` (negative inside solid) for the Marrone 2011
    Fig. 19 concave quarter-circle fillet at the tank's downstream bottom
    corner, **unioned with** the sharp-edged floor obstacle when
    `obstacle=True` (`marroneSharpEdge`) or on its own when `obstacle=False`
    (`marroneRoundedCorner` -- the isolated fillet test).

    Centred-domain coordinates: `x in [-W/2, W/2]`, `y in [-L/2, L/2]`, bed at
    `y = -L/2`, downstream wall at `x = +W/2`. `H = W / 10` so the fillet meets
    the wall exactly at its top point (`x = +W/2`, `y = -L/2 + H`); the caller
    (`scripts/probe_deltaSPHMarrone34.py`) is responsible for `W = 10 H` and
    `L = 8 H`.

    The obstacle is the convex quadrilateral `[apex, (back_x, top), (back_x,
    bed), toe]` -- the 45deg edge is the apex->toe side -- as a **single**
    half-plane-`max` SDF (`_convexPolygonSDF`). It used to be
    `min(triangle[apex,(toe_x,top),(toe_x,bed)], box[toe_x,back_x]x[bed,top])`,
    but a `min` of two primitives that only *touch* along `x = toe_x` leaves the
    merged value pinned near 0 all along that internal seam (both primitives are
    ~0 on their shared edge). A lattice column landing exactly on `toe_x` (it
    does: `toe_x = -W/2 + 6 H` is an integer number of `dx` from the centre)
    then samples `min = 0`, fails the `< 0` interior test, and drops a
    one-particle-wide gap up the whole seam that fluid slides into (~1.25 dx
    obstacle penetration, `DELTASPH_VALIDATION_PLAN` 5.2.2). One polygon has no
    internal seam.

    The fillet solid is the concave quarter-lens `box[wall-H, wall] x [bed,
    bed+H]` minus the disc of radius H centred at `(wall - H, bed + H)`.
    """
    r = _MARRONE_SE
    bed = -L / 2.0
    wall = W / 2.0
    H = W / 10.0
    top = bed + H
    apex_x = -wall + r['apex_x'] * H
    toe_x = -wall + r['toe_x'] * H
    back_x = -wall + r['back_x'] * H
    fillet_x = -wall + r['fillet_x'] * H            # == wall - H when W = 10 H

    boxfn = getSDF("box")["function"]
    circfn = getSDF("circle")["function"]

    discC = (fillet_x, top)
    filletBoxC = (wall - H / 2.0, bed + H / 2.0)
    #: wedge + toe-to-back-face block as one quadrilateral (apex -> along the
    #: roof to the back face -> down it -> along the floor -> up the 45deg edge).
    obstacleVerts = [(apex_x, top), (back_x, top), (back_x, bed), (toe_x, bed)]

    def sdf(x: torch.Tensor) -> torch.Tensor:
        dev, dt = x.device, x.dtype
        t = lambda v: torch.tensor(v, device=dev, dtype=dt)
        # concave fillet lens: difference(cornerBox, disc) = max(box, -discSDF)
        dFilletBox = boxfn(x - t(filletBoxC), t([H / 2.0, H / 2.0]))
        dOutDisc = -circfn(x - t(discC), t(H))
        dFillet = torch.maximum(dFilletBox, dOutDisc)
        if not obstacle:
            return dFillet
        dObstacle = _convexPolygonSDF(x, obstacleVerts)  # negative inside
        return torch.minimum(dObstacle, dFillet)

    return sdf


def _convexPolygonSDF(x: torch.Tensor, verts) -> torch.Tensor:
    """Signed distance (negative inside) to a **convex** polygon given by
    `verts` (a list of `(x, y)`, either winding), as the `max` over its edges of
    the outward half-plane distance. Exact everywhere inside and along an edge;
    just past a vertex it under-reads slightly (half-plane distance, not the
    point-to-vertex distance) -- irrelevant for interior sampling and for a
    near-surface normal. Only `+ - * @` and `torch.maximum`, so it composes and
    differentiates like the other primitives here without pulling in a new SDF.
    """
    K = len(verts)
    area2 = sum(verts[i][0] * verts[(i + 1) % K][1] - verts[(i + 1) % K][0] * verts[i][1]
                for i in range(K))
    ccw = area2 > 0.0
    V = torch.tensor(verts, device=x.device, dtype=x.dtype)
    d = x.new_full((x.shape[0],), -1e30)
    for i in range(K):
        a, b = V[i], V[(i + 1) % K]
        e = b - a
        nrm = torch.stack([e[1], -e[0]]) if ccw else torch.stack([-e[1], e[0]])
        nrm = nrm / nrm.norm().clamp_min(1e-12)
        d = torch.maximum(d, (x - a) @ nrm)
    return d


def _scale_points(points: torch.Tensor, scaleX: float, scaleY: float) -> torch.Tensor:
    new_points = points.clone()
    new_points[:, 0] = points[:, 0] / scaleX
    new_points[:, 1] = points[:, 1] / scaleY
    return new_points


def _translate_points(points: torch.Tensor, translateX: float, translateY: float) -> torch.Tensor:
    new_points = points.clone()
    new_points[:, 0] = points[:, 0] - translateX
    new_points[:, 1] = points[:, 1] - translateY
    return new_points


def _rotate_points(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    new_points = points.clone()
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    new_points[:, 0] = points[:, 0] * cos_angle - points[:, 1] * sin_angle
    new_points[:, 1] = points[:, 0] * sin_angle + points[:, 1] * cos_angle
    return new_points


def buildObstacleSDF(
    obstacleType: str,
    offsetX: float,
    offsetY: float,
    maxExtent: float,
    aspectRatio: float,
    aoa: float,
    config: Any,
    schemeConfig: Any,
    L: float,
    W: float | None = None,
):
    if W is None:
        W = L

    aoa_rad = aoa / 180 * np.pi
    scale = 1.0

    trs = lambda points: _scale_points(
        _rotate_points(
            _translate_points(points, offsetX, offsetY),
            torch.tensor(aoa_rad, device=points.device, dtype=points.dtype),
        ),
        scale,
        scale / aspectRatio,
    )

    if obstacleType == "circle":
        return lambda x: getSDF("circle")["function"](trs(x), torch.tensor(maxExtent, device=x.device, dtype=x.dtype))
    if obstacleType == "ellipse":
        return lambda x: getSDF("circle")["function"](trs(x), torch.tensor(maxExtent, device=x.device, dtype=x.dtype))
    if obstacleType == "box":
        return lambda x: getSDF("box")["function"](trs(x), torch.tensor([maxExtent, maxExtent], device=x.device, dtype=x.dtype))
    if obstacleType == "roundedBox":
        return lambda x: getSDF("roundedBox")["function"](
            trs(x),
            torch.tensor([maxExtent, maxExtent], device=x.device, dtype=x.dtype),
            torch.tensor([maxExtent / 5] * 4, device=x.device, dtype=x.dtype),
        )
    if obstacleType == "equilateralTriangle":
        return lambda x: getSDF("equilateralTriangle")["function"](trs(x), maxExtent)
    if obstacleType == "hexagon":
        return lambda x: getSDF("hexagon")["function"](trs(x), torch.tensor(maxExtent, device=x.device, dtype=x.dtype))
    if obstacleType == "horseshoe":
        aperture = np.pi / 4
        return lambda x: getSDF("horseshoe")["function"](
            trs(x),
            torch.tensor([np.sin(aperture), np.cos(aperture)], device=x.device, dtype=x.dtype),
            maxExtent * 0.85,
            maxExtent / 8,
        )
    if obstacleType == "star":
        return lambda x: getSDF("star5")["function"](trs(x), maxExtent, maxExtent * 1.25)
    if obstacleType == "marroneSharpEdge":
        # Fixed by Marrone 2011 Fig. 19 in units of H = W/10; `trs` (offset /
        # rotation / scale) does not apply. See `_marroneSharpEdgeSDF`.
        return _marroneSharpEdgeSDF(L, W)
    if obstacleType == "marroneRoundedCorner":
        return _marroneSharpEdgeSDF(L, W, obstacle=False)

    raise ValueError(f"Unsupported obstacleType: {obstacleType}")


def build_sdfs(config, schemeConfig, band: int, args, domain, interiorDomain, obstacle):
    fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert=True)
    union = lambda sdf1, sdf2: operatorDict["union"](sdf1, sdf2)

    fluidW = args.W
    fluidH = args.L * args.fillRatio
    box_sdf = lambda points: sampleSDF(
        points,
        operatorDict["translate"](
            lambda x: getSDF("box")["function"](
                x, torch.tensor([fluidW / 2, fluidH / 2], device=points.device, dtype=points.dtype)
            ),
            torch.tensor([0.0, interiorDomain.min[1] + fluidH / 2], device=points.device, dtype=points.dtype),
        ),
        invert=False,
    )

    maxExtent = obstacle["maxExtent"]
    aspectRatio = obstacle["aspectRatio"]
    offsetX = obstacle["offsetX"]
    offsetY = obstacle["offsetY"]
    aoa = obstacle["aoa"]

    domain_domain = copy.deepcopy(interiorDomain)
    if args.semiPeriodic:
        domain_domain.min[0] *= 1.5
        domain_domain.max[0] *= 1.5
    if args.fullyPeriodic:
        domain_domain.min[0] *= 1.5
        domain_domain.max[0] *= 1.5
        domain_domain.min[1] *= 1.5
        domain_domain.max[1] *= 1.5

    obstacle_sdf = None
    if args.obstacleActive:
        obstacle_sdf = buildObstacleSDF(
            obstacle["obstacleType"],
            offsetX,
            offsetY,
            maxExtent,
            aspectRatio,
            aoa,
            config,
            schemeConfig,
            args.L,
            args.W,
        )

    domain_sdf = lambda x: domainSDF(x, domain_domain, invert=False)
    merged_sdf = union(domain_sdf, obstacle_sdf) if args.obstacleActive else domain_sdf
    domain_sdf = lambda x: sampleSDF(x, merged_sdf, invert=False)

    regions = [
        buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions={}, shortEdge=args.W > args.L),
        buildRegion(
            config,
            schemeConfig,
            domain_sdf,
            RegionType.Boundary,
            initialConditions={},
            kind=BCType.noSlip,
            shortEdge=args.W > args.L,
        ),
    ]

    for region in regions:
        filterRegion(region, regions)

    return regions, fluid_sdf, domain_sdf, obstacle_sdf


def buildDomain(simSetup: SimulationProperties):
    device = simSetup.device
    dtype = simSetup.dtype
    domain = buildDomainDescription(
        simSetup.L + simSetup.dx * (simSetup.band) * 2,
        simSetup.dim,
        True,
        device,
        dtype,
    )
    domain.min = torch.tensor(
        [-simSetup.W / 2 - simSetup.dx * simSetup.band, -simSetup.L / 2 - simSetup.dx * simSetup.band],
        device=device,
        dtype=dtype,
    )
    domain.max = torch.tensor(
        [simSetup.W / 2 + simSetup.dx * simSetup.band, simSetup.L / 2 + simSetup.dx * simSetup.band],
        device=device,
        dtype=dtype,
    )

    if simSetup.semiPeriodic:
        domain.min = torch.tensor([-simSetup.W / 2, -simSetup.L / 2 - simSetup.dx * simSetup.band], device=device, dtype=dtype)
        domain.max = torch.tensor([simSetup.W / 2, simSetup.L / 2 + simSetup.dx * simSetup.band], device=device, dtype=dtype)

    if simSetup.fullyPeriodic:
        domain.min = torch.tensor([-simSetup.W / 2, -simSetup.L / 2], device=device, dtype=dtype)
        domain.max = torch.tensor([simSetup.W / 2, simSetup.L / 2], device=device, dtype=dtype)

    interiorDomain = buildDomainDescription(simSetup.L, simSetup.dim, False, device, dtype)
    interiorDomain.min = torch.tensor([-simSetup.W / 2, -simSetup.L / 2], device=device, dtype=dtype)
    interiorDomain.max = torch.tensor([simSetup.W / 2, simSetup.L / 2], device=device, dtype=dtype)
    return domain, interiorDomain


def alignInteriorDomainToLattice(domain, interiorDomain, dx, periodic=None):
    """Snap the interior (physical wall) box onto the sampling lattice so the
    innermost band of particles ends up ~dx/2 proud of every flat, grid-parallel
    wall -- i.e. so each wall falls *between* two lattice rows, not on top of one.

    Why this is legitimate as a pure pre-pass: the outer `domain` for these
    cases is always an axis-aligned box that never changes shape from run to
    run, and the regular lattice `sampleRegularParticles` lays over it is fixed
    once `domain` and `dx` are. Nothing about the physics pins the interior box
    to land at an arbitrary phase within that lattice; a walled tank is the same
    tank whether its wall sits at 5.00 dx or 5.43 dx from the domain edge. So we
    are free to move the wall by up to half a spacing to the nearest lattice
    mid-gap (ties broken *inward*, the "make the domain smaller" preference).
    `domain` -- hence the lattice, the neighbour search and every already-sampled
    particle -- is left exactly as it was. Moving the wall never opens a gap:
    the boundary and fluid rows straddling it are adjacent lattice points, `dn`
    apart no matter where between them the wall line is drawn.

    Without this, a lattice row landing on the wall gives a boundary layer a
    full dx out and a fluid row sitting on the wall (measured on the Marrone
    3.1 dam break at H/dx ~ 43: innermost boundary -0.95 dx, first fluid
    +0.02 dx, against the -0.5 / +0.5 dx ideal) -- which the mDBC ghost
    placement then has to paper over with geometry-retracting heuristics.

    Periodicity-aware: the explicit-`dx` sampler picks the same `nCells =
    ceil(l/dx - 1e-3)` either way, so `dn = l/nCells <= dx` is identical, but a
    *non-periodic* axis carries `nCells + 1` points (one on each face) at
    `centre + (k - nCells/2)*dn`, while a *periodic* axis carries `nCells`
    points at `centre + (k + 0.5 - nCells/2)*dn` (the half-cell wrap offset).
    That offset just flips which parity of `nCells` puts a mid-gap on the
    centre, so running this case with `wallPeriodic=True` lands the walls on the
    same +-dn/2 straddle and samples virtually identically -- identical fluid
    particle count, the same `dn`, differing only by a global dn/2 stagger of
    the lattice (physically irrelevant for a box) and one inert outer boundary
    row (the non-periodic closing endpoint). Mutates `interiorDomain.min` /
    `.max` in place and returns it.
    """
    lo = interiorDomain.min.clone()
    hi = interiorDomain.max.clone()
    per = domain.periodic if periodic is None else periodic
    for d in range(int(domain.min.shape[0])):
        isPeriodic = bool(per[d]) if per is not None else False
        pad = float(domain.max[d] - interiorDomain.max[d])
        if pad <= 0.5 * dx:
            continue                          # unpadded axis (truly periodic): no wall
        l = float(domain.max[d] - domain.min[d])
        nCells = max(1, int(np.ceil(l / dx - 1e-3)))   # matches buildPointCloud
        dn = l / nCells
        centre = 0.5 * float(domain.min[d] + domain.max[d])
        hNom = 0.5 * float(interiorDomain.max[d] - interiorDomain.min[d])
        # Mid-gap distances from the centre: `dn*{0.5, 1.5, ...}` when a lattice
        # *point* sits on the centre, `dn*{0, 1, 2, ...}` when a *mid-gap* does.
        # Non-periodic (points at integer*dn from centre): the former for even
        # `nCells`. Periodic (points at half-integer*dn -- the wrap offset): the
        # opposite parity. Snap the wall to the nearest such distance, ties
        # broken toward the smaller (inward) box -- so the move is <= dn/2.
        pointOnCentre = (nCells % 2 == 0) != isPeriodic
        half = 0.5 if pointOnCentre else 0.0
        # round-half-*down* (the `-1e-3` also absorbs float32 slop in `hNom/dn`
        # at a tie, where `hNom` sits exactly on a lattice point): nearest
        # mid-gap, ties inward.
        m = max(0, int(np.floor(hNom / dn - half + 0.5 - 1e-3)))
        hNew = dn * (m + half)
        if hNew <= 0.0 or abs(hNom - hNew) > dx + 1e-9:
            continue                          # nothing sensible within one spacing
        lo[d] = centre - hNew
        hi[d] = centre + hNew
    interiorDomain.min = lo
    interiorDomain.max = hi
    return interiorDomain


def buildRegions(config, schemeConfig, simSetup, args, domain, interiorDomain, obstacle):
    regions, _, domain_sdf, _ = build_sdfs(config, schemeConfig, args.band, args, domain, interiorDomain, obstacle)

    fluidW = args.fluidWidth * simSetup.W
    fluidH = args.fillRatio * simSetup.L
    box_sdf = lambda points: sampleSDF(
        points,
        operatorDict["translate"](
            lambda x: getSDF("box")["function"](
                x, torch.tensor([fluidW / 2, fluidH / 2], device=points.device, dtype=points.dtype)
            ),
            torch.tensor(
                [interiorDomain.min[0] + fluidW / 2, interiorDomain.min[1] + fluidH / 2],
                device=points.device,
                dtype=points.dtype,
            ),
        ),
        invert=False,
    )

    regions = [
        buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions={}, kind=BCType.constant),
        buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions={}),
    ]

    for region in regions:
        filterRegion(region, regions)

    return regions


def sampleNoise(compressibleSystem, config, schemeConfig, simSetup, args):
    if args.enableNoise:
        velocities = sampleDivergenceFreeNoise(
            compressibleSystem.state,
            config.domain,
            config,
            schemeConfig,
            int(simSetup.nx * 2),
            octaves=args.octaves,
            lacunarity=args.lacunarity,
            persistence=args.persistence,
            baseFrequency=args.baseFrequency,
            tileable=True,
            kind=args.kind,
            seed=args.seed,
        )
        compressibleSystem.state.velocities[:] = velocities * args.noiseAmplitude


def setupFreestream(compressibleSystem, config, schemeConfig, simSetup, args):
    if not args.enableFreestream:
        return

    rho0 = schemeConfig.fluid.restDensity
    W = simSetup.W
    L = simSetup.L

    u_freestream = args.freeStreamVelocity
    forcingWidth = args.forcingWidth

    forcingSDF = lambda points: sampleSDF(
        points,
        lambda x: getSDF("box")["function"](
            x, torch.tensor([W / 2 - forcingWidth, L / 2], device=points.device, dtype=points.dtype)
        ),
        invert=True,
    )

    def ldcForcing(state, cfg, schemeCfg, positions, d, n, t, dt):
        forcing = torch.zeros_like(state.velocities)
        v_diff = u_freestream - state.velocities[:, 0]
        forcing[:, 0] = v_diff * dt / 0.1
        return forcing

    ldcBC = BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=forcingSDF,
        forcingFunctions=[ldcForcing],
    )
    schemeConfig.boundaryConditions.append(ldcBC)

    minBoundaryDistance = torch.ones_like(compressibleSystem.state.positions[:, 0]) * np.inf
    for region in schemeConfig.regions:
        if region.type == RegionType.Boundary:
            distances = region.sdf(compressibleSystem.state.positions)[0]
            minBoundaryDistance = torch.min(minBoundaryDistance, distances)

    maxDistance = compressibleSystem.state.supports.max() * 2.0
    minBoundaryDistance = torch.clamp(minBoundaryDistance, min=0.0, max=maxDistance)
    ramp = minBoundaryDistance / maxDistance

    def rampFn(r):
        ramped = 15 / 8 * r - 10 / 8 * r**3 + 3 / 8 * r**5
        return torch.clamp(ramped, min=0.0, max=1.0)

    ramped = rampFn(ramp)
    fluid_mask = compressibleSystem.state.kinds == 0
    compressibleSystem.state.velocities[fluid_mask, 0] += u_freestream * ramped[fluid_mask]


def setupKolmogorov(compressibleSystem, config, schemeConfig, simSetup, args):
    if not args.enableKolmogorovForcing:
        return

    xi = args.kolmogorovForcingAmplitude
    k = args.kolmogorovForcingWavenumber
    noiseLevel = 0.01 * xi

    nxGrid = simSetup.nx * 2
    dim = simSetup.dim
    dtype = simSetup.dtype
    domain = config.domain

    # This domain only supplies the extents the noise grid is laid out over -- the
    # band-padded box, which is not config.domain -- so it stays on the host; the
    # noise field itself is moved to the simulation device by the first forcing call
    # and stays there.
    domain_cpu = buildDomainDescription(
        simSetup.L + simSetup.dx * simSetup.band * 2,
        dim,
        True,
        "cpu",
        dtype,
    )
    domain_cpu.min = torch.tensor(
        [-simSetup.W / 2 - simSetup.dx * simSetup.band, -simSetup.L / 2 - simSetup.dx * simSetup.band],
        device="cpu",
        dtype=dtype,
    )
    domain_cpu.max = torch.tensor(
        [simSetup.W / 2 + simSetup.dx * simSetup.band, simSetup.L / 2 + simSetup.dx * simSetup.band],
        device="cpu",
        dtype=dtype,
    )
    noiseGen = generateNoiseInterpolator(
        nxGrid,
        nxGrid,
        domain_cpu,
        dim=domain.dim,
        octaves=args.octaves,
        lacunarity=args.lacunarity,
        persistence=args.persistence,
        baseFrequency=args.baseFrequency,
        tileable=True,
        kind=args.kind,
        seed=args.seed,
    )

    def forcing(state, cfg, compParams, x, d, n, t, dt):
        pos = getPeriodicPositions(x, domain)
        u_x = xi * torch.sin(k * np.pi * pos[:, 1])
        # noiseGen samples a device-resident grid, so this stays on the GPU; it used to
        # be a scipy RegularGridInterpolator called on the host, which round-tripped the
        # whole position array per stage and cost ~68% of host self-time at 155k particles.
        # The positions stay detached: the noise perturbs the forcing but is not meant to
        # be optimized through, and the interpolator is differentiable now that it is torch.
        u_y = noiseGen(pos.detach()).to(dtype=x.dtype, device=x.device) * noiseLevel
        return torch.stack([u_x, u_y], dim=1) * state.masses.unsqueeze(1)

    kolmogorovForcing = BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=lambda x: (torch.ones_like(x[:, 0]) * -1.0, torch.zeros_like(x)),
        forcingFunctions=[forcing],
    )
    schemeConfig.boundaryConditions.append(kolmogorovForcing)


__all__ = [
    "SimulationProperties",
    "buildPresetObstacles",
    "buildObstacleSDF",
    "build_sdfs",
    "buildDomain",
    "buildRegions",
    "sampleNoise",
    "setupFreestream",
    "setupKolmogorov",
]
