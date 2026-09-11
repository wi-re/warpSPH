"""Builds the mirrored boundary-ghost-particle layer that mDBC needs: for each
`RegionType.Boundary` region, projects its fluid-kind particles across the
fluid-facing interface into the fluid, appends the ghost particles to every
per-particle array, and records each boundary particle's ghost index/offset
(and the reverse, on the new ghost particle) on the returned state. Assumes one
region per boundary material ID, assigned in region order starting at 0
(`boundaryMaterial`); called once during initialization, from
`initializers/weaklyCompressible.py`.

**Ghost placement is a pure function of the boundary geometry**, evaluated once
here. It never consults the fluid, so the layer is identical whether the domain
starts wet, dry, or filling, and -- because the boundary is static in these
cases -- the ghost offsets never change during a run. Re-placing ghost
particles mid-simulation for a *static* boundary is a bug, not a feature: it
makes the discretisation depend on the solution. A genuinely moving boundary
would re-run this from the new geometry.

The method (Marrone 2011 App. A; English et al. 2022 §3): mirror each boundary
particle across the fluid-facing interface so its ghost node sits ~`dp/2` into
the fluid for the first layer, ~`k dp` for the k-th.

**Placement mode (`WARPSPH_GHOST_PLACEMENT`, or the `_GHOST_PLACEMENT_DEFAULT`
constant).** Four paths, but only **`'gridsnap'` is used** -- nothing in the
tree sets the env var. `'simple'`, `'bodynode'` and `'geometric'` are **parked**:
kept as troubleshooting aids (each isolates a different piece of the placement
-- raw mirror / polyline surface / analytic normal) for a geometry `'gridsnap'`
might get wrong, not as alternatives to select in normal use.

* **`'gridsnap'` -- `_gridSnapGhostOffsets` (default, the only live path).**
  Take the distance `d`
  behind, and inward normal `n`, from the **merged** solid SDF (every
  boundary/obstacle SDF `min`-combined, so a wall running into an obstacle is
  one surface with one normal through the junction -- `_mergedSurface`), then
  place the node at `ceil(d/dx)*dx` **past the surface**: `r_g = r_b + (d +
  ceil(d/dx)*dx) n`. Every first-layer node then sits exactly `dx` into the
  fluid, every second-layer node `2 dx`, etc. -- **independent of where the
  global lattice happened to fall against the surface**, so it needs no
  lattice alignment and copes with an obstacle that breaks it (the Marrone 3.4
  wedge on the floor). Deep layers (`d > 2 h`) -> zero offset; a node inside
  another solid is retracted one `dx` at a time, floored at `dx` past the
  surface.
* **`'simple'` -- the plain capped mirror, no validity pass.** `r_g = r_b - 2
  phi(r_b) n_hat` from this region's own SDF, node depth capped at `1.5 h`.
  Exactly right where the boundary particles are already well placed -- flat,
  grid-parallel walls with the first band `dp/2` proud
  (`alignInteriorDomainToLattice`) -- but phase-fragile once an obstacle breaks
  that alignment, which is why `'gridsnap'` supersedes it as the default.
* **`'bodynode'` -- 2D `_bodyNodeGhostOffsets`.** The boundary is discretised into
  "body nodes" -- a fine marching-squares polyline of the region SDF
  (`_boundaryPolyline`) -- and each boundary particle is mirrored **through its
  nearest point on that polyline**, `r_g = 2 r_s - r_b`. One rule covers every
  Marrone case: flat wall -> perpendicular foot -> standard mirror; convex
  solid wedge -> `r_s` collapses onto the apex -> central symmetry through the
  vertex; re-entrant corner -> `r_s` on the nearest wall -> mirror across it
  into the fluid. This is what fixed the Marrone §3.4 obstacle-toe leak, where
  the SDF-gradient direction (below) mis-placed ~28 % of the near-surface bed
  ghosts (`DELTASPH_VALIDATION_PLAN.md` §5.2.3).

* **`'geometric'` -- `_geometricGhostOffsets`** (also the fallback for 3D and
  for any 2D region with no extractable polyline). Mirror along `grad(sdf)` of
  the composed region SDF, `r_g = r_b - 2 phi(r_b) n_hat`. Fine on flats and
  smooth curves; **wrong at re-entrant / sharp corners** where `grad` of a
  `min`/`max` tree points into a wall.

The `'gridsnap'`, `'bodynode'` and `'geometric'` paths apply a **validity
retract** -- shorten (never lengthen) the offset so the node clears every solid
region -- and collapse to a zero offset where no clean fluid-facing node exists
(deep solid interior; `computeMdbcDensity` then Shepard-/rest-falls-back, the
right answer for a particle the fluid never reaches). `'simple'` does neither:
it trusts the sample.

Not done (quality, not correctness): an analytic per-primitive normal would be
exact on curved walls and cheaper than the polyline; Marrone's explicit
bisector rule for re-entrant corners (mirror-through-nearest-point suffices for
the cases tried).
"""

import os

import numpy as np
import torch
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion
from typing import Any, Optional, Union

__all__ = ['addBoundaryGhostParticles']

#: The fluid-facing interface sits this many spacings `dp` proud of the
#: outermost boundary layer (English et al. 2022 §3: `dp/2`).
GHOST_FLUID_DEPTH = 0.5

#: mDBC ghost placement path, `WARPSPH_GHOST_PLACEMENT`.
#:
#: **`'gridsnap'` is the only production path** and no case / script / test sets
#: the env var, so that is what everything runs. `'simple'`, `'bodynode'` and
#: `'geometric'` are **parked** -- kept solely as troubleshooting aids for a
#: geometry `'gridsnap'` might mishandle (they each isolate a different failure
#: mode: `'simple'` = the raw capped mirror with no validity pass at all;
#: `'bodynode'` = a marching-squares polyline instead of the merged SDF;
#: `'geometric'` = the analytic `grad(sdf)` normal instead of the mollified
#: central difference). Do not reach for them in normal use; if one of them
#: fixes a case that `'gridsnap'` breaks, that is a bug report for
#: `_gridSnapGhostOffsets`, not a mode to switch to.
#:
#:  * `'gridsnap'` -- node at `ceil(d/dx)*dx` past the merged solid surface, so
#:    the layout is independent of the sub-dx lattice phase; the one strategy
#:    that also copes with an obstacle touching the box wall
#:    (`_gridSnapGhostOffsets`).
#:  * `'simple'` (parked) -- the pristine capped mirror, no retract. Exact on a
#:    lattice-aligned flat wall (`alignInteriorDomainToLattice`); phase-fragile
#:    once an obstacle breaks the alignment.
#:  * `'bodynode'` (parked) -- the 2D marching-squares polyline mirror (Marrone
#:    2011 App. A). `'geometric'` (parked) -- grad(sdf) mirror + validity retract.
_GHOST_PLACEMENT_DEFAULT = 'gridsnap'
_GHOST_PLACEMENT_MODES = ('gridsnap', 'simple', 'bodynode', 'geometric')


def _ghostPlacementMode():
    m = os.environ.get('WARPSPH_GHOST_PLACEMENT', _GHOST_PLACEMENT_DEFAULT).strip().lower()
    return m if m in _GHOST_PLACEMENT_MODES else _GHOST_PLACEMENT_DEFAULT

#: Body-node polyline resolution, in samples per `dp`, for the 2D
#: `_bodyNodeGhostOffsets` path.
_BODYNODE_SAMPLES_PER_DP = 2.5


def _boundaryPolyline(regionSdf, bpos, dx, samplesPerDp=_BODYNODE_SAMPLES_PER_DP):
    """Marching-squares discretisation of this boundary region's fluid-facing
    surface into a segment set `(A, B)` -- the "body nodes" Marrone 2011 App. A
    and English et al. 2022 §3 place ghosts from. Built once, from the geometry
    (`regionSdf`, positive on the fluid side); no fluid consulted. 2D only.
    """
    from skimage import measure
    lo = bpos.amin(0) - 0.5
    hi = bpos.amax(0) + 0.5
    nx = max(16, int(((hi[0] - lo[0]) / dx * samplesPerDp).item()))
    ny = max(16, int(((hi[1] - lo[1]) / dx * samplesPerDp).item()))
    # Cap the marching-squares grid so a very fine `dp` cannot blow up the
    # one-off init cost (the SDF eval carries an autograd graph). ~1 sample/dp
    # still resolves the boundary well enough for the mirror.
    MAXCELLS = 4_000_000
    if nx * ny > MAXCELLS:
        s = (MAXCELLS / (nx * ny)) ** 0.5
        nx, ny = max(16, int(nx * s)), max(16, int(ny * s))
    gx = torch.linspace(float(lo[0]), float(hi[0]), nx, device=bpos.device, dtype=bpos.dtype)
    gy = torch.linspace(float(lo[1]), float(hi[1]), ny, device=bpos.device, dtype=bpos.dtype)
    GX, GY = torch.meshgrid(gx, gy, indexing='ij')
    F = regionSdf(torch.stack([GX.reshape(-1), GY.reshape(-1)], -1))[0].reshape(nx, ny)
    F = F.detach().cpu().numpy()
    verts = []
    for c in measure.find_contours(F, 0.0):
        p = np.empty_like(c)
        p[:, 0] = c[:, 0] / (nx - 1) * float(hi[0] - lo[0]) + float(lo[0])
        p[:, 1] = c[:, 1] / (ny - 1) * float(hi[1] - lo[1]) + float(lo[1])
        if len(p) >= 2:
            verts.append(p)
    if not verts:
        return None
    A = np.vstack([p[:-1] for p in verts])
    B = np.vstack([p[1:] for p in verts])
    seg = np.linalg.norm(B - A, axis=-1) > 1e-9
    A, B = A[seg], B[seg]
    t = lambda a: torch.as_tensor(a, device=bpos.device, dtype=bpos.dtype)
    return t(A), t(B)


def _bodyNodeGhostOffsets(bpos, regionSdf, solidSdfs, dx, hMean, *,
                          nSamples: int = 20, segChunk: int = 8192):
    """PARKED -- reachable only via `WARPSPH_GHOST_PLACEMENT=bodynode`, which
    nothing sets. Kept as a troubleshooting fallback that swaps the merged-SDF
    surface for a marching-squares polyline; `_gridSnapGhostOffsets` is the live
    path. See the module docstring.

    mDBC ghost offset `r_b - r_g` per boundary particle, placed the way the
    papers specify (Marrone 2011 App. A; English et al. 2022 §3): from an
    analytic-ish boundary discretisation, **not** the gradient of a composed
    `min`/`max` SDF (which points into the wall at a re-entrant corner -- the
    Marrone §3.4 obstacle-toe leak, `DELTASPH_VALIDATION_PLAN.md` §5.2.3).

    For each boundary particle: nearest point `r_s` on the boundary polyline,
    then **mirror through that point**, `r_g = 2 r_s - r_b`. This is one rule
    that covers all of Marrone's cases:

    * flat wall -> `r_s` is the perpendicular foot -> standard mirror;
    * convex solid wedge (concave fluid corner) -> `r_s` collapses onto the
      apex vertex -> central symmetry through the vertex (Fig. A.31-32);
    * re-entrant corner (fluid angle > pi, Fig. A.33) -> `r_s` on the nearest
      wall -> mirror across it into the fluid.

    Then a validity retract (shorten `r_g` toward `r_s`, never lengthen) so the
    node clears every solid, capped at ~one support radius from `r_b` -- past
    that the node is dropped (zero offset -> Shepard / rest fallback), which is
    Marrone's "not considered" rule and the right answer for a particle the
    fluid barely reaches.
    """
    poly = _boundaryPolyline(regionSdf, bpos, dx)
    if poly is None:
        return None
    A, B = poly
    n = bpos.shape[0]
    eps = 0.1 * float(dx)
    # `nodeDepthCap`: how far past the surface a ghost node may sit at a corner.
    # `layerReach`: a boundary particle whose nearest surface point is further
    # than this cannot be within kernel support of any fluid particle, so its
    # ghost is inert -- zero offset -> Shepard / rest fallback (Marrone's "not
    # considered"). On a smooth wall these bound the node *position*, never the
    # offset *magnitude*: a deep boundary layer of a multi-layer flat wall has a
    # legitimate offset of ~2|s| and rejecting it on magnitude drops the whole
    # deep layer to rest density, collapsing the wall pressure under a fast
    # near-wall flow (Marrone 2011 3.1 wall-climb blowup, DELTASPH_VALIDATION_PLAN 5.1.2).
    nodeDepthCap = 1.5 * float(hMean)
    layerReach = 2.0 * float(hMean)

    # nearest point on the polyline, chunked over segments
    foot = torch.zeros_like(bpos)
    dmin = bpos.new_full((n,), float('inf'))
    for s in range(0, A.shape[0], segChunk):
        a = A[s:s + segChunk].unsqueeze(0)                  # (1,S,2)
        ab = B[s:s + segChunk].unsqueeze(0) - a             # (1,S,2)
        tt = ((bpos.unsqueeze(1) - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
        tt = tt.clamp(0.0, 1.0)
        proj = a + tt.unsqueeze(-1) * ab                    # (n,S,2)
        d = torch.linalg.norm(bpos.unsqueeze(1) - proj, dim=-1)
        dv, di = d.min(dim=1)
        upd = dv < dmin
        dmin = torch.where(upd, dv, dmin)
        foot = torch.where(upd.unsqueeze(-1), proj[torch.arange(n, device=bpos.device), di], foot)

    def clearance(pos):
        c = pos.new_full((pos.shape[0],), float('inf'))
        for sdf in solidSdfs:
            c = torch.minimum(c, sdf(pos)[0].reshape(-1))
        return c

    # Mirror through the nearest surface point, node depth capped at
    # `nodeDepthCap` (a standard mirror otherwise puts a k-th-layer node ~k dp
    # deep). `dir` is the unit outward direction bpos -> foot -> fluid.
    dmn = dmin.unsqueeze(-1)
    dirUnit = (foot - bpos) / dmn.clamp_min(1e-12)
    mirrorDepth = torch.clamp(dmn, max=nodeDepthCap)        # (n,1)
    realLayer = dmin <= layerReach

    # Returned offset is `r_b - r_g` (fluid -> boundary particle); ghost node is
    # `r_b - offset`.
    rgCap = foot + mirrorDepth * dirUnit
    best = torch.zeros_like(bpos)
    found = realLayer & (clearance(rgCap) > eps)
    best = torch.where(found.unsqueeze(-1), bpos - rgCap, best)
    if not bool((found | ~realLayer).all()):
        # capped mirror landed in a solid (thin obstacle far face, the other
        # wall of a re-entrant corner): slide the node in toward the surface
        # foot along the same ray -- never past it into the solid.
        for frac in torch.linspace(1.0, 1.0 / nSamples, nSamples,
                                   device=bpos.device, dtype=bpos.dtype):
            cand = foot + frac * mirrorDepth * dirUnit
            take = realLayer & (~found) & (clearance(cand) > eps)
            best = torch.where(take.unsqueeze(-1), bpos - cand, best)
            found = found | take
    return best


def _geometricGhostOffsets(bpos, solidSdfs, dx, legacyOffsets, *, nSamples: int = 16):
    """PARKED -- reachable only via `WARPSPH_GHOST_PLACEMENT=geometric`, which
    nothing sets. Kept as a troubleshooting fallback that uses the analytic
    `grad(sdf)` normal (no mollifier) with a magnitude-only validity retract;
    `_gridSnapGhostOffsets` is the live path. See the module docstring.

    Place each boundary particle's mDBC ghost node from the boundary geometry
    alone -- no fluid particles consulted, so the layout is identical whether the
    domain is wet, dry, or filling, and never needs a mid-run refresh.

    Start from the English et al. 2022 / DualSPHysics reflection (`legacyOffsets`
    -- mirror across this region's SDF surface, `r_g = r_b - 2 phi(r_b) n_hat`).
    On a flat grid-aligned wall that node already sits `dp/2` into the fluid and
    is returned unchanged. Otherwise a geometry validity pass: step the offset
    magnitude down from the full reflection and take the *largest* value at which
    the node clears `dp/2` from **every** solid region -- the far face of a thin
    obstacle, the other wall of a re-entrant corner. If no point on the segment
    clears (a boundary particle with no clean fluid-facing interface), the offset
    collapses to zero: the node coincides with the boundary particle, its MLS
    moment matrix is singular, and `computeMdbcDensity` falls back to Shepard /
    rest density for that particle -- the correct behaviour there.

    `solidSdfs` is the list of every boundary region's `sdf` callable; each
    returns `(values, normals)` with `values > 0` on its fluid side.
    """
    m0 = torch.linalg.norm(legacyOffsets, dim=-1, keepdim=True)          # (n,1)
    nHat = legacyOffsets / m0.clamp_min(1e-12)                           # (n,D)

    # A node is "in the fluid" for a solid region when that region's SDF reads
    # positive there. `eps` sits well below the `dp/2` a clean layer-1 node
    # keeps from its *own* wall, but above zero, so it only trips when the node
    # has actually crossed into another solid.
    eps = 0.1 * float(dx)

    def clearance(pos):
        c = pos.new_full((pos.shape[0],), float('inf'))
        for sdf in solidSdfs:
            c = torch.minimum(c, sdf(pos)[0].reshape(-1))
        return c

    best = m0.clone()
    ok = clearance(bpos - m0 * nHat) > eps                               # (n,)
    if not bool(ok.all()):
        found = ok.clone()
        for frac in torch.linspace(1.0, 1.0 / nSamples, nSamples,
                                   device=bpos.device, dtype=bpos.dtype):
            m = m0 * frac
            take = (~found) & (clearance(bpos - m * nHat) > eps)
            best = torch.where(take.view(-1, 1), m, best)
            found = found | take
        best = torch.where(found.view(-1, 1), best, torch.zeros_like(best))
    return best * nHat


def _mergedSurface(pos, solidSdfs, eps):
    """Distance `d` behind, and unit inward normal `n`, of the *merged* solid
    surface at each `pos` -- every boundary/obstacle SDF `min`-combined into one,
    so a wall that runs into an obstacle (the Marrone 3.4 wedge sits on the
    floor) is treated as one continuous surface with one consistent normal
    through the junction, not two primitives fighting over the corner.

    A point is in the fluid only if it is in the fluid of *every* solid, so the
    merged value is `min_i sdf_i` (each `> 0` on its fluid side); `d = max(-min,
    0)` is how far the point sits behind that surface. The normal is a **central
    difference of the merged value over a step `eps ~ dx`**, not the analytic
    gradient: the finite step mollifies the direction discontinuity at a
    convex edge / re-entrant corner of the `min`/`max` tree (which is exactly
    where the raw gradient points into a wall -- the Marrone 3.4 toe leak).
    """
    def mval(p):
        v = None
        for s in solidSdfs:
            vi = s(p)[0].reshape(-1)
            v = vi if v is None else torch.minimum(v, vi)
        return v

    d = (-mval(pos)).clamp_min(0.0)
    grad = torch.zeros_like(pos)
    for i in range(pos.shape[-1]):
        o = torch.zeros_like(pos)
        o[..., i] = eps
        grad[..., i] = (mval(pos + o) - mval(pos - o)) / (2.0 * eps)
    return d, torch.nn.functional.normalize(grad, dim=-1)


def _gridSnapGhostOffsets(bpos, solidSdfs, dx, hMean, *, nRetract: int = 12):
    """mDBC ghost offset `r_b - r_g`, placed so the layout is **independent of
    where the global particle lattice happened to fall against the surface**
    (`DELTASPH_VALIDATION_PLAN.md` 5.2.x).

    A plain mirror puts the node a distance `d` (the boundary particle's own
    distance behind the surface) into the fluid -- and with an obstacle in the
    box, `d` is whatever the lattice phase gave, anywhere in `(0, dx)` for the
    first layer, so the node can land right on the surface with no one-sided
    support. Instead: take `d` and the mollified inward normal `n` from the
    merged solid SDF (`_mergedSurface`), and place the node at
    **`(floor(d/dx) + 0.5)*dx` past the surface** -- the fluid-particle lattice
    phase. When the boundary and fluid bands straddle the wall cleanly (the
    usual case after `alignInteriorDomainToLattice`), the first fluid row sits
    `0.5 dx` past the surface, the second `1.5 dx`, etc., so the node lands
    *exactly on a fluid particle* (one-sided but well-conditioned MLS) rather
    than `0.5 dx` into the gap between two fluid rows -- which `ceil(d/dx)*dx`
    (an integer count from the *surface*, not the fluid) did, biasing the
    extrapolation at every wall particle. Node depth is capped at `1.5 h` so a
    slightly-off deep-layer normal cannot fling the node across the domain.

    Deep layers (`d > 2 h`, past any fluid particle's kernel support) get a zero
    offset -> Shepard / rest fallback (Marrone's "not considered"). A node that
    lands inside another solid (thin obstacle far face, opposite wall of a
    narrow gap) is retracted half a `dx` at a time toward the surface; if it
    never clears, the offset collapses to zero.
    """
    eps = 0.1 * float(dx)
    d, n = _mergedSurface(bpos, solidSdfs, float(dx))
    realLayer = d <= 2.0 * float(hMean)
    nodeDepth = ((torch.floor(d / dx - 1e-3).clamp_min(0.0) + 0.5) * dx).clamp_max(1.5 * float(hMean))

    def clearance(pos):
        c = pos.new_full((pos.shape[0],), float('inf'))
        for s in solidSdfs:
            c = torch.minimum(c, s(pos)[0].reshape(-1))
        return c

    best = torch.zeros_like(bpos)
    found = ~realLayer                                                        # deep: leave at 0
    for k in range(nRetract):
        depth = (nodeDepth - k * 0.5 * dx).clamp_min(0.5 * dx)
        rg = bpos + (d + depth).unsqueeze(-1) * n
        take = (~found) & (clearance(rg) > eps)
        best = torch.where(take.unsqueeze(-1), bpos - rg, best)
        found = found | take
    return best


def addBoundaryGhostParticles(regions, particleState : Any):
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    
    if not torch.any(particleState.kinds == 1):
        # print("No Boundary particles found. Returning original state.")
        return particleState
    
    ghostIndices = particleState.positions.new_ones(particleState.positions.shape[0], dtype = torch.int64) * -1
    ghostOffsets = torch.zeros_like(particleState.positions)
    
    boundaryMaterial = 0
    numParticles = particleState.positions.shape[0]
    ghostPositions = []
    boundaryIndices = []

    # Every solid region's SDF (values > 0 on its fluid side). The ghost
    # validity pass checks a candidate node against all of them, so a ghost is
    # never placed inside another wall or the far face of a thin obstacle.
    solidSdfs = [r.sdf for r in regions if r.type == RegionType.Boundary]

    for region in regions:
        # print(f"Processing region {region} of type {region.type}")
        if region.type == RegionType.Boundary:
            particleIndices = torch.arange(particleState.positions.shape[0], device = device, dtype = torch.int64)
            relevantParticles = torch.logical_and(particleState.kinds == 1, particleState.materials == boundaryMaterial)
            # print('Boundary region', boundaryMaterial, 'has', torch.sum(relevantParticles).item(), 'fluid particles.')
            relevantParticles = particleIndices[relevantParticles]

            dim = float(particleState.positions.shape[-1])
            dx = particleState.masses.mean() ** (1.0 / dim)
            bpos = particleState.positions[relevantParticles]
            hMean = float(particleState.supports.mean())

            # Ghost placement, geometry-only. `'gridsnap'` is the only live
            # path; the other branches are parked troubleshooting aids (see the
            # module docstring -- nothing sets `WARPSPH_GHOST_PLACEMENT`).
            mode = _ghostPlacementMode()
            offsets = None
            if mode == 'gridsnap':
                # Node at ceil(d/dx)*dx past the merged solid surface -- layout
                # independent of the sub-dx lattice phase, copes with an
                # obstacle touching the box wall.
                offsets = _gridSnapGhostOffsets(bpos, solidSdfs, dx, hMean)
            else:
                # Parked paths. The English et al. 2022 reflection along
                # grad(sdf), node depth capped at 1.5 h, is their shared start.
                sdfValues, sdfNormals = region.sdf(bpos)
                clampedDist = sdfValues - torch.clamp(-sdfValues, max=1.5 * hMean)
                legacyOffsets = clampedDist.view(-1, 1) * sdfNormals
                if mode == 'simple':
                    offsets = legacyOffsets                      # raw capped mirror, no retract
                elif mode == 'bodynode' and int(dim) == 2:
                    offsets = _bodyNodeGhostOffsets(bpos, region.sdf, solidSdfs, dx, hMean)
                if offsets is None:                              # 'geometric', or 3D bodynode
                    offsets = _geometricGhostOffsets(bpos, solidSdfs, dx, legacyOffsets)

            bIndices = relevantParticles
            boundaryIndices.append(bIndices)
            gUIDs = particleState.UIDs[relevantParticles]
            gIndices = torch.arange(numParticles, numParticles + offsets.shape[0], device = device, dtype = torch.int64)
            ghostIndices[bIndices] = gIndices
            ghostOffsets[bIndices] = offsets
            # print(bIndices, gIndices)

            # ghostIndices[gIndices] = bIndices
            # ghostOffsets[gIndices] = -offsets
            numParticles += offsets.shape[0]
            # print(sdfValues)
            
            
            ghostPositions.append(particleState.positions[relevantParticles] - offsets)
            boundaryMaterial += 1

            # print(f"Added {offsets.shape[0]} ghost particles for boundary region {region}.")
            
    boundaryIndices = torch.cat(boundaryIndices, dim = 0)
    WeaklyCompressibleState = type(particleState)
    return WeaklyCompressibleState(
        positions = torch.cat([particleState.positions, torch.cat(ghostPositions, dim = 0)], dim = 0),
        supports = torch.cat([particleState.supports, particleState.supports[boundaryIndices]], dim = 0),
        masses = torch.cat([particleState.masses, particleState.masses[boundaryIndices]], dim = 0),
        densities = torch.cat([particleState.densities, particleState.densities[boundaryIndices]], dim = 0),
        velocities = torch.cat([particleState.velocities, particleState.velocities[boundaryIndices]], dim = 0),
        
        pressures= torch.cat([particleState.pressures, particleState.pressures[boundaryIndices]], dim = 0) if particleState.pressures is not None else None,
        soundspeeds= torch.cat([particleState.soundspeeds, particleState.soundspeeds[boundaryIndices]], dim = 0) if particleState.soundspeeds is not None else None,
        
        kinds = torch.cat([particleState.kinds, torch.ones_like(boundaryIndices).to(particleState.kinds.dtype) * 2], dim = 0),
        materials = torch.cat([particleState.materials, particleState.materials[boundaryIndices]], dim = 0),
        UIDs = torch.cat([particleState.UIDs, -particleState.UIDs[boundaryIndices]], dim = 0),
        
        ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0),
        ghostOffsets= torch.cat([ghostOffsets, -ghostOffsets[boundaryIndices]], dim = 0),
        
        UIDcounter = particleState.UIDcounter
    )
            
