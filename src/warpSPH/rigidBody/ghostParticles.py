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
the fluid for the first layer, ~`k dp` for the k-th. Two implementations here:

* **2D -- `_bodyNodeGhostOffsets` (default).** The boundary is discretised into
  "body nodes" -- a fine marching-squares polyline of the region SDF
  (`_boundaryPolyline`) -- and each boundary particle is mirrored **through its
  nearest point on that polyline**, `r_g = 2 r_s - r_b`. One rule covers every
  Marrone case: flat wall -> perpendicular foot -> standard mirror; convex
  solid wedge -> `r_s` collapses onto the apex -> central symmetry through the
  vertex; re-entrant corner -> `r_s` on the nearest wall -> mirror across it
  into the fluid. This is what fixed the Marrone §3.4 obstacle-toe leak, where
  the SDF-gradient direction (below) mis-placed ~28 % of the near-surface bed
  ghosts (`DELTASPH_VALIDATION_PLAN.md` §5.2.3).

* **3D, or no polyline -- `_geometricGhostOffsets`.** Mirror along `grad(sdf)`
  of the composed region SDF, `r_g = r_b - 2 phi(r_b) n_hat`. Fine on flats and
  smooth curves; **wrong at re-entrant / sharp corners** where `grad` of a
  `min`/`max` tree points into a wall.

Both then apply a **validity retract** -- shorten (never lengthen) the offset
so the node clears every solid region, capped at ~one support radius -- and
collapse to a zero offset where no clean fluid-facing node exists (deep solid
interior; `computeMdbcDensity` then Shepard-/rest-falls-back, the right answer
for a particle the fluid never reaches).

Not done (quality, not correctness): an analytic per-primitive normal would be
exact on curved walls and cheaper than the polyline; Marrone's explicit
bisector rule for re-entrant corners (mirror-through-nearest-point suffices for
the cases tried).
"""

import numpy as np
import torch
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion
from typing import Any, Optional, Union

__all__ = ['addBoundaryGhostParticles']

#: The fluid-facing interface sits this many spacings `dp` proud of the
#: outermost boundary layer (English et al. 2022 §3: `dp/2`).
GHOST_FLUID_DEPTH = 0.5

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
    """mDBC ghost offset `r_b - r_g` per boundary particle, placed the way the
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
    capMax = 1.5 * float(hMean)

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

    rg = 2.0 * foot - bpos                                  # mirror through r_s

    def clearance(pos):
        c = pos.new_full((pos.shape[0],), float('inf'))
        for sdf in solidSdfs:
            c = torch.minimum(c, sdf(pos)[0].reshape(-1))
        return c

    # Returned offset is `r_b - r_g` (points from the fluid back to the boundary
    # particle); the ghost node is `r_b - offset`.
    offFull = bpos - rg                                     # (n,2)
    tooFar = torch.linalg.norm(offFull, dim=-1) > capMax
    best = torch.zeros_like(bpos)
    found = (clearance(rg) > eps) & ~tooFar
    best = torch.where(found.unsqueeze(-1), offFull, best)
    if not bool(found.all()):
        # retract the node from the full mirror toward the surface point
        for frac in torch.linspace(1.0, 1.0 / nSamples, nSamples,
                                   device=bpos.device, dtype=bpos.dtype):
            off = frac * offFull
            cand = bpos - off
            take = (~found) & (clearance(cand) > eps) & \
                   (torch.linalg.norm(off, dim=-1) <= capMax)
            best = torch.where(take.unsqueeze(-1), off, best)
            found = found | take
    return best


def _geometricGhostOffsets(bpos, solidSdfs, dx, legacyOffsets, *, nSamples: int = 16):
    """Place each boundary particle's mDBC ghost node from the boundary geometry
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

            # Ghost placement, geometry-only. In 2D use the body-node method
            # (Marrone 2011 App. A / English §3): a polyline discretisation of
            # the surface, then mirror each boundary particle through its
            # nearest surface point -- which does the right thing at re-entrant
            # / sharp corners where `grad(sdf)` of a composed min/max points
            # into the wall (the Marrone §3.4 obstacle-toe leak, §5.2.3).
            offsets = None
            if int(dim) == 2:
                offsets = _bodyNodeGhostOffsets(bpos, region.sdf, solidSdfs, dx, hMean)
            if offsets is None:
                # 3D, or no polyline: English et al. 2022 reflection along
                # grad(sdf), capped, then a validity retract toward the boundary
                # particle where the node would cross into another solid.
                sdfValues, sdfNormals = region.sdf(bpos)
                clampedDist = sdfValues - torch.clamp(-sdfValues, max=1.5 * hMean)
                legacyOffsets = clampedDist.view(-1, 1) * sdfNormals
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
            
