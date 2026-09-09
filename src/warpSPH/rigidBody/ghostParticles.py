"""Builds the mirrored boundary-ghost-particle layer that mDBC needs: for each
`RegionType.Boundary` region, projects its fluid-kind particles across the
fluid-facing interface into the fluid, appends the ghost particles to every
per-particle array, and records each boundary particle's ghost index/offset
(and the reverse, on the new ghost particle) on the returned state. Assumes one
region per boundary material ID, assigned in region order starting at 0
(`boundaryMaterial`); called once during initialization, from
`initializers/weaklyCompressible.py`.

**Ghost placement is a pure function of the boundary geometry**, evaluated once
here (`_geometricGhostOffsets`). It never consults the fluid, so the layer is
identical whether the domain starts wet, dry, or filling, and -- because the
boundary is static in these cases -- the ghost offsets never change during a
run. Re-placing ghost particles mid-simulation for a *static* boundary is a
bug, not a feature: it makes the discretisation depend on the solution. A
genuinely moving boundary would re-run this from the new geometry.

The method (English et al. 2022 §3, matching DualSPHysics
`InteractionMdbcCorrectionT2`): each boundary particle's ghost node is the
particle mirrored across the region SDF surface along its gradient,
`r_g = r_b - 2 phi(r_b) n_hat`, so a first-layer node sits `dp/2` into the
fluid and a k-th-layer node ~`k dp`. Then a validity retract -- shorten (never
lengthen) the offset so the node clears every solid region: the far face of a
thin obstacle, the other wall of a re-entrant corner. A boundary particle with
no clean fluid-facing node on that ray (deep solid interior; the medial region
of a sharp convex solid edge; a concave fillet where `∇(sdf)` of the composed
`min`/`max` points into the wall) keeps a zero offset -- its MLS moment matrix
is singular and `computeMdbcDensity` Shepard-/rest-falls-back, the right answer
for a particle the fluid barely reaches.

Known gap (`DELTASPH_VALIDATION_PLAN.md` §5.2.2 / §5.2.3): on the Marrone 2011
§3.4 geometry the retract-along-`∇(sdf)` collapses ~1100/7600 ghost nodes at
the 45° edge / re-entrant corners / concave fillet, and the resulting weak wall
support lets a run-up jet leak. A literature-grounded per-boundary-layer normal
(Marrone 2011 App. A; DualSPHysics `JSphBoundCorr::ComputeNormals`) is the
right fix; δ⁺-SPH + PST sidesteps it by keeping fluid off the wall.
"""

import torch
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion
from typing import Any, Optional, Union

__all__ = ['addBoundaryGhostParticles']

#: The fluid-facing interface sits this many spacings `dp` proud of the
#: outermost boundary layer (English et al. 2022 §3: `dp/2`).
GHOST_FLUID_DEPTH = 0.5


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

            # English et al. 2022 reflection: mirror across this region's SDF
            # surface by twice the signed distance (capped so a deep layer does
            # not project absurdly far).
            sdfValues, sdfNormals = region.sdf(bpos)
            clampedDist = sdfValues - torch.clamp(-sdfValues, max=1.5 * hMean)
            legacyOffsets = clampedDist.view(-1, 1) * sdfNormals

            # Geometry validity pass: keep the reflection where it lands cleanly
            # in the fluid, retract it toward the boundary particle where it
            # would cross into another solid, collapse it (-> Shepard/rest
            # fallback) where no clean node exists. Fluid-independent.
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
            
