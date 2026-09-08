"""Builds the mirrored boundary-ghost-particle layer that mDBC needs: for each
`RegionType.Boundary` region, projects its fluid-kind particles across the
fluid-facing interface into the fluid, appends the ghost particles to every
per-particle array, and records each boundary particle's ghost index/offset
(and the reverse, on the new ghost particle) on the returned state. Assumes one
region per boundary material ID, assigned in region order starting at 0
(`boundaryMaterial`); called once during initialization, from
`initializers/weaklyCompressible.py`.

**Ghost placement** (`DELTASPH_VALIDATION_PLAN.md` §5.2.1). Every *wetted*
boundary particle's ghost is the particle mirrored across the **fluid-facing
interface**, with the interface distance and the mirror direction taken from
the surrounding *fluid* particles' actual positions -- not from the boundary
region's SDF gradient.

Why not the SDF gradient: `regions/sample.py` cuts a regular lattice with
`sdf < 0`, so a boundary particle's perpendicular distance to the true surface
is only `dp/2` for the outer layer of an axis-aligned, grid-registered face.
The old code reflected each particle by `2·s` along `∇(sdf)`, and `∇(sdf)`
points to the nearest *solid* surface -- which past a thick wall band's medial
axis, or in a re-entrant corner, is the wrong surface and the wrong direction.
Measured on the English 2022 §4.1 wedge: a 4th/5th-layer boundary particle
0.1 below the bed beside the wedge got a ghost mirrored *straight up* by
`10·dp` into open fluid, and English Eq. (12)'s `ρ_b = ρ_g + (r_b−r_g)·∇ρ_g`
then extrapolated the corner-contaminated fluid gradient back over that
`10·dp` lever arm -- a ~13 % hydrostatic-pressure error at the re-entrant
corner (`DELTASPH_VALIDATION_PLAN.md` §5.2.1).

The fluid-direction mirror fixes both: the direction is toward the actual
fluid (diagonally, at a corner), and the mirror distance is `2·(d_fluid −
dp/2)` where `d_fluid` is the distance to the nearest fluid particle -- so a
corner particle whose nearest fluid is `2·dp` away gets a `~3·dp` offset, not
`10·dp`. On a grid-aligned flat wall `d_fluid` for the k-th layer is exactly
`k·dp`, `2·(k·dp − dp/2)` reproduces the old `2·|s|` mirror for **every**
layer, and the direction is the outward normal, so no grid-aligned case moves.

Boundary particles whose nearest fluid is beyond `_GHOST_NEAR_FLUID·dp` (deep
solid interior) keep the legacy SDF reflection and Shepard-fallback to rest
density in `interpolateLiuLiu` -- which is the right answer for a particle the
fluid never reaches.
"""

import torch
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion
from typing import Any, Optional, Union

__all__ = ['addBoundaryGhostParticles']

#: The fluid-facing interface sits this many spacings `dp` proud of the
#: outermost boundary layer (English et al. 2022 §3: `dp/2`). Used only as the
#: floor on the mirror distance when fluid is touching the boundary particle.
GHOST_FLUID_DEPTH = 0.5

#: Fluid-neighbour gather radius for the direction / interface estimate, as a
#: multiple of the mean support radius.
_GHOST_GATHER_SCALE = 2.5

#: Minimum fluid neighbours before the fluid-direction mirror is trusted;
#: below this the particle keeps the legacy SDF reflection (it will
#: Shepard-fallback in `interpolateLiuLiu` anyway).
_GHOST_MIN_FLUID_NBRS = 3

#: A boundary particle is "wetted" -- and so gets the fluid-direction mirror --
#: only if its nearest fluid particle is within this many spacings. A particle
#: deeper in the solid does not exchange with the fluid and keeps rest density.
#: ~ the kernel reach (2h = 4 dp at h/dp = 2), plus a margin for a re-entrant
#: corner where the nearest fluid is diagonally offset.
_GHOST_NEAR_FLUID = 6.0

#: Cap on the ghost offset `|r_b - r_g|`, in spacings `dp`, applied *only* to
#: boundary particles the misalignment gate below flags as being at a corner /
#: curved region. English Eq. (12) is a *local* linear extrapolation of the
#: fluid density gradient, error ~ `offset^2 * curvature(rho)`; capping the
#: lever arm there keeps the extrapolation in the clean near-wall fluid. On a
#: flat wall `curvature(rho) = 0` and the per-layer mirror (up to `2*band*dp`)
#: is harmless -- so those particles are left exactly as the legacy path had
#: them (`DELTASPH_VALIDATION_PLAN.md` §5.2.1; capping them too regressed the
#: coarse flat tank).
_GHOST_MAX_OFFSET = 2.0

#: The fluid-direction placement replaces the legacy SDF reflection only where
#: the two materially disagree: `dot(nHat_fluid, nHat_legacy) < this`. On a
#: flat, grid-aligned wall they are collinear (~1.0) and every boundary
#: particle keeps its legacy offset byte-for-byte; at a re-entrant corner the
#: fluid direction bends away from the SDF normal and the gate opens. 0.985 ~
#: 10 degrees.
_GHOST_MISALIGN_COS = 0.985


def _fluidDirectedGhostOffsets(bpos, fluidPos, dx, hMean, legacyOffsets, *,
                               chunk: int = 4096):
    """Place every *wetted* boundary particle's ghost by mirroring it across the
    fluid-facing interface, with the interface distance and direction taken from
    the nearby fluid particles' positions. Returns `bpos - ghostPos` per
    particle.

    For the k-th layer of a grid-aligned flat wall the nearest fluid sits at
    `k·dp`, the mirror distance `2·(k·dp − dp/2)` equals the old `2·|s|`
    reflection, and the direction is the outward normal -- so a grid-aligned
    wall is byte-identical to the legacy path at every layer. At a re-entrant
    corner the direction bends toward the actual (diagonal) fluid and the
    distance shrinks to the real fluid gap, killing the `10·dp` straight-up
    lever arm the legacy SDF reflection produced there.

    A boundary particle whose nearest fluid is beyond `_GHOST_NEAR_FLUID·dp`,
    or that has too few fluid neighbours for a reliable direction, keeps
    `legacyOffsets` and Shepard-falls-back to rest density downstream.
    """
    if fluidPos.shape[0] == 0:
        return legacyOffsets.clone()

    R = _GHOST_GATHER_SCALE * hMean
    d_floor = GHOST_FLUID_DEPTH * dx
    n = bpos.shape[0]
    offsets = legacyOffsets.clone()

    maxOff = _GHOST_MAX_OFFSET * dx

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        pc = bpos[lo:hi]                                    # (c, D)
        legc = legacyOffsets[lo:hi]                         # (c, D)

        rel = fluidPos.unsqueeze(0) - pc.unsqueeze(1)       # (c, F, D)
        dist2 = (rel * rel).sum(-1)                         # (c, F)
        dNearest = dist2.min(1).values.clamp_min(0.0).sqrt()
        wetted = dNearest <= _GHOST_NEAR_FLUID * dx         # (c,) -- fluid can reach it

        within = dist2 < R * R
        cnt = within.sum(1)                                 # (c,)

        # Direction to the local fluid: distance-weighted mean of the offsets
        # to fluid particles in range. At a re-entrant corner this is the
        # diagonal into the corner opening; on a flat wall it is the outward
        # normal.
        w = torch.clamp(1.0 - torch.sqrt(dist2.clamp_min(0.0)) / R, min=0.0) * within
        meanRel = (w.unsqueeze(-1) * rel).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        dirNorm = torch.linalg.norm(meanRel, dim=-1, keepdim=True)
        nHat = meanRel / dirNorm.clamp_min(1e-12)           # (c, D)
        goodDir = (cnt >= _GHOST_MIN_FLUID_NBRS) & (dirNorm.squeeze(-1) > 1e-9)

        # Distance to the nearest fluid particle *along* nHat (the fluid gap).
        proj = (rel * nHat.unsqueeze(1)).sum(-1)            # (c, F)
        projIn = torch.where(within & (proj > 0.0), proj,
                             torch.full_like(proj, float('inf')))
        dFluid = projIn.min(1).values
        dFluid = torch.where(torch.isfinite(dFluid), dFluid,
                             dNearest.clamp_max(_GHOST_NEAR_FLUID * dx))

        # Mirror across the interface (dp/2 inside the fluid gap), capped to one
        # fluid layer so the Eq. (12) extrapolation stays local -- see
        # `_GHOST_MAX_OFFSET`.
        mirror = torch.clamp(2.0 * (dFluid - 0.5 * dx), min=d_floor, max=maxOff)
        fluidDirOff = -mirror.unsqueeze(-1) * nHat          # = pc - ghost

        # The gate: does the fluid direction disagree with the legacy SDF
        # reflection? `-legc` points from the boundary particle toward its
        # legacy ghost. On a flat grid-aligned wall this is collinear with
        # `nHat`, the gate stays shut, and the particle keeps its legacy offset
        # exactly. At a corner the two diverge and the fluid-direction placement
        # takes over.
        legDir = torch.nn.functional.normalize(-legc, dim=-1)
        aligned = (legDir * nHat).sum(-1) >= _GHOST_MISALIGN_COS

        useFluidDir = (wetted & goodDir & ~aligned)
        offsets[lo:hi] = torch.where(useFluidDir.unsqueeze(-1), fluidDirOff, legc)

    return offsets


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

    fluidPos = particleState.positions[particleState.kinds == 0]

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

            # Legacy path, kept as the fallback: reflect across the boundary
            # SDF surface by twice the signed distance.
            sdfValues, sdfNormals = region.sdf(bpos)
            clampedDist = sdfValues - torch.clamp(-sdfValues, max=1.5 * hMean)
            legacyOffsets = clampedDist.view(-1, 1) * sdfNormals

            # Primary path: place the ghost `GHOST_FLUID_DEPTH * dx` into the
            # fluid, past the fluid-facing interface, along the direction to the
            # local fluid -- see the module docstring.
            offsets = _fluidDirectedGhostOffsets(bpos, fluidPos, dx, hMean, legacyOffsets)

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
            
