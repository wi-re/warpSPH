"""Viscous reference velocities for sampled walls without stored ghost rows."""

import torch
from warpSPHCore import (ParticleState, OperationProperties, OperationDirection,
    SupportScheme, WarpOperation, buildVerletList, warpOperation)
from ...configurations.compute import neighborSearchOptions
from ...configurations.region import BCType
from ...systems.identity import particleBodyIDs
from ..boundaryConditions.mirrorVelocity import mirrorVelocity


def computeViscosityBoundaryVelocities(state, config, schemeConfig):
    """Sample reflected fluid points and return a separate reference-velocity field.

    Existing mDBC ghost rows already supply boundary velocities through
    computeBoundaryVelocities. Without them, construct the same mirrored
    sampling geometry from the contour or the stored boundary-to-ghost offset.
    Stored particle velocities continue to describe the rigid body's motion.
    """
    if not bool((state.kinds == 1).any()) or bool((state.kinds == 2).any()):
        return state.velocities
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    bodies = getattr(schemeConfig, 'rigidBodies', [])
    if not bodies:
        return state.velocities
    if provider is not None:
        from ...boundary.provider import bindBodies
        bindBodies(provider, bodies)
    bodyIDs = particleBodyIDs(state)
    reference = state.velocities.clone()
    for body in bodies:
        if body.kind not in (BCType.freeSlip, BCType.noSlip):
            continue
        indices = torch.nonzero((state.kinds == 1) & (bodyIDs == body.bodyID)).flatten()
        if len(indices) == 0:
            continue
        positions = state.positions[indices]
        if provider is not None:
            distance, normal, hit = provider.signed_distance(positions, body=body.bodyID)
            if not bool(hit.all()):
                raise ValueError('Viscous mirror sampling requires a covered solid contour')
            normal = normal.to(positions)
            points = positions - 2 * distance.to(positions)[:, None] * normal
        else:
            offsets = state.ghostOffsets[indices]
            length = offsets.norm(dim=-1, keepdim=True)
            if bool((length == 0).any()):
                raise ValueError('Viscous mirror sampling requires boundary-to-ghost offsets or a contour')
            normal = offsets / length
            points = positions - offsets
        query = ParticleState(positions=points.contiguous(), supports=state.supports[indices],
            masses=state.masses[indices], densities=state.densities[indices],
            kinds=torch.zeros_like(state.kinds[indices]))
        adjacency = buildVerletList(query, config.domain, supportMode=SupportScheme.Gather,
            referenceParticles=state, **neighborSearchOptions(config))
        properties = OperationProperties(kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.Gather, operationMode=OperationDirection.FluidToFluid)
        velocity = warpOperation(query, properties, domain=config.domain, adjacency=adjacency,
            referenceParticles=state, queryValues=torch.zeros_like(points), referenceValues=state.velocities)
        weight = warpOperation(query, properties, domain=config.domain, adjacency=adjacency,
            referenceParticles=state, queryValues=torch.zeros_like(query.masses),
            referenceValues=torch.ones_like(state.masses))
        supported = weight > 0
        wallVelocity = state.velocities[indices]
        velocity = velocity / torch.where(supported, weight, 1)[:, None]
        velocity = torch.where(supported[:, None], velocity, wallVelocity)
        reference[indices] = (mirrorVelocity(velocity, wallVelocity, normal)
            if body.kind == BCType.freeSlip else 2 * wallVelocity - velocity)
    return reference
