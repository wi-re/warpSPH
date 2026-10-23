"""Pressure interpolation at the closest wall contacts."""

import torch
from warpSPHCore import ParticleState, OperationProperties, OperationDirection, warpOperation


def computeIISPHContactPressure(state, config, normalizedPressure, restVolumes,
                              contactPoints, contactAdjacency, contactDenominator):
    """Return Shepard-interpolated q=p/rho0, or zero when contact pressure is disabled."""
    if contactPoints is None:
        return torch.zeros_like(normalizedPressure)
    query = ParticleState(positions=contactPoints, supports=state.supports,
        masses=state.masses, densities=state.densities, kinds=state.kinds)
    numerator = warpOperation(query,
        OperationProperties(kernel=config.kernel, operationMode=OperationDirection.FluidToFluid),
        config.domain, adjacency=contactAdjacency, referenceParticles=state,
        queryVolumes=restVolumes, referenceVolumes=restVolumes,
        referenceValues=normalizedPressure)
    active = contactDenominator > 0
    denominator = torch.where(active, contactDenominator, 1)
    return torch.where(active, numerator / denominator, 0)
