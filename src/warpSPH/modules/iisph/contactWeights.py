"""Shepard interpolation weights at the closest wall contact of each query."""

import torch
from warpSPHCore import (buildVerletList, SupportScheme, ParticleState, OperationProperties,
                         OperationDirection, warpOperation)
from ...configurations.compute import neighborSearchOptions


def computeIISPHContactWeights(state, config, points, wallGradient, restVolumes):
    # Contact-centered support differs from fluid-centered support.
    query = ParticleState(positions=points, supports=state.supports,
        masses=state.masses, densities=state.densities, kinds=state.kinds)
    adjacency = buildVerletList(query, config.domain, supportMode=SupportScheme.Gather,
        referenceParticles=state, **neighborSearchOptions(config))
    denominator = warpOperation(query,
        OperationProperties(kernel=config.kernel, operationMode=OperationDirection.FluidToFluid),
        config.domain, adjacency=adjacency, referenceParticles=state,
        queryVolumes=restVolumes, referenceVolumes=restVolumes,
        referenceValues=torch.ones_like(restVolumes))
    active = (state.kinds == 0) & (wallGradient.square().sum(-1) > 1.e-24)
    denominator = torch.where(active, denominator, 0)
    missing = active & (denominator <= 0)
    return adjacency, denominator, missing.any()
