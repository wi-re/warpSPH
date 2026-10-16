"""De-correlates a regular initial particle lattice by jittering fluid particles and relaxing them with repeated delta-SPH shifting steps.

Positions are first perturbed by Gaussian noise scaled by each particle's own
support (``jitterAmount`` fraction), then pushed through ``shiftIters``
rounds of ``computeDeltaShift`` (each shift scaled by a fixed factor of 10)
to spread out the resulting clumping/voids before the density field is used.
Only fluid particles (``kinds == 0``) are moved; boundary/ghost particles are
excluded from both the jitter and the shift update. Density is restored to
its pre-call value on return, and the original positions are restored on
``state`` -- the shuffled positions are returned as a fresh tensor rather
than mutating ``state`` in place. An earlier version of this function is left
commented out above the active one.
"""

from warpSPH.configurations.compute import neighborSearchOptions

import torch
from warpSPHCore import *

from ..shifting.delta import computeDeltaShift

__all__ = ['shuffleParticles']

# def shuffleParticles(state, config, schemeConfig, shiftIters, jitterAmount = 0.01):
#     priorPositions = state.positions.clone()

#     state.positions[state.kinds == 0] += torch.randn_like(state.positions)[state.kinds == 0] * state.supports[state.kinds == 0].view(-1,1) * jitterAmount

#     adjacency = None
#     for i in range(shiftIters):
#         adjacency = buildVerletList(
#             state, 
#             config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
#             priorNeighborhood = adjacency,
#             verbose = False)
        
#         shiftVector, adjacency = computeDeltaShift(
#             currentState = state,
#             config = config,
#             schemeConfig = schemeConfig,
#             domain = config.domain,
#             adjacency = adjacency,
#         )
#         state.positions[state.kinds == 0] += shiftVector[state.kinds == 0]
    
#     newPositions = state.positions.clone()
#     state.positions = priorPositions
#     return newPositions


def shuffleParticles(state, config, schemeConfig, shiftIters, jitterAmount = 0.01):
    priorPositions = state.positions.clone()

    state.positions[state.kinds == 0] += torch.randn_like(state.positions)[state.kinds == 0] * state.supports[state.kinds == 0].view(-1,1) * jitterAmount

    initialDensities = state.densities.clone()

    adjacency = None
    for i in range(shiftIters):
        adjacency = buildVerletList(
            state,
            config.domain,
            verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=adjacency,
            verbose=False,
            **neighborSearchOptions(config))
        state.densities = warpOperation(
            state,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = config.kernel, 
                supportMode = SupportScheme.Gather
            ),
            domain = config.domain,
            adjacency = None
        )
        
        shiftVector, adjacency = computeDeltaShift(
            currentState = state,
            config = config,
            schemeConfig = schemeConfig,
            domain = config.domain,
            adjacency = adjacency,
        )
        state.positions[state.kinds == 0] += 10. * shiftVector[state.kinds == 0]
        # print(f'Iteration {i+1}/{shiftIters}: max shift = {shiftVector[state.kinds == 0].norm(dim=1).max().cpu().item():.6g}')
    
    newPositions = state.positions.clone()
    state.positions = priorPositions
    state.densities = initialDensities
    return newPositions