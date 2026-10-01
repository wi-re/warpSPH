"""compSPH multistep internal-energy update.

Combines the per-Butcher-stage pairwise pressure/viscosity accelerations and
energy-partition factors (`f_ij`, from `balance.py`) recorded across a
Runge-Kutta step into a single internal-energy increment: builds a half-step
velocity from the weighted stage accelerations, then scatter-sums each pair's
partitioned thermal work back to particles. `EnergyScheme.PdV` uses a
different weighting (half viscosity, unweighted pressure) than the other
schemes.
"""

import torch

from ...systems.compressibleMonaghan import CompressibleSystemUpdate
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...enumTypes import EnergyScheme
from warpSPHCore import *

from typing import Any, List, Optional
import numpy as np
from ...math.scatter import segment_sum


from typing import Tuple

__all__ = ['compSPH_deltaU_multistep']

def compSPH_deltaU_multistep(
        dt : float,
        initialState: Any, #CompSPHState,
        returnValues: Any, # List[Tuple[AdjacencyList, CompSPHState]],
        updates : List[CompressibleSystemUpdate],
        butcherTerms: np.ndarray,
        generalConfig: SimulationConfig,
        solverConfig: CompSPHConfig,
        verbose: bool = False
):
    # verbosePrint(verbose, '[DeltaU] Computing Accelerations')
    fullAcceleration = torch.zeros_like(initialState.velocities)
    for i, update in enumerate(updates):
        # print(f'[DeltaU]\tStep {i:2d}/{len(updates):2d}\tk = {butcherTerms[i]:.2f}')
        # print(f'dudt shape: {update.dudt.shape}')
        fullAcceleration += butcherTerms[i] * update.dvdt
    halfStepVelocity = initialState.velocities + 0.5 * fullAcceleration * dt

    deltaU = torch.zeros_like(initialState.internalEnergies)
    # verbosePrint(verbose, '[DeltaU] Computing DeltaU')
    for ii, (stepAdjacency, stepState) in enumerate(returnValues):
        # verbosePrint(verbose, f'[DeltaU]\tStep {ii:2d}/{len(particleStates):2d}\tk = {butcherTerms[ii]:.2f}')
        i = stepAdjacency.i
        j = stepAdjacency.j

        v_i = halfStepVelocity[i]
        v_j = halfStepVelocity[j]
        v_ji = v_j - v_i

        f_ij = stepState.f_ij
        ap_ij = stepState.ap_ij
        av_ij = stepState.av_ij
        
        k = butcherTerms[ii]
        if solverConfig.energyScheme == EnergyScheme.PdV:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij)
            term += k * 1/2 * torch.einsum('ij, ij -> i', v_ji, av_ij)
        else:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij + av_ij)
            
        # row sums in a fixed order (scatter_sum's atomics made CRKSPH non-reproducible)
        deltaU += segment_sum(term, stepAdjacency.numNeighbors)
    # verbosePrint(verbose, '[DeltaU] Done')
    return deltaU

