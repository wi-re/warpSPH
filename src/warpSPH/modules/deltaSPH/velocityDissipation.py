"""Public entry point for the delta-SPH momentum-dissipation term `dvdt_diss`:
a thin wrapper that runs `computeVelocityDiffusionDeltaSPH` as a Laplacian
operation over `currentState.velocities` and forwards the viscosity
selection straight from `schemeConfig.diffusionParams` (`inviscid`, its
artificial-viscosity `inviscidAlpha`, `fixedSoundSpeed`, and the physical
kinematic viscosity `viscidNu`) — see `wp_viscosityDelta.py` for which of the
two forms is actually applied.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
from ...enumTypes import *

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

__all__ = ['computeVelocityDiffusion']

def computeVelocityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], approachOnly: bool = True) -> torch.Tensor:
    """`approachOnly=False` lifts the approaching-neighbours clamp, turning the
    `inviscid=False` branch into the Monaghan & Gingold (1983) velocity
    Laplacian proper -- see `wp_viscosityDelta.py`'s docstring."""
    with record_function("[warpSPH] - (deltaSPH) - computeVelocityDiffusion"):
        dvdt_diss = computeVelocityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Laplacian,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryVelocities = currentState.velocities,
            inviscid = schemeConfig.diffusionParams.inviscid,
            c_s = schemeConfig.fluid.fixedSoundSpeed,
            alpha = schemeConfig.diffusionParams.inviscidAlpha,
            nu = schemeConfig.diffusionParams.viscidNu,
            approachOnly = approachOnly,
        )
        return dvdt_diss