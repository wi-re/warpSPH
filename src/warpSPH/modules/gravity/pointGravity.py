"""Point-source gravity: acceleration of constant magnitude
(`schemeConfig.gravityConfig.magnitude`) directed from each particle's
(periodic-image-aware) position toward a fixed `origin`; not distance-scaled
(no inverse-square falloff), with a `1e-8` epsilon guarding the
normalization near the origin.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.math import getPeriodicPositions
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

__all__ = ['computePointGravity']


def computePointGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    v = currentState.velocities
    origin = schemeConfig.gravityConfig.origin
    magnitude = schemeConfig.gravityConfig.magnitude
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin, dtype = v.dtype, device = v.device)

    x = currentState.positions
    minD = config.domain.min
    maxD = config.domain.max
    periodic = config.domain.periodic

    periodicPositions = getPeriodicPositions(x, config.domain)
    direction = origin - periodicPositions
    directionNorm = torch.norm(direction, dim=1, keepdim=True)
    directionNormalized = direction / (directionNorm + 1e-8)  # Avoid division by zero
    return (directionNormalized * magnitude).repeat(v.shape[0], 1)