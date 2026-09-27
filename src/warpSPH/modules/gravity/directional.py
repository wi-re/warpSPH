"""Uniform directional gravity: returns a constant acceleration vector
(`schemeConfig.gravityConfig.direction * magnitude`) broadcast to every
particle; ignores particle positions entirely.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from ...utils.syncFree import deviceConstant
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

__all__ = ['computeDirectionalGravity']


def computeDirectionalGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    v = currentState.velocities
    direction = schemeConfig.gravityConfig.direction
    magnitude = schemeConfig.gravityConfig.magnitude
    if not isinstance(direction, torch.Tensor):
        # cached: a per-call torch.tensor(list) is a synchronous host->device copy
        direction = deviceConstant(direction, v.dtype, v.device)
    return (direction[:v.shape[1]] * magnitude).repeat(v.shape[0], 1)