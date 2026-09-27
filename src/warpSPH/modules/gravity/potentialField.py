"""Harmonic potential-field gravity: acceleration `-magnitude**2 * (x -
origin)` (a linear restoring force toward `origin`, i.e. a quadratic
potential well), computed via a normalize-then-rescale-by-distance detour
with a `1e-7` epsilon guarding points at the origin; periodic images of
position are used via `getPeriodicPositions`.
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

__all__ = ['computePotentialFieldGravity']


def computePotentialFieldGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    x = currentState.positions
    minD = config.domain.min
    maxD = config.domain.max
    periodic = config.domain.periodic

    periodicPositions = getPeriodicPositions(x, config.domain)

    origin = schemeConfig.gravityConfig.origin
    magnitude = schemeConfig.gravityConfig.magnitude
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin, dtype = x.dtype, device = x.device)
    
    xij = periodicPositions - origin
    rij = torch.linalg.norm(xij, dim = -1)
    xij[rij > 1e-7] = xij[rij > 1e-7] / rij[rij > 1e-7, None]

    return - magnitude**2 * xij * (rij)[:,None]