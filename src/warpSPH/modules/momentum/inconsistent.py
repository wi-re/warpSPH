"""Momentum equation source term, plain (non-renormalized) SPH divergence.

Computes `-rho * div(v)` using the standard "inconsistent" (not
gradient-renormalized) SPH divergence estimator, all-to-all with
super-symmetric support.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeMomentum']



from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *


def computeMomentum(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], wall: Optional[Any] = None) -> torch.Tensor:
    """`wall`: the analytic boundary's `WallState` (`modules/analyticBoundary`) adds the free-slip mirror of the wall to the divergence (the continuity
    equation's wall flux); `None`: resolved from `schemeConfig.boundaryProvider` (`resolveWall`), no wall term for boundary particles."""
    with record_function("[warpSPH] - computeMomentum"):
        drhodt = -currentState.densities * warpOperation(
            currentState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
                gradientMode = GradientScheme.Difference
            ),
            queryValues = currentState.velocities,
            domain = config.domain,
            adjacency = adjacency,
            consistentDivergence = False
        )
        from ..analyticBoundary import resolveWall
        wall = resolveWall(currentState, config, schemeConfig, adjacency, wall)
        if wall is not None:
            from ..analyticBoundary import wallContinuity
            drhodt = drhodt + wallContinuity(wall, currentState.densities, currentState.velocities)
        return drhodt
