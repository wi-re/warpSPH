"""Momentum equation source term for the incompressible solver.

Computes `-rho0 * div(advectionVelocities)` using the scheme's fixed rest
density (rather than the current per-particle density, unlike DFSPH's
choice, per the inline comment) and an externally supplied advection
velocity field, via a plain (non-renormalized) scatter-support divergence.
"""

from warpSPHCore import *
from ...systems.baseState import *
from warpSPH.configurations import SimulationConfig
from typing import Any, Optional, Union
import torch

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
__all__ = ['computeMomentumIncompressible']


def computeMomentumIncompressible(
        currentState: Any, 
        config: SimulationConfig, 
        schemeConfig: Any, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        advectionVelocities: torch.Tensor        
):
        # drhodt can be computed either using the current densities or the rest density
        # the latter is the option chosen in dfsph

    rho = schemeConfig.fluid.restDensity
#     rho = currentState.densities

    return - rho * warpOperation(
                currentState,
                OperationProperties(
                        kernel = config.kernel,
                        operation = WarpOperation.Divergence,
                        gradientMode = GradientScheme.Difference,
                        supportMode = SupportScheme.Scatter,
                ),
                queryValues = advectionVelocities,
                domain = config.domain,
                adjacency=adjacency,
                consistentDivergence = False,
        )       