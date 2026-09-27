"""Surface normal for the Maronne detection scheme: an SPH (naive-gradient,
Scatter) estimate of the unit-field gradient, corrected by the
renormalization matrix `L`, then negated and normalized.
"""

# @torch.jit.script
# def computeNormalsMaronne(
#     particles : WeaklyCompressibleState,
#     L: torch.Tensor,
#     lambdas: torch.Tensor,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter):
    
#     with record_function("[SPH] - [Surface Detection] - Compute Normals (Maronne)"):
#         ones = particles.positions.new_ones(particles.positions.shape[0])
#         term = SPHOperationCompiled(
#             particles,
#             quantity = (ones, ones),
#             neighborhood= neighborhood[0],
#             kernelValues = neighborhood[1],
#             operation= Operation.Gradient,
#             gradientMode = GradientMode.Naive,
#             supportScheme= supportScheme
#         )
        
#         nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
#         n = -torch.nn.functional.normalize(nu, dim = -1)
#         lMin = torch.min(torch.abs(lambdas), dim = -1).values
    
#     return n, lMin



import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

__all__ = ['computeNormalsMaronne']

def computeNormalsMaronne(currentState: Any, L: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    ones = currentState.positions.new_ones(currentState.positions.shape[0])
    term = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Scatter,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Naive
        ),
        queryValues = ones,
        domain = config.domain,
        adjacency = adjacency
    )
    
    nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
    n = -torch.nn.functional.normalize(nu, dim = -1)
    # lMin = torch.min(torch.abs(lambdas), dim = -1).values
    return n