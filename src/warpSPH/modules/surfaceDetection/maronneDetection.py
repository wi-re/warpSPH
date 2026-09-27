"""Maronne-style free-surface detection: builds a trial point offset along
each particle's normal and classifies neighbors into two geometric conditions
(A: far and inside a normal-aligned tube; B: near and inside a tangential
band) via `computeMaronneSurfaceDetection`; a particle with neither condition
satisfied is flagged as surface. Note the leading comment block is a stale,
unrelated (`ColorFieldGradient`) commented-out predecessor, left in place.
"""

# @torch.jit.script
# def detectFreeSurfaceColorFieldGradient(
#     particles : WeaklyCompressibleState,
#     colorField : torch.Tensor,
#     colorGrad: torch.Tensor,
#     xi :float,
#     colorFieldGradientThreshold : float,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
#     ):
#     with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Color Field Gradient)"):

#         fs = torch.linalg.norm(colorGrad, dim = -1) > colorFieldGradientThreshold * particles.supports / xi
#         return fs
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

from .maronneNormals import computeNormalsMaronne
from .wp_maronne import computeMaronneSurfaceDetection

__all__ = ['detectFreeSurfaceMaronne']



def detectFreeSurfaceMaronne(
        currentState: Any, 
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 

        renormalizationState: Optional[RenormalizationState] = None, 
        eigenValues: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None, 
        
        returnNormals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if renormalizationState is None:
        C, Evals, renormalizationState = computeRenormalizationMatrices(
            queryParticles = currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Gradient,
                operationMode = OperationDirection.AllToAll,
                supportMode = SupportScheme.SuperSymmetric
            ),
            domain = config.domain,
            adjacency = adjacency,
            returnEigVals = True
        )
    if normals is None:
        normals_maronne = computeNormalsMaronne(
            currentState,
            renormalizationState.renormalizationMatrices,
            config, schemeConfig, schemeConfig.surfaceDetectionConfig,
            adjacency,
        )
    else:
        normals_maronne = normals
    fsm = computeMaronneSurfaceDetection(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            operationMode = OperationDirection.AllToAll,
            supportMode = SupportScheme.SuperSymmetric
        ),
        domain = config.domain,
        adjacency = adjacency,
        surfaceNormals = normals_maronne,
    )

    return fsm, normals_maronne if returnNormals else fsm