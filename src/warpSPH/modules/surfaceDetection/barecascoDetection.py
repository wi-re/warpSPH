"""Barecasco-style free-surface detection: a particle is flagged as surface
(`fsm < 0.5`) when its kernel-weighted neighbor directions leave a wide
enough angular gap uncovered, as computed by
`computeBarecascoSurfaceDetectionWarp`; the accompanying cover-vector direction
doubles as the surface normal.
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
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from .maronneNormals import computeNormalsMaronne
from .wp_barecasco import computeBarecascoSurfaceDetectionWarp

__all__ = ['detectFreeSurfaceBarecasco']




def detectFreeSurfaceBarecasco(
        currentState: Any, 
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        
        returnNormals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    normals, fsm = computeBarecascoSurfaceDetectionWarp(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            operationMode = OperationDirection.AllToAll,
            supportMode = SupportScheme.SuperSymmetric
        ),
        domain = config.domain,
        adjacency = adjacency,
        barecascoThreshold = surfaceConfig.barecascoThreshold
    )

    fs = fsm < 0.5
    return fs if not returnNormals else (fs, normals)