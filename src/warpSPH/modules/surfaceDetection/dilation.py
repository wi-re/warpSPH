"""Expands a free-surface mask outward by repeatedly summing the mask value
over kernel-supported neighbors (`dilateSurfaceMaskWarp`),
`surfaceConfig.expansionIterations` times unless `overrideIterations` is given.
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

from .wp_dilate import dilateSurfaceMaskWarp

__all__ = ['dilateSurface']


def dilateSurface(currentState: Any, freeSurfaceMask: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], overrideIterations: Optional[int] = None) -> torch.Tensor:

    out = freeSurfaceMask.clone()
    if overrideIterations is not None:
        iterations = overrideIterations
    else:
        iterations = surfaceConfig.expansionIterations
    for i in range(iterations):
        out = dilateSurfaceMaskWarp(
            currentState,
            OperationProperties(
                kernel = config.kernel,
                # operation = WarpOperation.DilateSurfaceMask,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
                gradientMode = GradientScheme.Naive
            ),
            freeSurfaceMask = out,
            domain = config.domain,
            adjacency = adjacency
        )
    return out