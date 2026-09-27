"""Free-surface flag from the norm of the color-field gradient: a particle is
surface when `|grad(colorField)| > colorFieldGradThreshold * supports / xi`,
where `xi` is the kernel's characteristic support ratio
(`sphKernel_xi`). Recomputes `computeColorField` internally when not passed in.
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


from .colorFieldCompute import computeColorField

__all__ = ['detectFreeSurfaceColorFieldGradient']



def detectFreeSurfaceColorFieldGradient(
        currentState: Any, 
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        
        
        colorField: Optional[torch.Tensor] = None, colorFieldGradient: Optional[torch.Tensor] = None, 
        
        returnNormals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if colorField is None or colorFieldGradient is None:
        colorField, colorFieldGradient = computeColorField(
            currentState,
            config = config,
            schemeConfig = schemeConfig,
            adjacency = adjacency
        )

    xi = sphKernel_xi(config.kernel.value, config.dim)
    fs = torch.linalg.norm(colorFieldGradient, dim = -1) > surfaceConfig.colorFieldGradThreshold * currentState.supports / xi
    return fs if not returnNormals else (fs, -torch.nn.functional.normalize(colorFieldGradient, dim=-1))
