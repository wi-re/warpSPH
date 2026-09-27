"""Symmetric SPH pressure-force acceleration: the negative symmetric
pressure gradient (`P_i/rho_i^2 + P_j/rho_j^2` form, via
`GradientScheme.Symmetric`) divided by density. Exposed by the package as
`computePressureForceSymmetric`.
"""

from warpSPHCore import *
from ...systems.compressibleMonaghan import *
from warpSPH.configurations import SimulationConfig

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
from typing import Optional
import torch

__all__ = ['pressureForce_warp']

def pressureForce_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH - pressureForceSymmetric"):
        return -warpOperation(
            state,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Gradient,
                gradientMode = GradientScheme.Symmetric,
                supportMode = supportScheme,
            ),
            queryValues = state.pressures,
            domain = config.domain,
            adjacency=adjacency,
            gradHState = gradH
        ) / state.densities[:, None]
