"""IISPH-style pressure acceleration: the symmetric SPH pressure gradient of
`pressureValues` (typically the IISPH pressure or pressure-increment field),
divided by density and negated to give an acceleration. Used by the
incompressible (IISPH / divergence-free) solvers in `modules/incompressible/`.
"""

from warpSPHCore import *
from ...systems.compressibleMonaghan import *
from warpSPH.configurations import SimulationConfig

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
from typing import Optional
import torch

__all__ = ['computePressureAccelIISPH']

def computePressureAccelIISPH(
    state: CompressibleState,
    pressureValues: torch.Tensor,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None):
    return -warpOperation(
        state,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            gradientMode = GradientScheme.Symmetric,
            supportMode = supportScheme,
        ),
        queryValues = pressureValues,
        domain = config.domain,
        adjacency=adjacency,
    ) / state.densities[:, None]
