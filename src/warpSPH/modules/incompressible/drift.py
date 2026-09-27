"""IISPH pressure-induced position drift `dx_p = -sum_j (f_j - f_i)` (a
divergence of the per-particle pressure accelerations), used as the
predictor term in the divergence-free/incompressible Jacobi iterations to
turn a trial pressure field into a displacement residual.
"""

from warpSPHCore import *

from warpSPH.systems.compressibleMonaghan import CompressibleState
from ...systems.baseState import *
from warpSPH.configurations import SimulationConfig
from typing import Any, Optional, Union
import torch


from torch.profiler import profile, ProfilerActivity


from warpSPHCore.profiling import record_function
__all__ = ['computePressureShiftIISPH']



def computePressureShiftIISPH(
    state: CompressibleState,
    config: SimulationConfig,
    pressureAccels: torch.Tensor,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,):
    kernelSum = warpOperation(
        state,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Divergence,
            gradientMode = GradientScheme.Difference,
            supportMode = SupportScheme.Scatter,
        ),
        queryValues = pressureAccels,
        domain = config.domain,
        adjacency=adjacency,
        consistentDivergence = False, # Computes fj - fi, (9) in PBSPH states fi -fj thus the minus sign
    ) #* state.densities
    return - kernelSum
