"""Internal-energy time derivative du/dt = -(p/rho) * div(v), Monaghan-style,
via an SPH divergence of velocity. Divides by the grad-h `queryOmegas`
correction when `gradH` is supplied.
"""

from warpSPHCore import *
from ...systems.compressibleMonaghan import *
from warpSPH.configurations import SimulationConfig

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
from typing import Optional
import torch

__all__ = ['compute_dudt_warp']

def compute_dudt_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH[compute_dudt]"):
        term = - state.pressures / state.densities
        if gradH is not None:
            term /= gradH.queryOmegas

        sphInterp = warpOperation(
            state,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                gradientMode = GradientScheme.Difference,
                supportMode = supportScheme,
            ),
            queryValues = state.velocities,
            domain = config.domain,
            adjacency=adjacency,
            consistentDivergence = True,
        )
        return term * sphInterp
