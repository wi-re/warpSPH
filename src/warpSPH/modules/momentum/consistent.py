"""Momentum equation source term with grad-h / gradient-renormalization consistency.

Computes `-rho * div(v)` (or `-rho/omega * div(v)` when a `GradHState` is
supplied) using a consistent (renormalized) difference-form divergence over
all-to-all, super-symmetric support -- the term used where schemes carrying
grad-h corrections need the momentum equation kept consistent with the
density estimate.
"""

from warpSPHCore import *
from ...systems.baseState import *
from warpSPH.configurations import SimulationConfig

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
from typing import Optional, Any
import torch

__all__ = ['computeMomentumConsistent_warp']

def computeMomentumConsistent_warp(
    state: BaseParticleState,
    config: SimulationConfig,
    schemeConfig: Any, 
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH - computeMomentumConsistent"):
        densities = state.densities 
        omega = gradH.queryOmegas if gradH is not None else None
        
        term = - densities / omega if gradH is not None else - densities

        sphInterp = warpOperation(
            state,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
                gradientMode = GradientScheme.Difference,
            ),
            queryValues = state.velocities,
            domain = config.domain,
            adjacency=adjacency,
            consistentDivergence = True,
        )
        return term * sphInterp