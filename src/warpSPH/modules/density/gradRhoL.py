"""Gradient-renormalized SPH scalar gradient.

Computes a scalar field's gradient corrected by a gradient-renormalization
matrix `L` (derived from the local particle-distribution covariance via
`computeRenormalizationMatrices`, computed on demand if not supplied) --
used by delta-SPH's density diffusion term, where an uncorrected gradient
estimate is too noisy near boundaries/free surfaces.

The field defaults to `currentState.densities`, which is the delta-SPH case and
the reason for the name. Pass `field=` for anything else: with the pressure it
is De Courcy et al. 2024's Eq. (34) renormalized pressure gradient
`<grad p>^L_i = -sum_j (p_i - p_j) L_i grad_i W_ij V_j` verbatim -- the
`GradientScheme.Difference` mode is exactly that difference form, and `L` here
is exactly their `L_i` (ACSPH_PLAN.md Sec. 4.3).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *

__all__ = ['computeGradRhoL']


def computeCovariance(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeCovariance"):
        C, Evals, L = computeRenormalizationMatrices(
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
        return C, Evals, L

def computeGradRhoL(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], L: Optional[RenormalizationState], field: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`field` defaults to the state's density; pass a scalar tensor to
    renormalize any other field's gradient (see the module docstring)."""
    with record_function("[warpSPH] - (deltaSPH) - computeGradRhoL"):
        if L is None:
            C, Evals, L = computeCovariance(currentState, config, schemeConfig, adjacency)

        return warpOperation(
                currentState,
                OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    supportMode = SupportScheme.SuperSymmetric,
                    operationMode = OperationDirection.AllToAll,
                    gradientMode = GradientScheme.Difference
                ),
                queryValues = currentState.densities if field is None else field,
                # queryValues = testQuantity,
                domain = config.domain,
                adjacency = adjacency,
                renormalizationState = L
            )