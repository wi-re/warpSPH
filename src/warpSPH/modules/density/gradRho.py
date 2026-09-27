"""SPH density gradient (unrenormalized).

Computes the naive (non-renormalized) SPH gradient of density via an
all-to-all, super-symmetric difference-form gradient operation. See
`gradRhoL.py` for the gradient-renormalization-corrected variant.
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

__all__ = ['computeGradRho']


def computeGradRho(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeGradRho"):
        return warpOperation(
                currentState,
                OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    supportMode = SupportScheme.SuperSymmetric,
                    operationMode = OperationDirection.AllToAll,
                    gradientMode = GradientScheme.Naive
                ),
                queryValues = currentState.densities,
                # queryValues = testQuantity,
                domain = config.domain,
                adjacency = adjacency,
                # renormalizationState = RenormalizationState(C=C, eigVals=Evals, L=L)
            )