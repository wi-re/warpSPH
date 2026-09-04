"""SPH particle density estimation.

Computes particle densities via plain SPH kernel summation with gather
support -- the base density estimator most schemes build on before layering
grad-h or gradient-renormalization corrections. Gather (not scatter) support
is used because it matches the Cullen-Dehnen switch's E.1 density estimate
(per the CRK paper), per the inline comment.
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

__all__ = ['computeDensities']


def computeDensities(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeDensities"):
        return warpOperation(
            currentState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
                # Lattice-normalisation correction (LATTICE_DENSITY_PLAN.md).
                # Off unless the config asks for it. The summation density is
                # the operator the offset is defined on, so this is where it is
                # wired first; the other operators still default to off.
                n_h = getattr(config, 'n_h', None),
                calibrateNormalization = getattr(config, 'calibrateNormalization', False),
            ),
            domain = config.domain,
            adjacency = adjacency,
    )