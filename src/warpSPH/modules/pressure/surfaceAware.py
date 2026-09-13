"""Case/scheme-facing wrapper around `wp_surfaceAware.computePressureSurfaceAwareWarp`:
assembles the operation properties (`SupportScheme.SuperSymmetric`) and
forwards the current pressures, the configured pressure-force formulation
(`schemeConfig.pressureForceTerm`) and the surface-indicator mask consumed
by the Antuono surface-aware term.
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
from .wp_surfaceAware import computePressureSurfaceAwareWarp

__all__ = ['computePressureForceSurfaceAware']

def computePressureForceSurfaceAware(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], renormalizationState: Optional[Any] = None) -> torch.Tensor:
    with record_function("[warpSPH] - computePressureForceSurfaceAware"):
        dvdt = computePressureSurfaceAwareWarp(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                supportMode = SupportScheme.SuperSymmetric,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryPressures = currentState.pressures,
            pressureTerm = schemeConfig.pressureForceTerm,
            querySurfaceMask = currentState.surfaceIndicators,
            # `renormalizationState` (`None` unless the caller opts in via
            # `schemeConfig.pressureForceRenormalized`) makes
            # `wp_surfaceAware.py`'s existing but previously-unused
            # `useGradientRenormalization`/`Li` path apply the same L matrix
            # `deltaSPH.py` already computes for `gradRhoL` to this kernel
            # gradient too. See `configurations/weaklyCompressible.py`'s
            # `pressureForceRenormalized` docstring.
            renormalizationState = renormalizationState,
        )
        return dvdt