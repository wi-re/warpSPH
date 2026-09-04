"""Public entry point for the delta-SPH density-diffusion term `drhodt_diss`:
scales `computeDensityDiffusionDeltaSPH`'s raw (unscaled) divergence output
by `delta * h / xi * c_s`, the standard delta-SPH prefactor, where `delta` is
`schemeConfig.diffusionParams.densityDelta`, `h` is the per-particle support,
`xi = sphKernel_xi(kernel, dim)` is the kernel's normalization constant, and
`c_s` is `schemeConfig.fluid.fixedSoundSpeed`. Which density-diffusion
formulation is evaluated (delta-SPH, non-renormalized, density-only, ...) is
selected via `schemeConfig.diffusionParams.densityDiffusionTerm`
(`DensityDiffusionScheme`) and forwarded to the underlying kernel; the caller
(`schemes/deltaSPH.py`) is responsible for only supplying `gradRho`/`gradRhoL`
when the selected scheme actually needs them.

`computeScalarFieldDiffusion` is the same operator with neither the field nor
the prefactor fixed: it returns the raw (unscaled) divergence for any scalar
field, which is what ACSPH needs (ACSPH_PLAN.md Sec. 4.3 -- its pressure
smoothing operators carry a `k2 = 0.1 h beta` prefactor, not delta-SPH's
`delta h c_s`, and `beta` is a pseudo-time wave speed with no `c_s` behind it).
`computeDensityDiffusion` is the delta-SPH specialisation of it.
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

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

__all__ = ['computeDensityDiffusion', 'computeScalarFieldDiffusion']

def computeScalarFieldDiffusion(currentState: Any, config: SimulationConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], scheme: DensityDiffusionScheme, gradField: Optional[torch.Tensor] = None, gradFieldL: Optional[torch.Tensor] = None, field: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The raw (unscaled) delta-SPH diffusion divergence for an arbitrary scalar
    `field` and its gradients. `field=None` diffuses the state's density, i.e.
    reproduces `computeDensityDiffusion` without its prefactor. No
    `schemeConfig`: nothing here is scheme-specific, which is the point -- see
    the module docstring."""
    with record_function("[warpSPH] - (deltaSPH) - computeScalarFieldDiffusion"):
        return computeDensityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryGradRho = gradField,
            queryGradRhoL = gradFieldL,
            queryField = field,
            densityScheme = scheme
        )


def computeDensityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], gradRho: Optional[torch.Tensor], gradRhoL: Optional[torch.Tensor]) -> torch.Tensor:
    with record_function("[warpSPH] - (deltaSPH) - computeDensityDiffusion"):
        delta = schemeConfig.diffusionParams.densityDelta
        xi = sphKernel_xi(config.kernel.value, config.dim)
        drhodt_scaling = delta * currentState.supports / xi * schemeConfig.fluid.fixedSoundSpeed
        drhodt_diss = drhodt_scaling * computeScalarFieldDiffusion(
            currentState, config, adjacency,
            schemeConfig.diffusionParams.densityDiffusionTerm,
            gradField = gradRho, gradFieldL = gradRhoL,
        )
        return drhodt_diss