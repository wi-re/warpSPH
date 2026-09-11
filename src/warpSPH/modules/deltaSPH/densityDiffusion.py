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

def computeScalarFieldDiffusion(currentState: Any, config: SimulationConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], scheme: DensityDiffusionScheme, gradField: Optional[torch.Tensor] = None, gradFieldL: Optional[torch.Tensor] = None, field: Optional[torch.Tensor] = None, operationMode: OperationDirection = OperationDirection.AllToAll) -> torch.Tensor:
    """The raw (unscaled) delta-SPH diffusion divergence for an arbitrary scalar
    `field` and its gradients. `field=None` diffuses the state's density, i.e.
    reproduces `computeDensityDiffusion` without its prefactor. No
    `schemeConfig`: nothing here is scheme-specific, which is the point -- see
    the module docstring.

    `operationMode` defaults to `AllToAll` (unchanged for every caller besides
    `computeDensityDiffusion`, e.g. ACSPH's pressure smoothing)."""
    with record_function("[warpSPH] - (deltaSPH) - computeScalarFieldDiffusion"):
        return computeDensityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = operationMode,
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
        # Fluid-to-fluid only: a boundary neighbour's density is the mDBC
        # extrapolation, a phase-lagged copy of the fluid's own field, not an
        # independent measurement. Diffusing a fluid particle toward it is a
        # delayed self-coupling that can sustain/pump near-wall oscillations
        # instead of draining them (DELTASPH_VALIDATION_PLAN.md item C).
        # DualSPHysics's DDT (all variants: DDT_DDT / DDT_DDT2 / DDT_DDT2Full,
        # JSphCpu.cpp ~L925-939) excludes boundary neighbours from this same
        # sum entirely. Only this outer pair-sum direction changes here --
        # `gradRho`/`gradRhoL` (passed in, computed elsewhere) are untouched.
        drhodt_diss = drhodt_scaling * computeScalarFieldDiffusion(
            currentState, config, adjacency,
            schemeConfig.diffusionParams.densityDiffusionTerm,
            gradField = gradRho, gradFieldL = gradRhoL,
            operationMode = OperationDirection.FluidToFluid,
        )
        return drhodt_diss