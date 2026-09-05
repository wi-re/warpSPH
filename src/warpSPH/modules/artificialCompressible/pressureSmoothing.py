"""`D^p`, the ACSPH pressure-smoothing operator (De Courcy et al. 2024
Eqs. 32-37; `ACSPH_PLAN.md` Secs. 1.4 and 4.3).

Returns the **unscaled** operator -- the `k2 = 0.1 h beta` prefactor of Eq. (24)
belongs to the caller (`schemes/artificialCompressible.py`), which is the only
place `beta` exists.

Nothing here is new numerics. AC-2 and AC-2L are the delta-SPH density-diffusion
operator with the pressure in place of the density, which
`computeScalarFieldDiffusion` supports directly:

    AC-2  (Eq. 32)      == DensityDiffusionScheme.densityOnly
    AC-2L (Eqs. 33-34)  == DensityDiffusionScheme.deltaSPH, with <grad p>^L

See Part 3 of the plan for why our *unprojected* psi reproduces the paper's
*projected* Eq. (33) exactly (`gradW || x_ij` for any isotropic kernel;
measured to 2e-15 by `scripts/probe_deltaSPHPsiProjection.py`) -- and for the
one condition under which it does not: gradient renormalisation on `gradW`
itself. `computeScalarFieldDiffusion` builds its own `OperationProperties` with
no `renormalizationState`, so that path is off by construction here; `L` enters
only through `<grad p>^L`, which is where Eq. (34) wants it.
"""

from typing import Any, Optional, Union

import torch
from torch.profiler import record_function
from warpSPHCore import AdjacencyList, CompactHashMap

from ...configurations.simulationConfig import SimulationConfig
from ...enumTypes import DensityDiffusionScheme, PressureSmoothingScheme
from ..deltaSPH import computeScalarFieldDiffusion
from ..density.gradRhoL import computeGradRhoL

__all__ = ['computePressureSmoothing']


def computePressureSmoothing(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Any = None,
    pressures: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`D^p_i` for the configured `acParams.pressureSmoothing`, unscaled.

    `pressures` defaults to `currentState.pressures`; pass it explicitly when
    the dual-time loop is evaluating at a stage value rather than the state's
    own. `renormalizationState` is the `L` from `detectFreeSurface` -- reused
    rather than recomputed, since AC-2L needs exactly the matrix Eq. (34) does.
    """
    scheme = schemeConfig.acParams.pressureSmoothing
    p = currentState.pressures if pressures is None else pressures

    with record_function("[warpSPH] - (acsph) - computePressureSmoothing"):
        if scheme is PressureSmoothingScheme.laplacian:
            return computeScalarFieldDiffusion(
                currentState, config, adjacency,
                DensityDiffusionScheme.densityOnly, field=p)

        if scheme is PressureSmoothingScheme.renormalizedBiLaplacian:
            gradPL = computeGradRhoL(currentState, config, schemeConfig, adjacency,
                                     renormalizationState, field=p)
            return computeScalarFieldDiffusion(
                currentState, config, adjacency,
                DensityDiffusionScheme.deltaSPH, gradFieldL=gradPL, field=p)

        raise NotImplementedError(
            f"pressureSmoothing={scheme.name} is not implemented yet -- AC-4 "
            f"(Eq. 35) and AC-JST (Eqs. 36-37) land with ACSPH_PLAN.md step 8. "
            f"Use PressureSmoothingScheme.renormalizedBiLaplacian (AC-2L, the "
            f"paper's default) or .laplacian (AC-2, its negative control).")
