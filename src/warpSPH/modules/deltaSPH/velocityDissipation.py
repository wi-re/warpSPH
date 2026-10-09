"""Public entry point for the delta-SPH momentum-dissipation term `dvdt_diss`:
a thin wrapper that runs `computeVelocityDiffusionDeltaSPH` as a Laplacian
operation over `currentState.velocities` and forwards the viscosity
selection straight from `schemeConfig.diffusionParams` (`inviscid`, its
artificial-viscosity `inviscidAlpha`, `fixedSoundSpeed`, and the physical
kinematic viscosity `viscidNu`) — see `wp_viscosityDelta.py` for which of the
two forms is actually applied.
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
from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
from ...enumTypes import *

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

__all__ = ['computeVelocityDiffusion']

def computeVelocityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], approachOnly: bool = True, wall: Optional[Any] = None) -> torch.Tensor:
    """`wall`: the analytic boundary's `WallState` (`modules/analyticBoundary`) adds the wall's share of the diffusion (exact wall Laplacian with the same prefactor
    as the pair term); `None`: resolved from `schemeConfig.boundaryProvider` (`resolveWall`), no wall term for boundary particles.

    `approachOnly=False` lifts the approaching-neighbours clamp, turning the
    `inviscid=False` branch into the Monaghan & Gingold (1983) velocity
    Laplacian proper -- see `wp_viscosityDelta.py`'s docstring."""
    with record_function("[warpSPH] - (deltaSPH) - computeVelocityDiffusion"):
        dvdt_diss = computeVelocityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Laplacian,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryVelocities = currentState.velocities,
            inviscid = schemeConfig.diffusionParams.inviscid,
            c_s = schemeConfig.fluid.fixedSoundSpeed,
            alpha = schemeConfig.diffusionParams.inviscidAlpha,
            nu = schemeConfig.diffusionParams.viscidNu,
            approachOnly = approachOnly,
            # opt-in shear-carrying Morris 1997 term (`ViscosityTerm`), only
            # on the physical-viscosity branch; the default leaves every
            # scheme on the projected form it had
            morris = (getattr(schemeConfig.diffusionParams, 'viscousTerm', ViscosityTerm.monaghanGingold)
                      == ViscosityTerm.morris1997),
        )
        from ..analyticBoundary import resolveWall
        wall = resolveWall(currentState, config, schemeConfig, adjacency, wall)
        if wall is not None:
            from ..analyticBoundary import viscousPrefactor, wallViscousAcceleration
            dvdt_diss = dvdt_diss + wallViscousAcceleration(wall, currentState.densities, currentState.velocities, viscousPrefactor(schemeConfig, config, wall.support), wall.support,
                                                            wallMass=wall.wm, kernel=config.kernel)
        return dvdt_diss
