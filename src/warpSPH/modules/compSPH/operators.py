"""Frontend interface of the compSPH operators, in the shape of the other modules.

`computeCompSPHAccelWarp`, `computeCompSPHdudtWarp` and `computeCompSPHBalanceTermWarp`
are the low-level launchers: they take an `OperationProperties`, a domain and every
field as a separate tensor (which is what the gradcheck scripts need). The scheme steps
want the same call shape as `computeMomentumConsistent` / `computeDensities`:
`(state, config, schemeConfig, adjacency, gradH)`. These wrappers read the fields from
the particle state, take the kernel / diffusion parameters / energy scheme from the
configs, and fix the support mode each operator is defined for (E.2, E.3 in the CRKSPH
paper), so a scheme cannot pick the wrong one by accident.
"""

from typing import Optional, Tuple, Union

import torch

from warpSPHCore import GradHState, OperationProperties, SupportScheme

from .accel import computeCompSPHAccelWarp
from .balance import computeCompSPHBalanceTermWarp
from .dudt import computeCompSPHdudtWarp

__all__ = ['computeCompSPHAccel', 'computeCompSPHdudt', 'computeCompSPHBalanceTerm']


def computeCompSPHAccel(
    state,
    config,
    schemeConfig,
    adjacency,
    gradH: Optional[GradHState] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pressure + artificial-viscosity acceleration (super-symmetric, E.2).

    Returns `(dvdt, ap_ij, av_ij)`: the acceleration and the per-pair pressure and
    viscosity contributions that `computeCompSPHBalanceTerm` partitions the energy with.
    """
    return computeCompSPHAccelWarp(
        queryParticles = state,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        conductivityParams = schemeConfig.diffusionParams,

        queryEnergies = state.internalEnergies,
        queryVelocities = state.velocities,
        queryCs = state.soundspeeds,
        queryAlphas = state.alphas,
        queryPressures = state.pressures,

        adjacency = adjacency,
        gradHState = gradH,
    )


def computeCompSPHdudt(
    state,
    config,
    schemeConfig,
    adjacency,
    gradH: Optional[GradHState] = None,
) -> torch.Tensor:
    """Specific internal-energy rate from pressure work and viscous dissipation (gather, E.3)."""
    return computeCompSPHdudtWarp(
        queryParticles = state,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.Gather,
        ),
        domain = config.domain,
        conductivityParams = schemeConfig.diffusionParams,

        queryEnergies = state.internalEnergies,
        queryVelocities = state.velocities,
        queryCs = state.soundspeeds,
        queryAlphas = state.alphas,
        queryPressures = state.pressures,

        adjacency = adjacency,
        gradHState = gradH,
    )


def computeCompSPHBalanceTerm(
    state,
    config,
    schemeConfig,
    adjacency,
    velocities: torch.Tensor,
    dt: Union[float, torch.Tensor],
    gradH: Optional[GradHState] = None,
) -> torch.Tensor:
    """Energy-partition fraction `f_ij` of every pair, evaluated at `velocities` (the half-step velocity).

    Reads the per-pair `ap_ij` / `av_ij` that `computeCompSPHAccel` left on the state.
    """
    return computeCompSPHBalanceTermWarp(
        queryParticles = state,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = config.supportMode,
        ),
        domain = config.domain,

        queryEnergies = state.internalEnergies,
        queryVelocities = velocities,
        queryPressures = state.pressures,

        pairWise_pressureAccel = state.ap_ij,
        pairWise_viscosityAccel = state.av_ij,
        energyScheme = schemeConfig.energyScheme,
        dt = dt,
        gamma = schemeConfig.gamma,

        adjacency = adjacency,
        gradHState = gradH,
    )
