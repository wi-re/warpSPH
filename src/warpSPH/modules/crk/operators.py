"""Frontend interface of the CRK operators, in the shape of the other modules: `(state, config, schemeConfig, adjacency, ...)`.

`computeCrkSPHAccelWarp` / `computeCrkSPHdudtWarp` are the low-level launchers (an `OperationProperties`, a domain, every field as its own
tensor). The CRKSPH step calls these wrappers instead: the kernel, the diffusion parameters and the kernel-mean pair kernel the CRK
formulation is defined for (Frontiere et al. 2017, Eq. 8) are read or fixed here, and the CRK data a stage computes once (apparent volumes, the
`CRKState` of A, B, grad A, grad B, the velocity gradient, the resolved limiter parameters) are passed as a `CRKStage`.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from warpSPHCore import CRKState, GradHState, OperationProperties, SupportScheme

from .accel import computeCrkSPHAccelWarp
from .dudt import computeCrkSPHdudtWarp

__all__ = ['CRKStage', 'computeCRKAccel', 'computeCRKdudt']


@dataclass
class CRKStage:
    """What one CRK stage computes once and every CRK operator of the stage reads."""
    volumes: torch.Tensor              # apparent volumes
    crkState: CRKState                 # A, B, grad A, grad B
    velocityGradient: torch.Tensor     # (N, dim, dim), `J[i] @ dx` = velocity change along dx
    viscosityParams: Any               # the limiter constants resolved from n_h (`resolveCRKLimiter`)


def _common(state, config, schemeConfig, adjacency, stage: CRKStage, gradH: Optional[GradHState]):
    return dict(
        queryParticles = state,
        operationProperties = OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric),
        domain = config.domain,
        conductivityParams = schemeConfig.diffusionParams,
        crkViscosityParams = stage.viscosityParams,
        queryVelocityTensor = stage.velocityGradient,
        queryEnergies = state.internalEnergies,
        queryVelocities = state.velocities,
        queryCs = state.soundspeeds,
        queryAlphas = state.alphas,
        queryPressures = state.pressures,
        queryVolumes = stage.volumes,
        crkState = stage.crkState,
        adjacency = adjacency,
        gradHState = gradH,
    )


def computeCRKAccel(state, config, schemeConfig, adjacency, stage: CRKStage, gradH: Optional[GradHState] = None):
    """Pressure + CRK pseudo-viscosity acceleration; returns `(dvdt, ap_ij, av_ij)` (the pair terms the energy balance partitions)."""
    return computeCrkSPHAccelWarp(**_common(state, config, schemeConfig, adjacency, stage, gradH))


def computeCRKdudt(state, config, schemeConfig, adjacency, stage: CRKStage, gradH: Optional[GradHState] = None):
    """The matching specific internal-energy rate."""
    return computeCrkSPHdudtWarp(**_common(state, config, schemeConfig, adjacency, stage, gradH))
