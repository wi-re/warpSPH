"""The Riemann states' inputs of a Godunov step: the resolved reconstruction limiter, the reconstruction order, and (second order) the velocity
Jacobian and the density / pressure gradients, shared by the GSPH and Inutsuka steps."""

from typing import Optional, Tuple

import torch

from warpSPHCore import SupportScheme

from ...configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, resolveReconstructionLimiter
from ..reconstruction import computeStateGradients, computeVelocityJacobian

__all__ = ['godunovReconstruction']


def godunovReconstruction(state, config, schemeConfig, adjacency, *, alwaysStateGradients: bool = False
                          ) -> Tuple[object, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """`(params, order, J, G)`: first order (`Raw` pair velocity) uses the particles' own values and needs no gradients. Second order needs the
    velocity Jacobian `J` and, for the reconstructed rho and P, their gradients `G` (only when `params.riemannReconstruction`, unless
    `alwaysStateGradients`: the Inutsuka form always uses them)."""
    params = resolveReconstructionLimiter(schemeConfig.diffusionParams, config.n_h)
    order = 1 if params.velocityPairPolicy == VelocityPairPolicy.Raw.value else 2
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(state, config, SupportScheme.SuperSymmetric, adjacency,
                                    corrected = params.correctReconstructionGradient)
        if alwaysStateGradients or params.riemannReconstruction:
            G = computeStateGradients(state, config, SupportScheme.SuperSymmetric, adjacency,
                                      corrected = params.correctReconstructionGradient)
    return params, order, J, G
