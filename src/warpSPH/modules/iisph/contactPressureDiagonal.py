"""Diagonal contribution from interpolated wall-contact pressure."""

import torch
from .operators import launchIISPHKernel, pairArguments, geometryParameters


def computeIISPHContactPressureDiagonal(state, config, pairs, restVolumes,
                                      wallGradient, gradientSum, contactPoints, contactWeights):
    """Return diag(C F_contact), without modifying the base diagonal.

    Includes the local wall response and neighboring wall responses to each
    fluid pressure unknown. No timestep factor is applied here.
    """
    diagonal = torch.empty_like(restVolumes)
    launchIISPHKernel('contact_diagonal', state, pairArguments(pairs) +
        [restVolumes, state.kinds.to(torch.int32), state.positions,
         contactPoints.contiguous(), state.supports, wallGradient, gradientSum,
         contactWeights.denominator, *geometryParameters(state, config), diagonal])
    return diagonal
