"""Total volume-weighted kernel gradient for the self-pressure force."""

import torch
from .operators import launchIISPHKernel, pairArguments


def computeIISPHGradientSum(state, pairs, restVolumes, wallGradient):
    """Return c_i = sum_fluid V_j grad W_ij + G_i without modifying G."""
    gradientSum = torch.empty_like(state.positions)
    launchIISPHKernel('gradient_sum', state, pairArguments(pairs) +
        [restVolumes, state.kinds.to(torch.int32), wallGradient, gradientSum])
    return gradientSum
