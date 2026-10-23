"""Diagonal of IISPH continuity composed with the self/fluid pressure force."""

import torch
from .operators import launchIISPHKernel, pairArguments


def computeIISPHPressureDiagonal(state, pairs, restVolumes, densityRatio, gradientSum):
    """Return -(c_i.c_i + V_i sum_fluid V_j |grad W_ij|²) / s_i².

    Solid rows are zero. The result excludes contact-pressure extrapolation
    and the timestep factor; the solver adds those explicitly.
    """
    diagonal = torch.empty_like(restVolumes)
    launchIISPHKernel('pressure_diagonal', state, pairArguments(pairs) +
        [restVolumes, state.kinds.to(torch.int32), densityRatio, gradientSum, diagonal])
    return diagonal
