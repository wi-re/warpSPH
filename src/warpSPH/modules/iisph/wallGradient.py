"""Solid contribution to the IISPH kernel-gradient sum."""

import torch
from .operators import launchIISPHKernel, pairArguments


def computeIISPHWallGradient(state, pairs, restVolumes, *, boundaryIntegrals=None, massFactor=1.0):
    """Return G_i = sum_solid V_j grad W_ij, or its analytic integral.

    Analytic integrals are scaled from geometric volume to calibrated rest
    volume. Sampled walls use the supplied particle rest volumes directly.
    """
    if boundaryIntegrals is not None:
        return massFactor * boundaryIntegrals.out['G'].sum(0).to(state.positions)
    wallGradient = torch.empty_like(state.positions)
    launchIISPHKernel('wall_gradient', state, pairArguments(pairs) +
        [restVolumes, state.kinds.to(torch.int32), wallGradient])
    return wallGradient
