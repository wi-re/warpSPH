"""Rows with nothing inside their kernel support.

Marrone-style detection classifies a row by the smallest eigenvalue of its
renormalization matrix. An empty support has no matrix to invert, falls back to
the identity and reads `lambda = 1` -- *bulk* -- so a fully isolated particle
(a splash droplet, an ejected flyer) is never flagged as free surface, although
its support is all air. `MDBC_CONTACT_LINE_PLAN.md` §7 found this on
sloshingTank: isolated rows carrying p = -320 that no surface-based treatment
could see.

Exact, no threshold: `sum_j V_j (x_j - x_i) . gradW_ij` (the trace of the
un-inverted renormalization matrix) is a sum of strictly negative terms, one
per neighbour inside the support, and exactly 0.0 when there is none.
"""

from typing import Any

import torch
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)

__all__ = ['detectIsolated', 'detectNoFluidNeighbours']


def detectIsolated(state: Any, config: Any, adjacency: Any) -> torch.Tensor:
    """Boolean per row: True when no other particle (fluid or boundary) lies
    inside the row's kernel support."""
    trace = warpOperation(
        state,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.AllToAll,
                            gradientMode=GradientScheme.Difference),
        queryValues=state.positions, domain=config.domain, adjacency=adjacency)
    return trace == 0.0


def detectNoFluidNeighbours(state: Any, config: Any, adjacency: Any) -> torch.Tensor:
    """Boolean per row: True for a fluid row with no *other* fluid particle
    (`kind == 0`) inside its kernel support; wall rows may be present.

    Exact integer count (`countNeighborsWarp`, FluidToFluid, which counts the
    row itself). Not `detectIsolated`'s trace restricted to fluid pairs by
    linearity: that needs two wall sums from different kernels to cancel,
    which float rounding does not do exactly, so a row with only wall
    neighbours never read as exactly zero."""
    from ..util.wp_numNeighbors import countNeighborsWarp
    n = countNeighborsWarp(
        state, OperationProperties(kernel=config.kernel, operation=WarpOperation.Gradient,
                                   supportMode=SupportScheme.SuperSymmetric,
                                   operationMode=OperationDirection.FluidToFluid,
                                   gradientMode=GradientScheme.Naive),
        config.domain, adjacency=adjacency)
    return (state.kinds == 0) & (n.to(torch.int64) <= 1)
