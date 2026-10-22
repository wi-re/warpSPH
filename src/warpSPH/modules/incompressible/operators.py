"""Projection geometry and pressure operators, independent of solver scheduling.

`scatterDivergence` is positive for expansion. The analytic wall pressure
closure can be affine/nonlinear (hydrostatic offset and clamps); callers
must not assume this whole pressure acceleration is a linear map.
"""

from typing import Any
import torch
from warpSPH.configurations.compute import neighborSearchOptions
from warpSPHCore import (
    GradientScheme,
    OperationProperties,
    SupportScheme,
    WarpOperation,
    buildVerletList,
    warpOperation,
)
from ..pressure.iisph import computePressureAccelIISPH


def rebuildAdjacency(state: Any, system: Any, config: Any):
    adjacency = buildVerletList(
        state,
        config.domain,
        verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency,
        verbose=False,
        **neighborSearchOptions(config)
    )
    system.adjacency = adjacency
    return adjacency


def scatterDivergence(
    state: Any,
    config: Any,
    adjacency: Any,
    field: torch.Tensor,
    wall: Any = None,
    bodyVelocity: bool = False,
) -> torch.Tensor:
    """Scatter / difference-form SPH divergence -- the operator
    `computeMomentumIncompressible` wraps (`WarpOperation.Divergence`,
    `GradientScheme.Difference`, `SupportScheme.Scatter`), without its
    `-rho0` factor. `sum_j (m_j/rho_j) (field_j - field_i) . gradW_ij`; for a
    `div = +1` field it returns `+1` in the bulk. omniSPH's per-particle sum
    `sum_j V_j (X_i - X_j) . gradW` is the negative of this.

    `wall` (analytic walls, `modules/analyticBoundary`): the wall's share
    `sum_b (f_b - f_i) . G_b`, `f_b` the body velocity with `bodyVelocity` (a
    velocity field), else 0 (the pressure acceleration: a static wall takes no
    reaction, as the boundary particles' zeroed rows)."""
    div = warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel,
            operation=WarpOperation.Divergence,
            gradientMode=GradientScheme.Difference,
            supportMode=SupportScheme.Scatter,
        ),
        queryValues=field,
        domain=config.domain,
        adjacency=adjacency,
        consistentDivergence=False,
    )
    if wall is not None:
        from ...modules.analyticBoundary import wallDivergence

        div = div + wallDivergence(wall, field, bodyVelocity=bodyVelocity)
    return div


def pressureAcceleration(
    state: Any,
    config: Any,
    adjacency: Any,
    p: torch.Tensor,
    fluidMask: torch.Tensor,
    wall: Any = None,
    rho0: float = 1.0,
    mlsFit: Any = None,
    clampPressure: bool = True,
) -> torch.Tensor:
    """omniSPH `computeAcceleration`: the symmetric SPH pressure gradient
    `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) gradW`, as an acceleration. Zeroed
    on non-fluid rows -- a static particle takes no reaction
    (`operatorMovesBoundary=False`), and ghosts are not real particles."""
    a_p = computePressureAccelIISPH(
        state=state,
        pressureValues=p,
        config=config,
        supportScheme=SupportScheme.Scatter,
        adjacency=adjacency,
    )
    if wall is not None:
        # analytic walls: the mirrored pressure `p_b = p_i^+` plus the hydrostatic wall pressure, clamped at 0, in omniSPH's own symmetric form (its triangle boundary term, the boundaries
        # repo's `DFSPH2D._boundary_accel`, equal to ~1e-6); `rho0` is the rest density of the mirror and of the hydrostatic offset
        from ...modules.analyticBoundary import wallPressureAccelerationOmni

        # `mlsFit` (analyticWallPressure = 'mls'): the wall pressure extrapolated linearly from the fluid's current pressure iterate instead of the hydrostatic one
        a_p = a_p + wallPressureAccelerationOmni(
            wall,
            p,
            state.densities,
            rho0,
            wallMass=wall.wm,
            h=wall.support,
            gradient=None if mlsFit is None else mlsFit.gradient(p),
            clampPressure=clampPressure,
        )
    return torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
