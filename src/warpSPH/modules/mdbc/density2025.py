"""mDBC boundary-particle density extrapolation (English et al. 2022).

Each boundary particle (kind 1) has a ghost node (kind 2) placed in the fluid.
`interpolateLiuLiu` fits a local linear fluid-density field (value + gradient)
at the node; English Eq. (12), `rho_b = rho_g + (r_b - r_g) . grad(rho)_g`,
extrapolates it back to the boundary particle. Where the node's stencil is too
thin for a trustworthy 1st-order fit -- few neighbours, or a near-coplanar
moment matrix (`|det(A_g)|` below `interpolateLiuLiu`'s per-kernel floor) -- the
result blends smoothly down to the plain 0th-order Shepard value at the node,
and to rest density where there is no fluid at all.

The blend is a ramp, not a hard `where(wellConditioned, ...)` switch: the switch
put a step in the boundary density right at the free-surface contact line (a
large moving region in a violent flow), where roughly a third of the wetted
boundary can land on the fallback during a dam-break run-up
(`DELTASPH_VALIDATION_PLAN.md` 5.2.3). The Shepard fallback is used raw --
**not** clamped to >= rho0 (that clamp is a DualSPHysics DBC anti-attraction
guard, not in English 2022) and with **no** added hydrostatic-gravity term.
One deviation from the cited paper, the ghost-normal normalization, is no longer
relevant since the gravity term is gone.

No-ops (returns `currentState.densities` unchanged) when there are no boundary
particles.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeMdbcDensity']



from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

from ..liu import interpolateLiuLiu, determinantThresholdFor
from ._util import stateHasBoundaryParticles

#: 1st-order (English Eq. 12) -> 0th-order (Shepard) blend ramps. Below the
#: `numNeighbors` floor / the `|det(A_g)|` floor the boundary particle gets the
#: pure Shepard value; the MLS fit reaches full weight `_*_RAMP` past each floor.
#: A smooth crossover instead of a hard `where(wellConditioned, ...)` switch, so
#: a particle near the free-surface contact line (or under a thin climbing
#: sheet), where the stencil is marginal, does not snap between the exact MLS
#: value and a crude estimate -- the jump that made mDBC boundary densities look
#: discontinuous from the fluid (DELTASPH_VALIDATION_PLAN 5.2.3).
_MDBC_NBR_FLOOR = 4.0
_MDBC_NBR_RAMP = 1.0
_MDBC_DET_RAMP = 0.25

def computeMdbcDensity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.densities
    with record_function("[warpSPH] - (mdbc) - computeMdbcDensity"):
        rho_interp, rho_interp_grad, numNeighbors, A_g, b, wellConditioned = interpolateLiuLiu(
            currentState.positions[currentState.kinds == 2],
            referenceParticles = currentState,
            referenceQuantities = currentState.densities,
            config = config,
            neighbor_threshold = 4,
            direction = OperationDirection.FluidToGhost,
            supportScale = 1.0,
            adjacency = adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None
        )
        # return res[:,0], res[:,1:], neighCounts

        ghostMask = currentState.kinds == 2
        bIndices = currentState.ghostIndices[ghostMask]

        rho0 = schemeConfig.fluid.restDensity

        # -- 0th order: the Shepard value at each ghost node, Sum rho_j W_gj /
        # Sum W_gj (= b[0] / A_g[0,0]). Always defined once the node has a fluid
        # neighbour; this is the graceful floor, NOT clamped to >= rho0 (English
        # et al. 2022 has no such clamp -- it is a DualSPHysics DBC anti-attraction
        # guard that pins the near-surface wall to rest density even when the
        # adjacent fluid is genuinely lighter) and with no hydrostatic-gravity
        # term added (that term is a separate m2dbc assumption; layered on top of
        # a Shepard density near a churning surface it just adds error).
        shepardDenominator = A_g[:, 0, 0]
        shepardDensity = torch.where(
            (shepardDenominator > 0) & (numNeighbors > 1),
            b[:, 0] / torch.where(shepardDenominator > 0, shepardDenominator,
                                  torch.ones_like(shepardDenominator)),
            torch.full_like(shepardDenominator, rho0))

        # -- 1st order: English et al. 2022 Eq. (12), rho_b = rho_g +
        # (r_b - r_g) . grad(rho)_g, with the MLS value + gradient from
        # `interpolateLiuLiu`.
        relPos = -currentState.ghostOffsets[ghostMask]
        drho = -torch.einsum('nu, nu -> n', relPos, rho_interp_grad)
        rho_proj = rho_interp + drho

        # -- Blend 1st -> 0th order by a smooth conditioning weight. `w` is
        # exactly 0 below the `|det(A_g)|` floor `interpolateLiuLiu` gates on
        # (pinv can blow `rho_proj` up there) and below the neighbour floor;
        # it reaches 1 a few multiples past each. Replaces the hard
        # `where(wellConditioned, MLS, fallback)` switch -- the source of the
        # step at the free-surface contact line. The determinant floor still
        # matters (a thin, near-coplanar stencil under a fast dam-break front
        # over a dry bed makes `grad(rho)_g` explode -- Marrone 3.1, Part 3);
        # ramping through it rather than switching keeps that protection while
        # removing the discontinuity.
        det = torch.linalg.det(A_g).abs()
        detFloor = determinantThresholdFor(config.kernel)
        wDet = torch.clamp((det - detFloor) / (detFloor * _MDBC_DET_RAMP), 0.0, 1.0)
        wN = torch.clamp((numNeighbors.to(det.dtype) - _MDBC_NBR_FLOOR) / _MDBC_NBR_RAMP,
                         0.0, 1.0)
        w = wDet * wN

        rho_b = w * rho_proj + (1.0 - w) * shepardDensity
        rho_b = torch.nan_to_num(rho_b, nan=rho0, posinf=rho0, neginf=rho0)

        mergedDensitities = currentState.densities.clone()
        mergedDensitities[bIndices] = rho_b
        return mergedDensitities