"""mDBC boundary-particle density extrapolation.

Computes densities for ghost/boundary particles (kind 2) by Liu-Liu
(moving-least-squares) extrapolation of fluid density to each ghost point
(see the linked ScienceDirect paper in-code), then converts that to a
hydrostatic pressure correction along the ghost-offset normal, including a
gravity term, clamped to at least rest density. One deviation from the cited
paper's formula is called out inline (the ghost-normal normalization) as
matching DualSPHysics rather than the paper. Falls back to a plain
Shepard-interpolated density, then to rest density, when a ghost point has
too few (<=1) or too few (<=9) fluid neighbors respectively. No-ops (returns
`currentState.densities` unchanged) when there are no boundary particles.
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

from ..liu import interpolateLiuLiu
from ._util import stateHasBoundaryParticles

def computeMdbcDensity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.densities
    with record_function("[warpSPH] - (mdbc) - computeMdbcDensity"):
        rho_interp, rho_interp_grad, numNeighbors, A_g, b = interpolateLiuLiu(
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
        c_s = schemeConfig.fluid.fixedSoundSpeed
        g = torch.tensor(schemeConfig.gravityConfig.direction, dtype = currentState.positions.dtype, device = currentState.positions.device) * schemeConfig.gravityConfig.magnitude if schemeConfig.gravityConfig.active else torch.zeros_like(currentState.positions[0])


        boundaryDensity = currentState.densities.new_ones(currentState.densities.shape) * schemeConfig.fluid.restDensity
        shepardNominator = b[:,0]
        shepardDenominator = A_g[:,0,0]
        shepardDensity = torch.where(shepardDenominator > 0, shepardNominator / shepardDenominator, rho0)

        boundaryDensity[bIndices] = torch.where(numNeighbors > 1, shepardDensity, rho0)

        rho_g = boundaryDensity[bIndices]
        P_g = c_s**2 * (rho_g - rho0)


        relPos = -currentState.ghostOffsets[ghostMask]
        nb = relPos
        # This normalization is not in the paper https://www.sciencedirect.com/science/article/pii/S0045793025003305?via%3Dihub
        # But it is correct, see the dualsphysics code and thanks Aaron!
        nb = torch.nn.functional.normalize(nb, dim = 1)

        dot = torch.einsum('ni, i -> n', nb, g)
        dot2 = torch.einsum('ni, ni -> n', relPos, nb)

        # if the normal is pointing in the direction of the gravity then we disable the gravity contribution as we want to avoid negative densities in the boundary particles
        # dot = torch.where(dot > 0, torch.zeros_like(dot), dot)

        P_b = P_g + rho0 * (dot * dot2)
        rho_b = rho0 + P_b / c_s**2
        rho_b = torch.clamp(rho_b, min = rho0)

        boundaryDensity[bIndices] = rho_b

        # Bulk path: English et al. 2022 Eq. (12) linear extrapolation from the
        # ghost node back to the boundary particle,
        #   rho_b = rho_g + (r_b - r_g) . grad(rho)_g ,
        # with the MLS value + gradient from `interpolateLiuLiu`. The
        # Shepard-density + hydrostatic `rho_b` above is DualSPHysics'
        # m2dbc-style fallback, used below this neighbour count.
        #
        # `threshold = 9`, NOT `interpolateLiuLiu`'s own 4: the MLS path here
        # has no conditioning guard (English §3 / DualSPHysics both gate on a
        # `determlimit`; `interp.py` only pinv's `A_g`, which passes a
        # near-singular direction straight through). A boundary particle under
        # the thin, fast dam-break front sliding over the dry bed sees ~5-9
        # fluid neighbours all in a shallow horizontal band -> the vertical
        # moment of `A_g` is tiny -> `rho_interp_grad` blows up -> `rho_proj`
        # is wild -> `P_b = c0^2 (rho_proj - rho0)` (with c0 = 40 sqrt(gH),
        # c0^2 ~ 9400) flings the sheet off the bed *before* it reaches the
        # end wall (t ~ 0.54 s in the Marrone 3.1 case). Dropping this to 4
        # this session is what broke that case; `DELTASPH_VALIDATION_PLAN.md`
        # Part 3 owns the principled fix (a determinant / condition gate so
        # the MLS path can safely extend below 9).
        threshold = 9

        drho = -torch.einsum('nu, nu -> n', relPos, rho_interp_grad)
        rho_proj = (rho_interp + drho)
        boundaryDensity[bIndices] = torch.where(numNeighbors > threshold, rho_proj, boundaryDensity[bIndices])

        mergedDensitities = currentState.densities.clone()
        mergedDensitities[bIndices] = boundaryDensity[bIndices]

        # print(f'Mdbc densities: min={boundaryDensity[bIndices].min():.3g}, max={boundaryDensity[bIndices].max():.3g}, mean={boundaryDensity[bIndices].mean():.3g}')
        return mergedDensitities