"""Per-iterate wall-pressure extrapolation for the incompressible pressure
solves (`omniIncompressible`, `iisph`, `dfsphReference`).

Why this exists (DFSPH_IMPROVEMENT_PLAN.md Part 41). omniSPH's `densitySolve`
recomputes an MLS wall pressure `p_b` from the *current* fluid pressure every
Jacobi iterate and feeds its gradient into the pressure acceleration -- a Robin
closure that makes the near-wall iteration contract (omniSPH: 4 iterations).
This codebase's composed solves put the wall into the diagonal `alpha` only
(via `applyConsistentCoupling`'s Akinci band) and run `computePressureAccelIISPH`
at boundary `p == 0` (Bender-Westhofen-Jeske 2023 Eq. 33). The near-wall
iteration matrix is then inconsistent -- `D` carries a wall term the operator
`A p` does not -- and the Jacobi does not contract at a wall band: on
`hydrostaticColumn` nx=128 it hits its iteration cap with the residual rising
and blows up at the bottom corners.

`wallPressureExtrapolation` closes that gap: given the current fluid pressure
vector `p`, it returns `p_all` with the `kind == 1` rows filled with an
extrapolated wall pressure, so a subsequent `computePressureAccelIISPH(p_all)`
picks up the wall term `-sum_k m_k (p_i/rho_i^2 + p_b/rho0^2) gradW_ik`. Call
it once per Jacobi iterate, inside `applyConsistentCoupling` (so boundary rows
already carry `rho_k = rho0`). Mask the resulting `a_p` back to the fluid rows
as before -- leaving `a_p == 0` on the boundary rows in the divergence step
reproduces omniSPH's `a_i . gk` self-term.

Modes:
  'shepard' -- zero-order mirror `p_b[k] = sum_f V_f p_f W_kf / sum_f V_f W_kf`
               (omniSPH's MLS `alpha` term, no linear correction).
  'mls'     -- the full first-order Liu-Liu fit (`modules/liu`, the same one
               `computeMdbcPressure` uses): value + gradient at each `kind == 2`
               ghost point, first-order Taylor-corrected to the owning boundary
               particle via `ghostOffsets` / `ghostIndices`, Shepard fallback
               where under-determined. This is omniSPH's
               `p_b = alpha + beta*x_b + gamma*y_b` in full.

Unlike `computeMdbcPressure` there is NO relaxation and NO cross-step carried
state, so this is not the `mdbcMlsPressure` boundary-pressure feedback
instability (DFSPH_FINDINGS.md Sec. 3.1 / probe_mdbcMlsPressureInstability.py):
`p_b` is a fresh function of the current iterate, exactly as omniSPH's is.
"""

from typing import Any

import torch

from warpSPHCore import (AdjacencyList, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)

from ..liu import interpolateLiuLiu

__all__ = ['wallPressureExtrapolation']

#: Boundary rows whose Shepard fluid-weight `sum_f V_f W_kf` is below this
#: (deep in a multi-layer band, no real fluid neighbours) get `p_b = 0`.
_MIN_WEIGHT = 1e-6
#: Liu-Liu neighbour-count thresholds: >`_SHEPARD_MIN` uses the 0th-order
#: Shepard value, >`_LINEAR_MIN` uses the full value+gradient projection
#: (matching `computeMdbcPressure`).
_SHEPARD_MIN = 1
_LINEAR_MIN = 9


def _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg=True):
    boundary = state.kinds == 1
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.FluidToBoundary)
    num = warpOperation(state, props, domain=config.domain,
                        referenceValues=p, adjacency=adjacency)
    den = warpOperation(state, props, domain=config.domain,
                        referenceValues=torch.ones_like(p), adjacency=adjacency)
    p_b = num / den.clamp_min(1e-12)
    if clampNonNeg:
        p_b = p_b.clamp(min=0.0)
    p_b = torch.where(den > _MIN_WEIGHT, p_b, torch.zeros_like(p_b))
    return torch.where(boundary, p_b, torch.where(fluid, p, torch.zeros_like(p)))


def wallPressureExtrapolation(state: Any, config: Any, adjacency: Any,
                              p: torch.Tensor, fluid: torch.Tensor, *,
                              mode: str = 'mls',
                              clampNonNeg: bool = True) -> torch.Tensor:
    """Return `p` with the `kind == 1` rows filled with the extrapolated wall
    pressure. `mode` in {'shepard', 'mls'}; falls back to 'shepard' when there
    are no ghost points to run the Liu-Liu fit at. A no-op (returns `p`
    unchanged) when the state has no boundary particles.

    `clampNonNeg` (default True) clamps the extrapolated `p_b` to `>= 0`, the
    physical (tensile-cut) wall pressure. Pass `False` to keep the closure a
    genuine *linear* function of `p` -- required when a Krylov method drives
    this operator (`omniIncompressible.CD_SOLVER in {'bicgstab', 'gmres'}`),
    where the iterate `p` legitimately goes negative and the clamp would make
    the matvec nonlinear."""
    boundary = state.kinds == 1
    if not bool(boundary.any()):
        return p

    if mode == 'shepard':
        return _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg)

    ghost = state.kinds == 2
    if getattr(state, 'ghostIndices', None) is None or not bool(ghost.any()):
        return _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg)

    hashMap = adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None
    p_interp, p_grad, nNbr, A_g, b = interpolateLiuLiu(
        state.positions[ghost], referenceParticles=state, referenceQuantities=p,
        config=config, neighbor_threshold=4,
        direction=OperationDirection.FluidToGhost, supportScale=1.0,
        adjacency=hashMap)
    bIdx = state.ghostIndices[ghost]
    relPos = -state.ghostOffsets[ghost]
    dP = -torch.einsum('nu,nu->n', relPos, p_grad)
    p_proj = p_interp + dP
    shepDen = A_g[:, 0, 0]
    shep = torch.where(shepDen > 0, b[:, 0] / shepDen.clamp_min(1e-12),
                       torch.zeros_like(shepDen))
    p_b = torch.where(nNbr > _SHEPARD_MIN, shep, torch.zeros_like(shep))
    p_b = torch.where(nNbr > _LINEAR_MIN, p_proj, p_b)
    if clampNonNeg:
        p_b = p_b.clamp(min=0.0)

    out = torch.where(fluid, p, torch.zeros_like(p)).clone()
    out[bIdx] = p_b
    return out
