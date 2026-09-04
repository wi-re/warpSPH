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
               (omniSPH's MLS `alpha` term, no linear correction). A
               *Dirichlet*-style closure: the wall takes the fluid's own
               extrapolated value directly.
  'mls'     -- the full first-order Liu-Liu fit (`modules/liu`, the same one
               `computeMdbcPressure` uses): value + gradient at each `kind == 2`
               ghost point, first-order Taylor-corrected to the owning boundary
               particle via `ghostOffsets` / `ghostIndices`, Shepard fallback
               where under-determined. This is omniSPH's
               `p_b = alpha + beta*x_b + gamma*y_b` in full.
  'mirror'  -- Adami et al. 2012's wall pressure BC, in the same *reflection*
               form `computeBoundaryVelocities`'s no-slip mirror already uses
               (`modules/mdbc/velocity.py`'s `u_g = 2*bodyVelocity - qVel`):
               `p_b = 2*p_wall - shepard_f(p) + bodyForceTerm`, reflecting the
               Shepard-interpolated fluid pressure about the wall's own
               currently-held value (`p_wall`, `state.pressures` at the
               boundary rows -- the "own value" `bodyVelocity` plays for
               velocity) rather than returning the fluid value directly. A
               *Neumann*-style closure in spirit: it is the two-point
               finite-difference approximation to `dp/dn = 0` at the wall
               plane (`p_wall` sits midway between the fluid sample and its
               image, so reflecting about it holds the derivative fixed at
               whatever `p_wall` already encodes) where 'shepard' is the
               zero-derivative-at-the-*fluid*-sample approximation instead.
               `bodyForceTerm` is the full Adami correction `rho_f *
               (bodyForce - a_wall) . (r_wall - r_f)`, Shepard-averaged the
               same way as the value term; pass `bodyForce` (`g`, or `g` plus
               any other per-particle acceleration) to enable it -- `None`
               (default) omits it and the mirror is unforced. Static walls
               have `a_wall = 0` (no per-particle prescribed wall
               acceleration is tracked anywhere in this codebase yet -- see
               `modules/mdbc/velocity.py`'s docstring on the same gap for the
               velocity mirror's dead `2*u_wall` term).

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


def _shepardValue(state, config, adjacency, referenceValues):
    """`sum_f V_f referenceValues_f W_kf / sum_f V_f W_kf` at every `kind==1`
    boundary row -- the raw Shepard gather, no fallback / merge / clamp.
    Shared by `_shepardMirror` (returns this directly) and `_adamiMirror`
    (reflects it about the wall's own value)."""
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.FluidToBoundary)
    num = warpOperation(state, props, domain=config.domain,
                        referenceValues=referenceValues, adjacency=adjacency)
    den = warpOperation(state, props, domain=config.domain,
                        referenceValues=torch.ones_like(referenceValues), adjacency=adjacency)
    return num / den.clamp_min(1e-12), den


def _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg=True):
    boundary = state.kinds == 1
    p_b, den = _shepardValue(state, config, adjacency, p)
    if clampNonNeg:
        p_b = p_b.clamp(min=0.0)
    p_b = torch.where(den > _MIN_WEIGHT, p_b, torch.zeros_like(p_b))
    return torch.where(boundary, p_b, torch.where(fluid, p, torch.zeros_like(p)))


def _adamiMirror(state, config, adjacency, p, fluid, clampNonNeg=True,
                 bodyForce=None):
    """Adami et al. 2012's wall pressure BC -- see `wallPressureExtrapolation`'s
    'mirror' mode docstring. `p_wall` is `state.pressures` at the boundary
    rows (the incoming/carried value, the same role `bodyVelocity` plays in
    `modules/mdbc/velocity.py`'s no-slip mirror).

    `bodyForce` (the Adami hydrostatic correction, `rho_f * (g - a_wall) .
    (r_wall - r_f)`, Shepard-averaged) is **not implemented** -- it needs a
    vector *moment* of the neighbourhood (`Interpolate` only returns scalar
    sums), i.e. a `Gradient`-style operator applied to a constant field. Every
    call site so far runs this solve with the body force excluded from `dvdt`
    (the VD+PS shift only fires when `not gravityConfig.active`), so the term
    would be provably zero anyway; wire it if a gravity-driven case ever needs
    this solve path. Raises if passed a non-`None` value rather than silently
    ignoring it.
    """
    if bodyForce is not None:
        raise NotImplementedError(
            "_adamiMirror's bodyForce correction is not implemented -- see "
            "this function's docstring")
    boundary = state.kinds == 1
    qP, den = _shepardValue(state, config, adjacency, p)
    p_wall = state.pressures
    p_b = 2.0 * p_wall - qP
    if clampNonNeg:
        p_b = p_b.clamp(min=0.0)
    p_b = torch.where(den > _MIN_WEIGHT, p_b, torch.zeros_like(p_b))
    return torch.where(boundary, p_b, torch.where(fluid, p, torch.zeros_like(p)))


def wallPressureExtrapolation(state: Any, config: Any, adjacency: Any,
                              p: torch.Tensor, fluid: torch.Tensor, *,
                              mode: str = 'mls',
                              clampNonNeg: bool = True) -> torch.Tensor:
    """Return `p` with the `kind == 1` rows filled with the extrapolated wall
    pressure. `mode` in {'shepard', 'mls', 'mirror'}; `'mls'` falls back to
    'shepard' when there are no ghost points to run the Liu-Liu fit at. A
    no-op (returns `p` unchanged) when the state has no boundary particles.

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
    if mode == 'mirror':
        return _adamiMirror(state, config, adjacency, p, fluid, clampNonNeg)

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
