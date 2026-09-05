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
               `bodyForceTerm` is the full Adami correction (see below);
               pass `bodyForce` to enable it -- `None` (default) omits it and
               the mirror is unforced.

The Adami body-force (hydrostatic) correction, shared by 'shepard' and
'mirror' (`_bodyForceMoment`). Adami et al. 2012 Eq. 27 is

    p_w = [ sum_f p_f W_wf + (g - a_w) . sum_f rho_f r_wf W_wf ] / sum_f W_wf

with `r_wf = r_w - r_f`; De Courcy et al. 2024 Eq. (61) is the same thing
volume-weighted, `sum_f [...] W_bf V_f / sum_f W_bf V_f`. This module's
Shepard gather is volume-weighted, so 'shepard' + `bodyForce=g` reproduces
Eq. (61) *exactly* -- that is the wall BC ACSPH needs, and without it a
hydrostatic column cannot hold its gradient (the wall reads back the
depth-averaged fluid pressure instead of the pressure at the wall plane).

The moment `sum_f V_f rho_f (r_w - r_f) W_wf` is a *vector* moment of the
neighbourhood, which no single `WarpOperation` returns. It is assembled from
two `Interpolate` gathers instead, using the fact that `(g - a_w)` is constant
over the sum (it is a per-*wall*-particle quantity):

    sum_f V_f rho_f r_wf W_wf = r_w * sum_f V_f rho_f W_wf - sum_f V_f rho_f r_f W_wf
                                \_____ scalar gather _____/   \____ vector gather ____/

Both gathers run with the same `OperationProperties` as the value term, so
numerator and denominator share one kernel evaluation. `bodyForce` is
`(g - a_wall)`: pass a `(dim,)` tensor for a uniform field or `(N, dim)` for a
per-particle one (read at the wall rows). Static walls have `a_wall = 0` -- no
per-particle prescribed wall acceleration is tracked anywhere in this codebase
yet, see `modules/mdbc/velocity.py`'s docstring on the same gap for the
velocity mirror's dead `2*u_wall` term.

    WARNING -- periodic domains. Splitting `r_w - r_f` across two gathers
    discards the minimum-image convention: a pair that wraps the domain
    contributes `+-L_d` of error in each periodic direction `d`. Two things
    make that harmless in every case this is used on, and `_bodyForceMoment`
    checks both rather than assuming either:

      1. The dot with `bodyForce` annihilates the error along any axis where
         `bodyForce` has no component -- a box periodic across the flow but
         walled along gravity contributes nothing.
      2. Along an axis where `bodyForce` *does* act, a pair can only wrap if
         there is a `kind == 1` row within one support radius of one domain
         face and a fluid row within one support radius of the other. Wall
         particles bound the domain in the walled directions, so in practice
         there is no fluid on the far side to gather.

    `_wrappingPairsArePossible` is that second test, `O(N)` and one-sided: it
    can only over-report, never miss a genuine wrap. When both fail,
    `_bodyForceMoment` raises rather than silently returning a wrong wall
    pressure. Lift the restriction properly by writing a real moment kernel.

Unlike the removed `BoundaryPressureMode.mdbcMlsPressure` (its
`computeMdbcPressure`, pre-merge cleanup pass 09-04) there is NO relaxation
and NO cross-step carried state, so this does not have that mode's
boundary-pressure feedback instability (DFSPH_FINDINGS.md Sec. 3.1):
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


def _normalizeBodyForce(bodyForce, state):
    """`bodyForce` -> an `(N, dim)` tensor of `(g - a_wall)` per row. Accepts a
    `(dim,)` uniform field, an `(N, dim)` per-particle one, or anything
    `torch.as_tensor` turns into either."""
    dim = state.positions.shape[1]
    bf = torch.as_tensor(bodyForce, device=state.positions.device,
                         dtype=state.positions.dtype)
    if bf.ndim == 1 and bf.shape[0] == dim:
        return bf.unsqueeze(0).expand(state.positions.shape[0], dim)
    if bf.shape == state.positions.shape:
        return bf
    raise ValueError(
        f"bodyForce must have shape ({dim},) or {tuple(state.positions.shape)}, "
        f"got {tuple(bf.shape)}")


def _wrappingPairsArePossible(state, config, axis):
    """Could a `kind == 1` row and a fluid row be minimum-image neighbours
    *across* the domain in `axis`? One-sided: True means "cannot rule it out".

    A wrap needs one of the pair within a support radius of the low face and
    the other within a support radius of the high face. Checked in both
    role assignments, since either can be the boundary row.
    """
    domain = config.domain
    low = float(torch.as_tensor(domain.min)[axis])
    high = float(torch.as_tensor(domain.max)[axis])
    coordinate = state.positions[:, axis]
    reach = state.supports

    boundary = state.kinds == 1
    fluid = state.kinds == 0
    nearLow = (coordinate - low) < reach
    nearHigh = (high - coordinate) < reach

    return (bool((boundary & nearLow).any()) and bool((fluid & nearHigh).any())) or \
           (bool((fluid & nearLow).any()) and bool((boundary & nearHigh).any()))


def _checkPeriodicSafety(state, config, bodyForce):
    """Raise unless the two-gather moment is exact for this configuration --
    see the module docstring's two conditions."""
    periodic = getattr(config.domain, 'periodic', None)
    if periodic is None:
        return
    periodic = torch.as_tensor(periodic)
    if not bool(periodic.any()):
        return
    acts = bodyForce.abs().amax(dim=0) > 0
    for axis in range(state.positions.shape[1]):
        if not bool(periodic[axis]) or not bool(acts[axis]):
            continue
        if _wrappingPairsArePossible(state, config, axis):
            raise ValueError(
                f"bodyForce acts along periodic axis {axis}, and fluid/boundary "
                f"rows sit within a support radius of both faces there, so a "
                f"minimum-image pair can wrap -- the two-gather moment "
                f"decomposition would be wrong. See `wallPressure.py`'s module "
                f"docstring; make the axis non-periodic, or wall it.")


def _bodyForceMoment(state, config, adjacency, den, bodyForce):
    """The Adami hydrostatic correction

        (g - a_w) . sum_f V_f rho_f (r_w - r_f) W_wf / sum_f V_f W_wf

    at every row, assembled from two `Interpolate` gathers -- see the module
    docstring for the decomposition, the shared normalisation `den`, and the
    periodic-domain restriction it asserts."""
    bf = _normalizeBodyForce(bodyForce, state)
    _checkPeriodicSafety(state, config, bf)

    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.FluidToBoundary)
    rho = state.densities
    # sum_f V_f rho_f W_wf  and  sum_f V_f rho_f r_f W_wf
    rhoW = warpOperation(state, props, domain=config.domain,
                         referenceValues=rho, adjacency=adjacency)
    rhoXW = warpOperation(state, props, domain=config.domain,
                          referenceValues=rho.unsqueeze(-1) * state.positions,
                          adjacency=adjacency)
    moment = state.positions * rhoW.unsqueeze(-1) - rhoXW
    return (bf * moment).sum(-1) / den.clamp_min(1e-12)


def _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg=True,
                   bodyForce=None):
    """Zero-order Shepard mirror. With `bodyForce` this is Adami et al. 2012
    Eq. 27 / De Courcy et al. 2024 Eq. (61) exactly -- see the module
    docstring's Adami-correction section."""
    boundary = state.kinds == 1
    p_b, den = _shepardValue(state, config, adjacency, p)
    if bodyForce is not None:
        p_b = p_b + _bodyForceMoment(state, config, adjacency, den, bodyForce)
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

    `bodyForce` is the Adami hydrostatic correction `(g - a_wall) . sum_f V_f
    rho_f (r_wall - r_f) W / sum_f V_f W`, added to the reflected value the
    same way `_shepardMirror` adds it to the interpolated one; see the module
    docstring's Adami-correction section for the decomposition and its
    periodic-domain restriction. `None` (the default) omits it -- every DFSPH
    call site so far runs this solve with the body force excluded from `dvdt`
    (the VD+PS shift only fires when `not gravityConfig.active`), so the term
    is provably zero there.
    """
    boundary = state.kinds == 1
    qP, den = _shepardValue(state, config, adjacency, p)
    p_wall = state.pressures
    p_b = 2.0 * p_wall - qP
    if bodyForce is not None:
        p_b = p_b + _bodyForceMoment(state, config, adjacency, den, bodyForce)
    if clampNonNeg:
        p_b = p_b.clamp(min=0.0)
    p_b = torch.where(den > _MIN_WEIGHT, p_b, torch.zeros_like(p_b))
    return torch.where(boundary, p_b, torch.where(fluid, p, torch.zeros_like(p)))


def wallPressureExtrapolation(state: Any, config: Any, adjacency: Any,
                              p: torch.Tensor, fluid: torch.Tensor, *,
                              mode: str = 'mls',
                              clampNonNeg: bool = True,
                              bodyForce: Any = None) -> torch.Tensor:
    """Return `p` with the `kind == 1` rows filled with the extrapolated wall
    pressure. `mode` in {'shepard', 'mls', 'mirror'}; `'mls'` falls back to
    'shepard' when there are no ghost points to run the Liu-Liu fit at. A
    no-op (returns `p` unchanged) when the state has no boundary particles.

    `clampNonNeg` (default True) clamps the extrapolated `p_b` to `>= 0`, the
    physical (tensile-cut) wall pressure. Pass `False` to keep the closure a
    genuine *linear* function of `p` -- required when a Krylov method drives
    this operator (`omniIncompressible.CD_SOLVER in {'bicgstab', 'gmres'}`),
    where the iterate `p` legitimately goes negative and the clamp would make
    the matvec nonlinear.

    `bodyForce` (default `None`) enables the Adami hydrostatic correction on
    the 'shepard' and 'mirror' closures -- pass `(g - a_wall)` as a `(dim,)`
    or `(N, dim)` tensor. `'shepard'` + `bodyForce` is Adami et al. 2012
    Eq. 27 / De Courcy et al. 2024 Eq. (61) exactly. Not supported by 'mls',
    whose first-order Liu-Liu fit already carries the local pressure gradient
    (adding the correction on top would double-count it)."""
    boundary = state.kinds == 1
    if not bool(boundary.any()):
        return p

    if mode == 'shepard':
        return _shepardMirror(state, config, adjacency, p, fluid, clampNonNeg,
                              bodyForce=bodyForce)
    if mode == 'mirror':
        return _adamiMirror(state, config, adjacency, p, fluid, clampNonNeg,
                            bodyForce=bodyForce)

    if bodyForce is not None:
        raise ValueError(
            "mode='mls' does not take a bodyForce correction -- the Liu-Liu "
            "linear fit already carries the local pressure gradient. Use "
            "mode='shepard' for the Adami/De Courcy Eq. (61) closure.")

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
