"""Wall closures of the ACSPH residual: the Eq. (61) wall pressure, the unilateral
(Batty et al. 2007) contact variants, the wall-adjacency test and the authors'
position-row wall bounce-back (docs/acsph/DYNGPUSPH_COMPARISON.md D5)."""

from dataclasses import dataclass
from typing import Optional

import torch
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)

from ...enumTypes import PressureForceScheme
from ..surfaceDetection import detectIsolated
from ..incompressible.wallPressure import wallPressureExtrapolation
from ..pressure import computePressureForceSurfaceAware
from .workingState import workingState


def wallPressures(view, config, adjacency, bodyForce, clampNonNeg=False,
                  edgeRelax='off', edgeFlags=None, lambdas=None):
    """`view.pressures` with the `kind == 1` rows filled by Eq. (61)/Adami
    et al. 2012 Eq. 27, the Shepard interpolation of the fluid pressure plus the
    hydrostatic correction `(g - a_w) . sum_f V_f rho_f (r_w - r_f) W_wf`.

    **Not optional for this scheme.** Without the correction the wall reads the
    depth-*average* of its fluid neighbours -- short by `rho0 g <y_f - y_w>`,
    a fraction of `h` -- and a hydrostatic column cannot balance against it.
    Without the extrapolation at all the wall reads `p = 0` (the non-fluid rows
    are masked every step) and the column simply falls out of the box.

    Recomputed at every residual evaluation, because it is a function of the
    *current* fluid pressure: exactly the Robin closure
    `modules/incompressible/wallPressure.py` was built for, applied per RK
    stage here rather than per Jacobi iterate.

    `clampNonNeg=False` by default, unlike the DFSPH call sites: ACSPH's
    pressure is a solved field that legitimately goes negative, and clamping
    the wall to `>= 0` would make it a one-sided constraint.

    `edgeRelax` ('zero' | 'lambda', `acParams.wallPressureEdgeRelax`): the
    authors' narrower alternative -- a NEGATIVE wall pressure is attenuated only
    where a raw free-surface fluid row (`edgeFlags`) is in the wall row's
    support (`wallPressure._edgeRelaxFactor`, `docs/acsph/DYNGPUSPH_COMPARISON.md`
    D1).
    """
    fluid = view.kinds == 0
    return wallPressureExtrapolation(view, config, adjacency, view.pressures,
                                     fluid, mode='shepard',
                                     clampNonNeg=clampNonNeg, bodyForce=bodyForce,
                                     edgeRelax=edgeRelax, edgeFlags=edgeFlags,
                                     lambdas=lambdas)


def unilateralWallForce(view, config, schemeConfig, adjacency):
    """Pressure force with the fluid-solid pair terms made unilateral
    (`cavitationProjection = 'wall'`, Batty et al. 2007 Eq. 15 applied to the
    solid contact only; its kinematic partner is `unilateralWallDivergence`): `-sum_wall V_j (p_i^+ + p_w^+) gradW` in place of
    `-sum_wall V_j (p_i + p_w) gradW`, every other term untouched -- the
    pressure solve, the divergence, and the fluid-fluid pairs, whose tension is
    what lets a droplet coalesce (MDBC_CONTACT_LINE_PLAN.md §9.2: projecting
    p >= 0 on fluid rows made every droplet-surface contact elastic). The wall
    can push but not pull; omniSPH's boundary force does the same
    (`max(0, p_fluid)`, `max(0, p_boundary)`, fluidMechanics.cpp:464-465).

    Exact by linearity of Eq. (25)'s `(p_i + p_j)` form, with `A(q)` the force
    operator on a pressure field `q`:
      F' = A(p) + A((p^+ - p) 1_wall) + (p_i^+ - p_i) A(1_wall)
    where `A(1_wall)` at a fluid row is the coefficient of its own `p_i` over
    its wall pairs."""
    if schemeConfig.pressureForceTerm != PressureForceScheme.nonConservative:
        raise ValueError("cavitationProjection='wall' needs the linear "
                         "pressureForceTerm=nonConservative ((p_i + p_j), Eq. 25)")
    p = view.pressures
    wall = view.kinds == 1

    def A(q):
        qView = workingState(view, view.positions, view.velocities, q)
        return computePressureForceSurfaceAware(qView, config, schemeConfig, adjacency)

    zero = torch.zeros_like(p)
    dp = p.clamp_min(0.0) - p
    return (A(p) + A(torch.where(wall, dp, zero))
            + dp.unsqueeze(-1) * A(wall.to(p.dtype)))


def unilateralWallDivergence(view, config, adjacency, divV):
    """`div v` with the fluid-solid pairs made unilateral: the kinematic half
    of Batty et al. 2007's complementarity `0 <= p ⊥ (u - v_s).n >= 0`, the
    partner of `unilateralWallForce`. With only the force unilateral, a row
    separating from a wall kept the positive divergence of its mirrored wall
    pairs, which the wall could no longer act on, and the pseudo-time drove
    its pressure without bound (Marrone 3.1 nx=24: -12 -> -1.8e6 in 0.02 s,
    MDBC_CONTACT_LINE_PLAN.md §9.3).

    The wall pairs' share of a row's divergence (Difference form,
    `sum_j V_j (v_j - v_i).gradW`) is split out by linearity,
      divWall_i = D(v 1_wall)_i - v_i . G_i,   G_i = sum_wall V_j gradW_ij,
    and kept only while it compresses (`min(divWall, 0)`): a wall can resist
    approach, never separation. Clamped per row, not per pair -- exact for a
    row against one wall, an approximation in a corner with one wall
    approaching and another receding."""
    wall = view.kinds == 1

    def op(operation, values):
        return warpOperation(
            view,
            OperationProperties(kernel=config.kernel, operation=operation,
                                supportMode=SupportScheme.SuperSymmetric,
                                operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Difference),
            queryValues=values, domain=config.domain, adjacency=adjacency)

    vWall = torch.where(wall.unsqueeze(-1), view.velocities, torch.zeros_like(view.velocities))
    G = op(WarpOperation.Gradient, wall.to(view.velocities.dtype))
    divWall = op(WarpOperation.Divergence, vWall) - (view.velocities * G).sum(-1)
    return divV - divWall + divWall.clamp_max(0.0)


def wallAdjacent(view, config, adjacency):
    """Boolean per row: True when at least one wall (`kind == 1`) row lies in
    its kernel support -- the SPH analogue of Batty et al. 2007's "pressure
    unknowns on solid boundaries" (`cavitationProjection = 'contact'`).

    Exact, no threshold, the `detectIsolated` construction restricted to the
    wall pairs: `sum_wall V_j (x_j - x_i) . gradW_ij` is a sum of same-signed
    terms, one per wall neighbour, and exactly 0.0 with none. Split out by
    linearity as `D(x 1_wall)_i - x_i . G_i` (`unilateralWallDivergence`)."""
    wall = view.kinds == 1

    def op(operation, values):
        return warpOperation(
            view,
            OperationProperties(kernel=config.kernel, operation=operation,
                                supportMode=SupportScheme.SuperSymmetric,
                                operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Difference),
            queryValues=values, domain=config.domain, adjacency=adjacency)

    x = view.positions
    xWall = torch.where(wall.unsqueeze(-1), x, torch.zeros_like(x))
    G = op(WarpOperation.Gradient, wall.to(x.dtype))
    trace = op(WarpOperation.Divergence, xWall) - (x * G).sum(-1)
    return trace != 0.0


def wallBounceSetup(state, adjacency):
    """Per-step constants for the position-row bounce-back: the fluid-wall
    candidate pairs `(pi, pj)` of the step's adjacency and an inward unit normal
    per wall row.

    The normal of wall row `b` is the mean direction of its mDBC ghost offsets,
    flipped if needed to point at its fluid neighbours (dynGPUSPH flips its
    kernel-gradient normal the same way, `noSlipHashD`). Rows with no ghost get
    a zero normal and never bounce."""
    kinds = state.kinds
    i, j = (t.long() for t in adjacency.livePairs())
    m = (kinds[i] == 0) & (kinds[j] == 1)
    pi, pj = i[m], j[m]
    nW = torch.zeros_like(state.positions)
    ghost = kinds == 2
    if getattr(state, 'ghostIndices', None) is not None and bool(ghost.any()):
        off = state.ghostOffsets[ghost]
        unit = off / (off.norm(dim=-1, keepdim=True) + 1e-12)
        nW.index_add_(0, state.ghostIndices[ghost].long(), unit)
        nW = torch.nn.functional.normalize(nW, dim=-1)
        side = ((state.positions[pi] - state.positions[pj]) * nW[pj]).sum(-1)
        score = torch.zeros(nW.shape[0], dtype=side.dtype, device=side.device).index_add_(0, pj, side)
        nW = torch.where((score < 0).unsqueeze(-1), -nW, nW)
    return pi, pj, nW


def wallBounce(positions, velocities, pi, pj, nW, dx, distance, ramp=0.0):
    """`velocities` with, for every fluid row within `distance * dx` of a wall
    row (its nearest wall by normal distance, among wall rows within 1.155 dx)
    and moving into that wall (`v . n < 0`), the normal component reversed.
    Pure tensor ops on fixed-shape per-step constants: graph-capturable.

    `ramp = 0` is the authors' hard switch (a jump at `distance`). `ramp in (0, 1]`
    makes the reversal continuous in the distance: full below `(1 - ramp) *
    distance`, a C2 quintic fall to nothing at `distance`. The jump makes a row
    that rests at the threshold flip between bounced and not bounced from one
    pseudo-iteration to the next, so its `tilde v` never converges."""
    n = positions.shape[0]
    d = ((positions[pi] - positions[pj]) * nW[pj]).sum(-1)
    near = (positions[pi] - positions[pj]).norm(dim=-1) < 1.155 * dx
    ad = torch.where(near, d.abs(), torch.full_like(d, float('inf')))
    best = torch.full((n,), float('inf'), dtype=d.dtype, device=d.device).scatter_reduce(
        0, pi, ad, 'amin', include_self=True)
    isBest = near & (ad <= best[pi])
    nRow = torch.zeros_like(positions).index_add_(0, pi, nW[pj] * isBest.unsqueeze(-1).to(d.dtype))
    nRow = torch.nn.functional.normalize(nRow, dim=-1)
    vn = (velocities * nRow).sum(-1, keepdim=True)
    if ramp <= 0.0:
        hit = ((best < distance * dx) & (vn.squeeze(-1) < 0.0)).unsqueeze(-1)
        return torch.where(hit, velocities - 2.0 * vn * nRow, velocities)
    d0 = (1.0 - ramp) * distance * dx
    s = ((best - d0) / (distance * dx - d0)).clamp(0.0, 1.0)
    weight = (1.0 - s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)).unsqueeze(-1)    # 1 -> 0, C2
    return velocities - 2.0 * weight * vn.clamp(max=0.0) * nRow



@dataclass(frozen=True)
class WallBounce:
    """The step's wall bounce-back (D5): the fluid-wall pair set and wall normals, fixed for
    the real step. Apply with `bounceVelocities`."""
    pi: torch.Tensor
    pj: torch.Tensor
    normals: torch.Tensor
    dx: float
    distance: float
    ramp: float


def bounceVelocities(bounce: Optional[WallBounce], positions, velocities):
    """`velocities` with the wall bounce applied (unchanged when the bounce is off, i.e. None)."""
    if bounce is None:
        return velocities
    return wallBounce(positions, velocities, bounce.pi, bounce.pj, bounce.normals,
                      bounce.dx, bounce.distance, bounce.ramp)


_warnedNoWallNormals = False


def makeWallBounce(state, adjacency, config, acParams) -> Optional[WallBounce]:
    """`WallBounce` for this step, or None when `acParams.wallBounceBack` is off.
    Prints once per run how many wall rows carry a normal (a no-op without mDBC
    ghost points, which provide the normals)."""
    if not acParams.wallBounceBack:
        return None
    pi, pj, normals = wallBounceSetup(state, adjacency)
    global _warnedNoWallNormals
    if not _warnedNoWallNormals:   # one host read per run, not per step
        _warnedNoWallNormals = True
        nWithNormal = int((normals.abs().sum(-1) > 0).sum())
        print(f"[warpSPH] acsph: wallBounceBack active, {nWithNormal} wall rows carry a normal, "
              f"{int(pi.numel())} fluid-wall candidate pairs", flush=True)
        if nWithNormal == 0:
            print("[warpSPH] acsph: WARNING wallBounceBack found no wall normals (needs mDBC ghost "
                  "points); it will do nothing", flush=True)
    return WallBounce(pi, pj, normals, float(config.dx), acParams.wallBounceDistance,
                      acParams.wallBounceRamp)


@dataclass(frozen=True)
class ContactMasks:
    """Unilateral-contact projections of the pressure (`cavitationProjection`,
    `isolatedZeroPressure`): rows that may not carry tension, rows set to zero (None = off).
    Apply with `projectPressure`."""
    cavity: Optional[torch.Tensor]
    isolated: Optional[torch.Tensor]


def projectPressure(contact: ContactMasks, p):
    if contact.cavity is not None:
        p = torch.where(contact.cavity, p.clamp_min(0.0), p)
    if contact.isolated is not None:
        p = torch.where(contact.isolated, torch.zeros_like(p), p)
    return p


def contactMasks(state, config, schemeConfig, adjacency, surface) -> ContactMasks:
    """Rows where air can reach (Batty et al. 2007 Eq. 15), by `acParams.cavitationProjection`:
    'off' / 'wall' (no projection; 'wall' changes the pair terms instead), 'vicinity' (the
    dilated surface set plus isolated rows, fluid and wall), 'contact' (every wall row and
    every fluid row with a wall in its support), 'fluid' (all fluid and wall rows). The
    fixed point of the projected pseudo-time supplies the kinematic half
    (MDBC_CONTACT_LINE_PLAN.md §10.3). `isolatedZeroPressure`: p = 0 on rows with an empty
    support (§11.1)."""
    acParams = schemeConfig.acParams
    fluid = state.kinds == 0
    wall = state.kinds == 1
    mode = acParams.cavitationProjection
    if mode in ('off', 'wall'):
        cavity = None
    elif mode == 'vicinity':
        cavity = (fluid | wall) & (surface.dilated | detectIsolated(state, config, adjacency))
    elif mode == 'contact':
        cavity = wall | (fluid & wallAdjacent(state, config, adjacency))
    elif mode == 'fluid':
        cavity = fluid | wall
    else:
        raise ValueError(f"acParams.cavitationProjection must be 'off', 'wall', "
                         f"'vicinity', 'contact' or 'fluid', got {mode!r}")
    isolated = (fluid & detectIsolated(state, config, adjacency)
                if acParams.isolatedZeroPressure else None)
    return ContactMasks(cavity, isolated)
