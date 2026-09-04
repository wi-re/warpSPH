"""`IncompressibleSPHScheme.band2018pb` -- IISPH with Pressure Boundaries
(Band, Gissler, Ihmsen, Cornelis, Peer, Teschner 2018, ACM TOG 37(2):14).

The extended PPE in which `kind == 1` boundary samples are pressure unknowns
with their own equation and diagonal, iterated in the *same* relaxed-Jacobi
loop as the fluid. This removes the near-wall rank deficiency that stalls the
`omniIncompressible` / `iisph` constant-density solve at a wall corner
(DFSPH_IMPROVEMENT_PLAN.md active track / Parts 41-44); see
`modules/incompressible/pressureBoundaries.py` for the operator-by-operator
mapping.

Step (Algorithm 1 in the paper, one-way static walls):

  1. neighbourhood + summation density (for a sane `m/rho` outside the solve)
  2. external forces -> a  (gravity + forcing), then v* = v + dt a
  3. band rest / actual volumes; source s_i = 1 - V0_i/V_i + dt div v*_i and
     diagonal dt^2 a_ii over fluid + boundary rows
  4. relaxed Jacobi over p = [p_f ; p_b], Eq. 18, min DENSITY_MIN / max
     DENSITY_MAX iterations, warm start 0.5 * p_prior; convergence = mean
     |(A p)_i - s_i| over fluid + boundary rows
  5. XSPH velocity filter (shared knobs with `omniIncompressible`; default off)
  6. integrate: v_f += dt a^p_f ;  x_f += dt v_f   (single symplectic step)

Boundary samples take the pressure solve but are never moved or advected --
`a^p_b := 0` (the paper's static-wall assumption), so only their pressure
value feeds back into the fluid rows through Eq. 8.

Reuses `DFSPHReferenceSystem` (the whole time integration happens in the step;
`finalize` copies the advanced fields across).
"""

from __future__ import annotations

from typing import Any

import torch

from warpSPHCore import (SupportScheme, buildVerletList)

from ..modules.boundaryConditions import (computeForcing, enforceDirichlet,
                                          enforceUpdates)
from ..modules.density import computeDensities
from ..modules.gravity import computeGravity
from ..modules.incompressible.pressureBoundaries import (
    bandActualVolumes, bandApplyOperator, bandBoundaryUnknownMask,
    bandConstantModeRatio, bandDiagonal, bandInjectVolumes, bandPressureAccel,
    bandRelaxation, bandRestVolumes, bandVelocityDivergence, bandWellPosedMask)
from ..systems.incompressible import IncompressibleSystemUpdate

__all__ = ['band2018pb_step']

#: Relaxed-Jacobi base relaxation. The paper uses 0.5 (and states that only
#: `0 < omega <= 0.5 V0_i/h^d` is stable); on this codebase's composed operator
#: at `n_h = 4` the fixed-point window is tighter, in the same family as
#: `omniIncompressible`'s ~0.3. `bandRelaxation` scales every row by this one
#: constant, per the paper's `omega_i = omega_base V0_i/h^d`.
#:
#: `0.1` at every resolution measured (`hydrostaticColumn` 200-300 steps):
#: nx=32 `embMin` 0.997; nx=64 `embMin` 0.999 / `slope` 1.007 / `|v|max` 0.13;
#: nx=128 `embMin` 1.001 / `slope` 1.024 / `|v|max` 0.22 / 2 spray particles.
#: `0.2` and above detonate at nx=128 (`|v|max` ~1e3).
#:
#: The earlier `0.05`, and the note that the window was resolution-dependent
#: (nx=128 "needs" 0.05) because boundary rows "stiffen the operator", were
#: both artifacts of the `bandRelaxation` scaling bug fixed alongside this --
#: the boundary rows, not the fluid, were what the small value was protecting.
OMEGA_FLUID = 0.1
#: Paper Algorithm 1: `while not converged` with a 3-iteration floor, no hard
#: cap stated; matched to `omniIncompressible`'s density solve budget.
DENSITY_MIN_ITERATIONS = 8
DENSITY_MAX_ITERATIONS = 256
#: `|alpha_i| < ALPHA_FLOOR` -> row's pressure held at 0 for the iterate.
ALPHA_FLOOR = 1e-25
#: Convergence-metric residual floor (omniSPH `updatePressure`:
#: `mean_i max(residual_i, -1e-3)`). The `n_h = 4` lattice reads the volume
#: setpoint low near a free surface (1 - V0/V > 0.3 there -- the volume-centric
#: form of DFSPH_FINDINGS.md Sec. 1.1), an irreducible positive source no
#: pressure field can cancel; flooring the residual there (the rows park at
#: `p = 0` under the Eq. 18 clamp anyway) is what lets the solve terminate.
RESIDUAL_FLOOR = -1e-3
DAMPING = 0.0

#: Diagonal-floor / Tikhonov fraction. Kernel-deficient rows (a free surface,
#: or a thin near-wall gap) have `|a_ii| -> 0` -- there is no meaningful
#: pressure equation there, and the Jacobi step `omega / a_ii` blows up. A
#: `computeAlpha`-exact `a_ii` does not help (verified: the band diagonal
#: matches the true `A_ii` to round-off); the fix is to solve the nearby
#: regularised problem `(A - eps|D_med|) p = s` -- deepen every row's diagonal
#: by a uniform absolute `eps * median(|dt^2 a_ii|)` over the fluid rows, the
#: same device as `omniIncompressible.CD_TIKHONOV` (Part 43). `band2018pb`
#: needs it non-zero (unlike `omniIncompressible`, where it is opt-in): the
#: relaxed Jacobi loses the column at nx>=64 without it -- `tik = 0` gives
#: `embMin` 0.44 / 107 spray at nx=64 and 0.87 / 117 at nx=128.
#:
#: Swept at `OMEGA_FLUID = 0.1` with the `bandRelaxation` scaling fixed:
#: `0.05` and `0.1` both hold nx=32/64/128, `0.3` loses nx=128 (`embMin`
#: 0.38). `0.05` is kept because it over-compresses least -- bulk `ρ` 1.0035
#: vs 1.0063 at nx=64 and 1.013 vs 1.023 at nx=128, with `pressureSlopeRatio`
#: correspondingly closer to 1 (1.004 / 1.015 vs 1.007 / 1.024) -- and because
#: the paper carries no Tikhonov term at all, so less of it is more faithful.
#: (The earlier `0.3`, then `0.1`, were tuned against the `bandRelaxation`
#: bug, which is what actually needed the heavy damping.)
DIAG_TIKHONOV = 0.05

#: Closed-domain pressure gauge. `'off'` / `'auto'` (default) / `'always'`.
#:
#: Eq. 8 is a summation gradient, so a uniform `p` produces no acceleration
#: wherever the kernel support is complete. In a fully enclosed domain that is
#: everywhere, so the constant is a null mode of `A` and the pressure is fixed
#: only up to an additive constant -- and the Eq. 18 `max(., 0)` clamp, which
#: can only push a row up, ratchets that constant away instead of pinning it.
#: On `randomFlowIncompressible --bounded` (a closed box, no free surface) this
#: is fatal: step 1 already solves to `p in [1.6e3, 2.6e3]` -- a pure offset,
#: `|a_p| ~ 2e3` -- and KE runs to 1e8 by step 200 while the density stays at
#: [0.992, 1.007], the "a solve overshoots" signature of Part 42.
#:
#: When the gauge fires, each iterate has its mean over the solve rows removed
#: and the non-negativity clamp is dropped: a closed domain has no free surface,
#: so there is no tensile instability for the clamp to suppress, and Part 42
#: reached the same pairing (`p` kept mean-zero, no clamp) for
#: `omniIncompressible`'s closed-box compatibility projection.
#:
#: `'auto'` keys on `bandConstantModeRatio` (`rms|A.1| / rms|a_ii|`) rather than
#: on the *source*: `omniIncompressible.CD_SOURCE_PROJECT`'s mean-dominated-
#: source test does NOT fire here (measured `fracUniform ~ 0.03`, threshold
#: 0.7), because the source is well-conditioned -- it is the *solution* that is
#: unpinned, not the source that is incompatible. Two distinct closed-domain
#: failure modes; this is the second.
CLOSED_DOMAIN_GAUGE = 'auto'
#: `bandConstantModeRatio` below this => the constant is a null mode => gauge.
#: Measured at nx=64 step 1: closed box 0.032, free-surface column 1.29.
NULL_MODE_THRESHOLD = 0.25

#: Post-solve XSPH velocity filter, identical in form to
#: `omniIncompressible._xsphFilter` (omniSPH `XSPH` + `BXSPH`). Default off --
#: `band2018pb` is measured first as the faithful no-dissipation loop.
XSPH_FLUID = 0.0
XSPH_BOUNDARY = 0.0


def _rebuildAdjacency(state: Any, system: Any, config: Any):
    adjacency = buildVerletList(
        state, config.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency, verbose=False)
    system.adjacency = adjacency
    return adjacency


def _solve(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *,
           solveRows: torch.Tensor, fluid: torch.Tensor, rho0: float,
           vStar: torch.Tensor, warmStart: torch.Tensor, dt: float):
    """The extended-PPE relaxed Jacobi over `p = [p_f ; p_b]`.

    Returns `(a_p, p, nIter, err)` with `a_p` the final Eq. 8 pressure
    acceleration (fluid rows only), `err` the last mean-residual value.
    """
    V0 = bandRestVolumes(state, config, adjacency, rho0)
    V = bandActualVolumes(state, config, adjacency, V0, rho0)
    omega = bandRelaxation(state, V0, rho0, OMEGA_FLUID)

    with bandInjectVolumes(state, V):
        diag = bandDiagonal(state, config, schemeConfig, adjacency, V, dt)
        # Drop kernel-deficient rows (near-null diagonal -- free surface, thin
        # gap) from the unknown set: they carry no meaningful pressure equation
        # and `omega / a_ii` detonates there. Held at `p = 0` instead of
        # regularised with a large diagonal shift (which over-compresses the
        # column). Recompute the source only over the rows that survive.
        rows = bandWellPosedMask(diag, fluid, solveRows)

        divVstar = bandVelocityDivergence(state, config, adjacency, vStar, rows)
        source = (1.0 - V0 / V.clamp_min(1e-12)) + dt * divVstar
        source = torch.where(rows, source, torch.zeros_like(source))

        # A small Tikhonov shift still helps the conditioning of what remains.
        if DIAG_TIKHONOV and bool(fluid.any()):
            shift = DIAG_TIKHONOV * float(diag[fluid].abs().median())
        else:
            shift = 0.0
        diagEff = diag - shift
        diagBad = (diagEff.abs() < ALPHA_FLOOR) | ~rows
        invDiag = torch.where(diagBad, torch.zeros_like(diagEff),
                              omega / diagEff.clamp(max=-1e-30))

        # Closed-domain gauge: is the constant a null mode of A? (See
        # `CLOSED_DOMAIN_GAUGE`.) One extra operator application per step.
        if CLOSED_DOMAIN_GAUGE == 'always':
            gauge = True
        elif CLOSED_DOMAIN_GAUGE == 'auto':
            gauge = bandConstantModeRatio(
                state, config, adjacency, diag, V, fluid, rows,
                dt) < NULL_MODE_THRESHOLD
        elif CLOSED_DOMAIN_GAUGE == 'off':
            gauge = False
        else:
            raise ValueError(
                f'Unknown CLOSED_DOMAIN_GAUGE: {CLOSED_DOMAIN_GAUGE!r}')

        p = torch.where(rows, warmStart, torch.zeros_like(warmStart))
        err = 0.0
        it = 0
        for it in range(DENSITY_MAX_ITERATIONS):
            a_p = bandPressureAccel(state, config, adjacency, p, V, fluid)
            Ap = bandApplyOperator(state, config, adjacency, a_p, dt, rows)
            if shift:
                Ap = Ap - shift * p          # `(A - shift*I) p`
            p = p + invDiag * (source - Ap)
            if gauge:
                # Pin the unconstrained constant instead of clamping (a closed
                # domain has no free surface, so no tensile instability).
                p = p - torch.where(rows, p[rows].mean().expand_as(p),
                                    torch.zeros_like(p))
                p = torch.where(rows, p, torch.zeros_like(p))
            else:
                p = p.clamp(min=0.0)
            bad = diagBad | (~torch.isfinite(p)) | (p.abs() > 1e25) | ~rows
            p = torch.where(bad, torch.zeros_like(p), p)
            if rows.any():
                residual = torch.clamp(Ap - source, min=RESIDUAL_FLOOR)
                err = float(residual[rows].mean())
            if it + 1 >= DENSITY_MIN_ITERATIONS and err <= _tol(schemeConfig):
                break
        a_p = bandPressureAccel(state, config, adjacency, p, V, fluid)
    return a_p, p, it + 1, err


def _tol(schemeConfig: Any) -> float:
    return float(schemeConfig.solverConfig.pressureSolver.tolerance)


def _xsphFilter(state: Any, config: Any, adjacency: Any,
                fluidMask: torch.Tensor) -> torch.Tensor:
    from warpSPHCore import OperationProperties, WarpOperation, warpOperation
    c = torch.where(fluidMask,
                    torch.full_like(state.densities, XSPH_FLUID),
                    torch.full_like(state.densities, XSPH_BOUNDARY))
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric)
    cv = warpOperation(state, props, domain=config.domain,
                       referenceValues=state.velocities * c.unsqueeze(-1),
                       adjacency=adjacency)
    cc = warpOperation(state, props, domain=config.domain,
                       referenceValues=c, adjacency=adjacency)
    dv = cv - state.velocities * cc.unsqueeze(-1)
    return torch.where(fluidMask.unsqueeze(-1), dv, torch.zeros_like(dv))


def band2018pb_step(system: Any, dt: float, config: Any,
                    schemeConfig: Any, verbose: bool = False):
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity

    # --- 1. neighbourhood + summation density ------------------------------
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)

    # `kind == 1` interface samples (fluid contact) are pressure unknowns; the
    # deeper band layers are not (Band et al. assume one boundary layer).
    boundaryUnknown = bandBoundaryUnknownMask(st, config, adjacency, rho0)
    boundary = boundaryUnknown
    solveRows = fluid | boundaryUnknown

    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    pPrior = st.pressures.clone()

    # --- 2. external forces -> a,  v* = v + dt a --------------------------
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    accel = computeGravity(st, config, schemeConfig, adjacency)
    forcing = computeForcing(system, dt, system.t, config, schemeConfig)
    accel = accel + forcing / st.masses.view(-1, 1)
    accel = torch.where(fcol, accel, torch.zeros_like(accel))
    vStar = st.velocities + dt * accel

    # --- 3-4. the extended-PPE solve over [p_f ; p_b] --------------------
    a_p, pSolve, nIt, err = _solve(
        st, config, schemeConfig, adjacency, solveRows=solveRows, fluid=fluid,
        rho0=rho0, vStar=vStar, warmStart=0.5 * pPrior, dt=dt)
    accel = accel + a_p

    # --- 5. XSPH velocity filter (default off) --------------------------
    if XSPH_FLUID != 0.0 or XSPH_BOUNDARY != 0.0:
        st.velocities = st.velocities + _xsphFilter(st, config, adjacency, fluid)

    # --- 6. integrate (single semi-implicit Euler, fluid only) ---------
    st.velocities = st.velocities + dt * torch.where(
        fcol, accel, torch.zeros_like(accel))
    if DAMPING != 0.0:
        st.velocities = st.velocities * (1.0 - DAMPING)
    st.positions = st.positions + dt * torch.where(
        fcol, st.velocities, torch.zeros_like(st.velocities))

    st.pressures = pSolve

    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        pf = pSolve[fluid]
        pb = pSolve[boundary]
        print(f'[band2018pb] t={system.t + dt:.4g}  RHO {nIt:3d} it err {err:.3g}'
              f'   |v|max {vmax:.4g}   p_f[{float(pf.min()):+.3g},'
              f' {float(pf.max()):+.3g}]  p_b[{float(pb.min()) if pb.numel() else 0:+.3g},'
              f' {float(pb.max()) if pb.numel() else 0:+.3g}]')

    zerosV = torch.zeros_like(st.velocities)
    update = IncompressibleSystemUpdate(
        dxdt=zerosV.clone(), dvdt=zerosV.clone(),
        drhodt=torch.zeros_like(st.densities),
        passive=torch.zeros_like(st.densities, dtype=torch.bool))
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    return update, adjacency, st, ([], [err])
