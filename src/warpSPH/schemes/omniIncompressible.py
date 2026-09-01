"""Direct port of the omniSPH incompressible solver loop
(`~/dev/omniSPH/simulation/{SPH,fluidMechanics}.cpp`).

This is a clean transcription of the DFSPH-style two-solve step omniSPH runs
in `SPHSimulation::timestep`, kept deliberately literal: no free-surface
pressure gauge, no particle-deficiency guard, no damped warm start, no
surface/deficiency masking -- only the loop as omniSPH runs it.

omniSPH per-step (`timestep`):

  1. neighbourhood + summation density (boundary particles included)
  2. external forces -> `a`  (gravity only)
  3. `divergenceSolve()`  -- exactly 3 relaxed-Jacobi iterations of `A p = s_div`
     with `s_div` the predicted-velocity divergence; `a += a_p^div`
  4. `densitySolve()`     -- relaxed-Jacobi of `A p = s_rho`, min 3 / max 256
     iterations, warm-started from `0.5 * p_prior`; `a += a_p^rho`
  5. XSPH velocity filter (`XSPH` + `BXSPH`): isotropic smoothing of `v`
     toward the neighbourhood mean + a static-wall drag (`XSPH_FLUID` /
     `XSPH_BOUNDARY`; 0/0 = the faithful no-dissipation loop)
  6. integrate: `v += dt a`,  `v *= (1 - damping)`,  `x += dt v`

Both solves run on the SAME positions and the SAME neighbourhood (omniSPH does
not move `x` between them -- the only `x` update is the final symplectic step),
and every pressure acceleration is accumulated into one `a` that a single
semi-implicit Euler step consumes. That accumulate-then-integrate structure,
and the two-solve order with omniSPH's own relaxation (`omega = 0.5`) and
iteration budgets, is what this module reproduces. Contrast `dfsphReference`
(SPlisHSPlasH order: an `x` advance and a neighbourhood rebuild between the
two solves, `v += dt a_p` applied per solve) and `iisph` (constant-density
solve only).

Operator mapping (omniSPH C++  ->  warpSPH composed operator):

  * `computeAlpha`      -> `computeAlpha(...)` with the IISPH `a_ii` bracket,
    `includeBoundaryReaction=False` (a static particle takes no reaction, so
    it enters only the vector-sum term) and `apparentVolumes = m/rho`. omniSPH
    folds `-dt**2` into `fluidAlpha`; here `alpha = dt*dt * computeAlpha(...)`
    (`computeAlpha` already returns the negated bracket, so `alpha <= 0`).
  * `sum_j V_j (X_i - X_j) . gradW`  -> `-_divergence(X)`, where `_divergence`
    is the scatter / difference-form `WarpOperation.Divergence` (the same
    operator `computeMomentumIncompressible` wraps: for a `div = +1` field it
    returns `+1` in the bulk). So omniSPH's source term
    `s = (rho_target - rho) - dt sum_j V_j (v_i^* - v_j^*) . gradW`
    becomes `s = (rho_target - rho) + dt * _divergence(v^*)`, and the
    Laplacian probe `dt**2 sum_j V_j (a_i - a_j) . gradW` becomes
    `A p = -dt**2 * _divergence(a_p(p))`.
  * `computeAcceleration` (symmetric `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2)
    gradW`)  -> `computePressureAccelIISPH`, masked to the fluid rows (a
    static particle takes no reaction).

Boundary: `kind == 1` particles enter both solves as "static fluid at rho0"
with Akinci rest-density volumes via `applyConsistentCoupling`
(`BoundaryPressureMode.consistent`) -- the particle-boundary analogue of
omniSPH's triangle `boundaryFunc`, and the same coupling `iisph` /
`dfsphReference` use. On top of that, `WALL_PRESSURE_MODE` (default `'mls'`
since Part 41) ports omniSPH's per-iterate MLS boundary-pressure
extrapolation into the *density* solve: every density-mode Jacobi iterate
recomputes `p_b` on the `kind == 1` rows from the current fluid pressure and
feeds its gradient into `a_p`, the Robin closure that makes the near-wall
iteration contract (without it `hydrostaticColumn` nx=128 diverges at the
bottom corners by step ~10 -- Part 35/41). `None` falls back to
Bender-Westhofen-Jeske 2023 Eq. 33 at zero boundary pressure. The divergence
solve always runs at zero boundary pressure (omniSPH's `divergenceSolve` has
no wall-pressure term). See `WALL_PRESSURE_MODE` below.

The step does its own integration and hands the integrator a no-op update;
`DFSPHReferenceSystem` (systems/incompressible.py) copies the advanced fields
across in `finalize`.
"""

from __future__ import annotations

from typing import Any

import torch

from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, buildVerletList,
                         warpOperation)

from ..configurations import BoundaryPressureMode
from ..modules.boundaryConditions import (computeForcing, enforceDirichlet,
                                          enforceUpdates)
from ..modules.density import computeDensities
from ..modules.gravity import computeGravity
from ..modules.incompressible.consistent import applyConsistentCoupling
from ..modules.incompressible.wallPressure import wallPressureExtrapolation
from ..modules.incompressible.wp_alpha import computeAlpha
from ..modules.pressure.iisph import computePressureAccelIISPH
from ..systems.incompressible import IncompressibleSystemUpdate

__all__ = ['omniIncompressible_step']

#: Relaxed-Jacobi relaxation. omniSPH's `updatePressure` hardcodes
#: `scalar omega = 0.5`, but on this codebase's *composed* pressure operator
#: the fixed-point Jacobi stability window is ~[0.2, 0.35] (measured -- see
#: `dfsphReference` Part 29 / `probe_dfsphReferenceContraction.py`), so the
#: faithful 0.5 detonates `hydrostaticColumn`'s density solve by t ~ 0.06.
#: 0.3 is the nearest stable value and holds the column bounded for 2000
#: steps. Set to 0.5 for the byte-faithful omniSPH iteration.
OMEGA = 0.3
#: omniSPH `divergenceSolve`: `while (counter++ < 3 || (error > limit && counter < 3))`
#: -- i.e. always exactly 3 iterations.
DIVERGENCE_ITERATIONS = 3
#: omniSPH `densitySolve`: `while (counter++ < 3 || (error > limit && counter < 256))`.
DENSITY_MIN_ITERATIONS = 3
DENSITY_MAX_ITERATIONS = 256
#: omniSPH `updatePressure`: `fluidDpDt[i] = max(residual, -0.001) * fluidArea[i]`
#: and the convergence sum is `mean_i max(residual_i, -0.001)`.
RESIDUAL_FLOOR = -1e-3
#: omniSPH `updatePressure`: `abs(fluidAlpha[i]) < 1e-25` -> pressure zeroed.
ALPHA_FLOOR = 1e-25
#: omniSPH `Integrate`: `vi *= (1.0 - damping)` with `props.damping` (0 here).
DAMPING = 0.0

#: omniSPH's XSPH velocity filter (`SPHSimulation::XSPH` / `BXSPH`), the
#: scheme's only dissipation -- a post-solve, isotropic smoothing of `v`
#: toward the neighbourhood mean:
#:     v_i <- v_i + sum_j c_j V_j W_ij (v_j - v_i)
#: with `c_j = XSPH_FLUID` for fluid neighbours and `XSPH_BOUNDARY` for the
#: static wall (v_j = 0 there, so that term is a wall drag -- omniSPH's
#: `BXSPH`, minus the wall-normal projection). omniSPH runs this *after* the
#: pressure solves and *before* the symplectic step, on the start-of-step
#: velocity (the solves only touch `accel`), so the filtered `v` and the
#: pressure impulse `dt*accel` are independent additions -- matched here.
#: This is what damps the free-slip bulk slosh the pressure solve leaves
#: behind (`dfsphReference` Part 32); at 0.0/0.0 the step is the faithful
#: no-dissipation omniSPH loop. `XSPH_FLUID ~ 0.05` is a light smoothing;
#: omniSPH's own `2*W` weighting makes its coefficient ~2x smaller for the
#: same effect.
XSPH_FLUID = 0.05
XSPH_BOUNDARY = 0.0

#: Per-iterate wall-pressure closure (Part 41). omniSPH's `densitySolve`
#: recomputes an MLS wall pressure `p_b` from the CURRENT fluid pressure every
#: Jacobi iterate and feeds its gradient into `a_p` / the Laplacian probe /
#: `alpha` -- a Robin closure that makes the near-wall iteration contract.
#: This port had the wall in `alpha` only (the Akinci band) with boundary
#: `p == 0` (BWJ23 Eq. 33), so the near-wall row is inconsistent (`D` carries
#: a wall term the operator does not) and the Jacobi does not contract at the
#: floor band -- `hydrostaticColumn` nx=128 hits the 256-iter cap with
#: `errRho` rising and blows up at the bottom corners. When set, every
#: density-mode `_solve` iterate recomputes a boundary pressure on the
#: `kind == 1` rows from the current fluid `p` and passes `p_all` into
#: `_pressureAccel`, so the symmetric gradient picks up the wall term
#: `-sum_k m_k (p_i/rho_i^2 + p_b/rho0^2) gradW_ik` (the divergence still sees
#: `a_p == 0` on the masked boundary rows, which reproduces omniSPH's
#: `a_i . gk` self-term).
#:
#:   None       -- off (faithful BWJ23 Eq. 33, boundary p == 0)
#:   'shepard'  -- zero-order mirror `p_b[k] = sum_f V_f p_f W_kf / sum_f V_f
#:                 W_kf` (omniSPH's MLS `alpha` term, no linear correction)
#:   'mls'      -- the full first-order Liu-Liu MLS extrapolation
#:                 (`modules/liu`, the same fit `computeMdbcPressure` uses):
#:                 evaluate the value+gradient fit at each ghost point and
#:                 Taylor-correct to the owning boundary particle. This is
#:                 omniSPH's `alpha + beta*x_b + gamma*y_b` in full.
#:
#: Default `'mls'` (Part 41): without it the composed density Jacobi does not
#: contract at the wall band and `hydrostaticColumn` nx=128 diverges at the
#: bottom corners by step ~10; with it the column holds 400+ steps at the
#: exact hydrostatic gradient. No relaxation / cross-step lag (unlike
#: `computeMdbcPressure`), so not the `mdbcMlsPressure` feedback instability.
WALL_PRESSURE_MODE = 'mls'


def _rebuildAdjacency(state: Any, system: Any, config: Any):
    adjacency = buildVerletList(
        state, config.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency, verbose=False)
    system.adjacency = adjacency
    return adjacency


def _divergence(state: Any, config: Any, adjacency: Any,
                field: torch.Tensor) -> torch.Tensor:
    """Scatter / difference-form SPH divergence -- the operator
    `computeMomentumIncompressible` wraps (`WarpOperation.Divergence`,
    `GradientScheme.Difference`, `SupportScheme.Scatter`), without its
    `-rho0` factor. `sum_j (m_j/rho_j) (field_j - field_i) . gradW_ij`; for a
    `div = +1` field it returns `+1` in the bulk. omniSPH's per-particle sum
    `sum_j V_j (X_i - X_j) . gradW` is the negative of this."""
    return warpOperation(
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


def _xsphFilter(state: Any, config: Any, adjacency: Any,
                fluidMask: torch.Tensor) -> torch.Tensor:
    """omniSPH's XSPH velocity filter (`SPHSimulation::XSPH` + `BXSPH`), as the
    per-fluid-row velocity increment `sum_j c_j V_j W_ij (v_j - v_i)`.

    `WarpOperation.Interpolate` computes `sum_j (m_j/rho_j) g_j W_ij` over all
    non-ghost neighbours, so folding the per-kind coefficient `c` into the
    reference values collapses the two omniSPH sums (fluid XSPH + wall drag)
    into `interp(c v) - v_i interp(c)`. The static wall enters with `v_j = 0`
    and `c_j = XSPH_BOUNDARY`, so its contribution `-c_k V_k W_ik v_i` is a
    drag; the output is masked back to the fluid rows (the wall takes no
    reaction, like the rest of this scheme)."""
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


def _pressureAccel(state: Any, config: Any, adjacency: Any,
                   p: torch.Tensor, fluidMask: torch.Tensor) -> torch.Tensor:
    """omniSPH `computeAcceleration`: the symmetric SPH pressure gradient
    `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) gradW`, as an acceleration. Zeroed
    on non-fluid rows -- a static particle takes no reaction
    (`operatorMovesBoundary=False`), and ghosts are not real particles."""
    a_p = computePressureAccelIISPH(
        state=state, pressureValues=p, config=config,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    return torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))


def _solve(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *,
           fluid: torch.Tensor, rho0: float, vEnter: torch.Tensor,
           warmStart: torch.Tensor, dt: float, mode: str,
           minIters: int, maxIters: int, tol: float):
    """One omniSPH pressure solve (`divergenceSolve` / `densitySolve` inner
    loop, minus the boundary-triangle passes).

    `A p = s` by relaxed Jacobi, `s` fixed from `vEnter` (the velocity after
    the accelerations accumulated so far):

      mode == 'divergence':  s = dt * _divergence(vEnter)
      mode == 'density':     s = (1 - rho/rho0) + dt * _divergence(vEnter)

    Each iteration (omniSPH `computeAcceleration` + `updatePressure`):
      a_p   = _pressureAccel(p)
      Ap    = -dt**2 * _divergence(a_p)                      # the Laplacian probe
      p    <- p + (omega / alpha) * (s - Ap)                 # alpha <= 0
      p    <- max(p, 0)   only for mode == 'density'
    `alpha` is omniSPH's `fluidAlpha` = `dt*dt * computeAlpha(...)`. Rows with
    `|alpha| < 1e-25`, non-finite `p`, or `|p| > 1e25` are zeroed
    (omniSPH: `pressure = residual = 0`). Convergence:
    `mean_fluid(max(Ap - s, -1e-3)) <= tol` after at least `minIters`.
    """
    with applyConsistentCoupling(state, config, schemeConfig, adjacency,
                                 BoundaryPressureMode.consistent):
        apparent = state.masses / state.densities
        # omniSPH folds -dt**2 into fluidAlpha; computeAlpha returns the
        # negated IISPH a_ii bracket, so `alpha <= 0` as omniSPH's is.
        alpha = dt * dt * computeAlpha(
            state, config, schemeConfig, adjacency,
            apparentVolumes=apparent, includeBoundaryReaction=False)
        invAlpha = OMEGA / alpha
        alphaBad = alpha.abs() < ALPHA_FLOOR

        divEnter = _divergence(state, config, adjacency, vEnter)
        if mode == 'density':
            source = (1.0 - state.densities / rho0) + dt * divEnter
        else:
            source = dt * divEnter

        wallP = WALL_PRESSURE_MODE if mode == 'density' else None

        def accel(pt):
            pin = wallPressureExtrapolation(
                state, config, adjacency, pt, fluid, mode=wallP) \
                if wallP else pt
            return _pressureAccel(state, config, adjacency, pin, fluid)

        p = torch.where(fluid, warmStart, torch.zeros_like(warmStart))
        err = 0.0
        it = 0
        for it in range(maxIters):
            a_p = accel(p)
            aP = -dt * dt * _divergence(state, config, adjacency, a_p)
            p = p + invAlpha * (source - aP)
            if mode == 'density':
                p = p.clamp(min=0.0)
            bad = alphaBad | (~torch.isfinite(p)) | (p.abs() > 1e25) | ~fluid
            p = torch.where(bad, torch.zeros_like(p), p)
            if fluid.any():
                residual = aP - source
                err = float(torch.clamp(residual, min=RESIDUAL_FLOOR)[fluid].mean())
            if it + 1 >= minIters and err <= tol:
                break
        a_p = accel(p)
    return a_p, p, it + 1, err


def omniIncompressible_step(system: Any, dt: float, config: Any,
                            schemeConfig: Any, verbose: bool = False):
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity

    solver = schemeConfig.solverConfig
    divCfg = solver.divergenceFreeSolver   # omniSPH dfsph.divergenceEta
    denCfg = solver.pressureSolver          # omniSPH dfsph.densityEta
    # Boundary particles enter both solves as static fluid at rho0 with Akinci
    # rest-density volumes -- the particle-boundary analogue of omniSPH's
    # triangle boundaryFunc (and what iisph / dfsphReference use).
    solver.akinciBoundaryVolume = True

    # --- 1. neighbourhood + summation density (boundary included) ----------
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)

    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    pPrior = st.pressures.clone()          # omniSPH fluidPriorPressure

    # --- 2. external forces -> a  (omniSPH externalForces: gravity only) ---
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    accel = computeGravity(st, config, schemeConfig, adjacency)
    forcing = computeForcing(system, dt, system.t, config, schemeConfig)
    accel = accel + forcing / st.masses.view(-1, 1)
    accel = torch.where(fcol, accel, torch.zeros_like(accel))

    # --- 3. divergenceSolve() -- exactly 3 iterations, zero warm start ----
    vEnter = st.velocities + dt * accel
    a_p_div, _, nDiv, errDiv = _solve(
        st, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=torch.zeros_like(st.densities), dt=dt,
        mode='divergence', minIters=DIVERGENCE_ITERATIONS,
        maxIters=DIVERGENCE_ITERATIONS, tol=divCfg.tolerance)
    accel = accel + a_p_div

    # --- 4. densitySolve() -- min 3 / max 256, warm start 0.5 * p_prior ---
    vEnter = st.velocities + dt * accel
    a_p_rho, pRho, nRho, errRho = _solve(
        st, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=0.5 * pPrior, dt=dt, mode='density',
        minIters=DENSITY_MIN_ITERATIONS, maxIters=DENSITY_MAX_ITERATIONS,
        tol=denCfg.tolerance)
    accel = accel + a_p_rho

    # --- 5. XSPH velocity filter (omniSPH XSPH + BXSPH, post-solve) --------
    # omniSPH filters the start-of-step velocity (the solves only touched
    # `accel`), so this and the pressure impulse `dt*accel` add independently.
    if XSPH_FLUID != 0.0 or XSPH_BOUNDARY != 0.0:
        st.velocities = st.velocities + _xsphFilter(st, config, adjacency, fluid)

    # --- 6. integrate (omniSPH Integrate: single semi-implicit Euler) -----
    st.velocities = st.velocities + dt * torch.where(
        fcol, accel, torch.zeros_like(accel))
    if DAMPING != 0.0:
        st.velocities = st.velocities * (1.0 - DAMPING)
    st.positions = st.positions + dt * torch.where(
        fcol, st.velocities, torch.zeros_like(st.velocities))

    # omniSPH: fluidPriorPressure = fluidPressure1 (the density solve's field).
    st.pressures = pRho

    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        pf = pRho[fluid]
        print(f'[omniIncompressible] t={system.t + dt:.4g}  '
              f'DIV {nDiv:2d} it err {errDiv:.3g}   '
              f'RHO {nRho:3d} it err {errRho:.3g}   |v|max {vmax:.4g}   '
              f'p[{float(pf.min()):+.3g}, {float(pf.max()):+.3g}]')

    zerosV = torch.zeros_like(st.velocities)
    update = IncompressibleSystemUpdate(
        dxdt=zerosV.clone(), dvdt=zerosV.clone(),
        drhodt=torch.zeros_like(st.densities),
        passive=torch.zeros_like(st.densities, dtype=torch.bool))
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    return update, adjacency, st, ([], [errDiv, errRho])
