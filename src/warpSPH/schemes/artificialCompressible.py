"""The artificial-compressibility SPH step (De Courcy et al. 2024; see
`ACSPH_PLAN.md` for the full equation inventory and this file's roadmap).

Each real step runs a **pseudo-time loop to steady state**, and the integrator
outside it has nothing left to do. Rather than teach `warpSPHIntegrators` a
`dualTime` scheme -- coupling a general library to one solver -- the step
returns an **exact delta**:

    dxdt = (x^{n+1} - x^n)/dt,  dvdt = (v^{n+1} - v^n)/dt,  dpdt = (p^{n+1} - p^n)/dt

Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte for byte with no framework change. That contract only
holds for a **one-stage, one-evaluation** integrator: under RK2 the solve would
run twice per step and the results would be *blended*, which is wrong and not
visibly wrong. `validateIntegrationScheme` therefore refuses anything else,
loudly, at step entry -- `cases/dambreak.py` documents the same class of trap
for `divergenceFree`/`semiImplicitEuler` and notes that nothing enforces it;
here it is enforced.

The loop (Eqs. 38-48)
---------------------
    for m in 0..maxPseudoIterations:
        u0 = u                                  # frozen for the BDF source
        D^p = pressureSmoothing(u0)             # frozen across RK stages
        for s in 1..rkStages:
            r  = spatial residual at u^{s-1}    # Eqs. 23, 25, 26
            r* = (r - I_c (alpha_t u0 + beta_t u^n + gamma_t u^{n-1})) / alpha_PI
            u^s = u0 + dtau sum_l a_{s,l} r*_l
        u = u0 + dtau sum_l b_l r*_l
        if eps_v(tilde v) < target: break
    roll the BDF history

`I_c = diag{0, 1, 1}` -- the continuity equation has no real-time derivative,
which is exactly what makes `r* -> 0` enforce `div v = 0` *at* time level n+1.

Three choices worth naming, all from `ACSPH_PLAN.md` Part 5:

- **The RK sweep is the general explicit Butcher form**, taken from
  `warpSPHIntegrators.getButcherTableau` ('midpoint' / 'SSPRK3' / 'RK4' are
  Fig. 1 of the paper verbatim). Eq. (40) as printed is the Jameson low-storage
  form, which can only represent a tableau whose `A` is sub-diagonal and whose
  `b` is its last stage row -- true of the RK2 midpoint tableau, false of both
  SSPRK3 and RK4 as Fig. 1 prints them. The general form reproduces Fig. 1
  exactly and degenerates to Eq. (40) for RK2, which is the recommended
  operating point anyway (Sec. 4.3: higher order buys no accuracy here, the
  BDF2 sets it). Plan Sec. 5.2; **ask the authors which the CUDA code does.**
- **`k2 D^p`, not `k2 h D^p`.** Eq. (30) prints the extra `h`; Eqs. (23), (51)
  and (54) do not, and dimensional analysis says they are right. Plan Sec. 5.3.
- **The diffusion is frozen across RK stages but re-evaluated every dual-time
  iteration**, which is what the paper says ("the diffusive terms are evaluated
  at each dual-time iteration and cannot be fixed without loss of stability").

Still to land (see `ACSPH_PLAN.md` Part 8)
------------------------------------------
step 6 the Eq. (46) timestep and its growth clamp; step 7 the Michel et al.
shifting and Eqs. (58)-(59); step 8 AC-4 and AC-JST; step 10 the `tilde v`
advective corrections (Eqs. 27-31), internal shifting, `k3`. Each is gated by
its config field and raises rather than silently no-opping.
"""

import copy
from typing import Any, Optional

import torch
from torch.profiler import record_function
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, buildVerletList,
                         sphKernel_xi, warpOperation)
from warpSPHIntegrators.butcher import getButcherTableau
from warpSPHIntegrators.integration import IntegrationSchemeType

from ..configurations import ArtificialCompressibleSPHConfig, SimulationConfig
from ..modules.artificialCompressible import computePressureSmoothing
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.deltaSPH import computeVelocityDiffusion
from ..modules.gravity import computeGravity
from ..modules.mdbc import computeBoundaryVelocities
from ..modules.pressure import computePressureForceSurfaceAware
from ..modules.surfaceDetection import detectFreeSurface
from ..systems.artificialCompressible import (ArtificialCompressibleSystem,
                                              ArtificialCompressibleSystemUpdate)

__all__ = ['artificialCompressible_step', 'validateIntegrationScheme',
           'acParameters', 'convergenceMetric', 'PHYSICS_IMPLEMENTED']

PHYSICS_IMPLEMENTED = True

#: The only integrators whose action on an exact delta is the identity.
_EXACT_DELTA_INTEGRATORS = {IntegrationSchemeType.forwardEuler,
                            IntegrationSchemeType.explicitEuler}

#: `rkStages` -> the tableau name in `warpSPHIntegrators.getButcherTableau`.
#: These three ARE Fig. 1 of the paper: explicit midpoint, SSPRK3 (Shu-Osher),
#: classical RK4.
_TABLEAUS = {2: 'midpoint', 3: 'SSPRK3', 4: 'RK4'}

#: `CFL_tau` the paper pairs with each stage count (Sec. 3.1.3). Not enforced --
#: `acParams.cflTau` is the knob -- but reported by `acParameters` so a
#: mismatch is visible rather than silent.
RECOMMENDED_CFL_TAU = {2: 0.5, 3: 1.0, 4: 1.5}


def validateIntegrationScheme(config: SimulationConfig) -> None:
    """Refuse any integrator that would evaluate the step more than once, or
    scale its result. See this module's docstring for why.

    Raised, not warned: a multi-stage integrator here does not fail visibly --
    it runs the whole dual-time solve twice and blends two converged states,
    producing a plausible-looking but wrong answer.
    """
    scheme = getattr(config, 'integrationScheme', None)
    if scheme in _EXACT_DELTA_INTEGRATORS:
        return
    name = getattr(scheme, 'name', scheme)
    raise ValueError(
        f"artificialCompressible requires integrationScheme=forwardEuler, got "
        f"{name!r}. The step returns an exact per-step delta (dx/dt = "
        f"(x^{{n+1}} - x^n)/dt), which only a single-evaluation integrator "
        f"applies unchanged; a multi-stage one would run the dual-time solve "
        f"once per stage and blend the results. Set "
        f"`config.integrationScheme = IntegrationSchemeType.forwardEuler` "
        f"(CaseSpec: `--integrationScheme forwardEuler`).")


def acParameters(currentState, config, schemeConfig, dt: float):
    """`(dtau, beta, k1, k2, nu)` for this step, Eq. (24).

        beta = CFL_tau h / dtau,   k1 = beta^2,   k2 = k2Factor h beta

    `beta` is the pseudo-time wave speed. Finite volumes prescribe it and let
    `dtau` vary locally; the paper inverts that -- `dtau = dt / dtOverDtau` is
    spatially constant (so particle displacements stay smooth) and `beta` is
    derived, per particle, from each particle's own `h`.

    `h` here is the paper's *smoothing length*, `supports / xi` -- this repo
    stores the kernel's support radius, and `modules/deltaSPH/densityDiffusion.py`
    applies the same `/xi` to the `delta h c_s` prefactor `k2` is modelled on.

    `nu` comes from `acParams.nu` unless `referenceSoundSpeedForViscosity` is
    set, in which case it is the paper's `nu = alpha_nu h c0 / K` -- see the
    config's docstring on why ACSPH still needs a reference `c0` for this and
    for nothing else.
    """
    acParams = schemeConfig.acParams
    xi = sphKernel_xi(config.kernel.value, config.dim)
    h = currentState.supports / xi

    dtau = dt / acParams.dtOverDtau
    beta = acParams.cflTau * h / dtau
    k1 = beta * beta
    k2 = acParams.k2Factor * h * beta

    K = 2 * (config.dim + 2)
    if acParams.referenceSoundSpeedForViscosity is not None:
        nu = acParams.alphaNu * h * acParams.referenceSoundSpeedForViscosity / K
    else:
        nu = torch.full_like(h, float(acParams.nu))
    return dtau, beta, k1, k2, nu


def convergenceMetric(tildeV: torch.Tensor, velocities: torch.Tensor,
                      fluid: torch.Tensor, schemeConfig) -> float:
    """`eps_v = log10( |tilde v|_2 / (N U_eps) )`, Eqs. (47)-(48), with

        U_eps = max( min(|v|_max, U_char), eps_s )

    Note the `1/N`, **not** `1/sqrt(N)`: this is not an RMS. A fixed `eps_v`
    target is therefore a *stricter per-particle* tolerance at higher
    resolution, by `-0.5 log10 N` -- about 0.6 of a decade across the paper's
    own `L/dx = 200 -> 800` sweep. Reproduced verbatim because it is what their
    numbers mean; recorded here because it is a real property of the metric,
    not of the scheme (`ACSPH_PLAN.md` Sec. 1.6).
    """
    n = int(fluid.sum())
    if n == 0:
        return float('-inf')
    acParams = schemeConfig.acParams
    vMax = float(velocities[fluid].norm(dim=-1).max()) if n else 0.0
    uChar = vMax if acParams.uChar is None else min(vMax, float(acParams.uChar))
    uEps = max(uChar, float(acParams.epsilonS))
    norm = float(tildeV[fluid].pow(2).sum().sqrt())
    if norm <= 0.0:
        return float('-inf')
    return float(torch.log10(torch.tensor(norm / (n * uEps))))


def _workingState(state, positions, velocities, pressures):
    """A shallow view of `state` with the three evolving fields replaced. The
    SPH operators read attributes off whatever they are handed, so this is all
    an intermediate RK stage needs -- and it keeps the real state untouched
    until the pseudo-time loop has converged."""
    view = copy.copy(state)
    view.positions = positions
    view.velocities = velocities
    view.pressures = pressures
    return view


def _spatialResidual(view, config, schemeConfig, adjacency, k1, k2, diffusion,
                     bodyForce, nu):
    """`(r_p, r_v)`: the right-hand sides of Eqs. (23) and (25) at `view`.

    Eq. (23):  Dp/Dtau = -k1 rho sum_j (v_j - v_i).gradW V_j + k2 D^p
    Eq. (25):  Dv/Dtau + Dv/Dt = -(1/rho) sum_j (p_i+p_j) gradW V_j
                                 + nu K sum_j (v_ij.x_ij)/|x_ij|^2 gradW V_j + f

    `diffusion` is `D^p`, passed in rather than computed here: it is frozen
    across the RK stages (Antuono/Jameson) and only re-evaluated once per
    dual-time iteration.
    """
    rho = view.densities

    divV = warpOperation(
        view,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.AllToAll,
                            gradientMode=GradientScheme.Difference),
        queryValues=view.velocities, domain=config.domain, adjacency=adjacency)
    rP = -k1 * rho * divV + k2 * diffusion

    # `computePressureSurfaceAwareWarp` returns `-sum_j V_j p_ij gradW`, which
    # is `rho * dv/dt`; Eq. (25) carries the explicit `1/rho_i`. delta-SPH
    # omits that division because it runs at `restDensity = 1`, where it is a
    # no-op; ACSPH does not assume that.
    rV = computePressureForceSurfaceAware(view, config, schemeConfig, adjacency) / rho.unsqueeze(-1)

    rV = rV + _viscosity(view, config, adjacency, nu)

    return rP, rV + bodyForce


class _ViscosityShim:
    """Adapter presenting `computeVelocityDiffusion`'s expected
    `schemeConfig.diffusionParams` / `.fluid` surface over ACSPH's config.
    ACSPH has no `diffusionParams` block -- its viscosity is one number derived
    from Eq. (25) -- so rather than bolt a delta-SPH block onto its config just
    to satisfy an accessor, the accessor is satisfied here."""

    class _Params:
        def __init__(self, nu):
            self.inviscid = False
            self.inviscidAlpha = 0.0
            self.viscidNu = nu

    class _Fluid:
        fixedSoundSpeed = 0.0

    def __init__(self, nu):
        self.diffusionParams = self._Params(nu)
        self.fluid = self._Fluid()


def _viscosity(view, config, adjacency, nu):
    """Eq. (25)'s viscous term,
    `nu K sum_j (v_ij . x_ij)/|x_ij|^2 gradW_ij V_j` with `K = 2(dim+2)`.

    Two adaptations of `computeVelocityDiffusion`, both exact rather than
    approximate:

    - it divides by `mean(rho_i, rho_j)` where Eq. (25) does not. Density is
      invariant here, so passing `nu * rho0` compensates exactly.
    - `approachOnly=False` lifts the artificial-viscosity clamp, which is what
      makes this the Monaghan-Gingold velocity Laplacian rather than a
      one-sided half of it (see `wp_viscosityDelta.py`'s docstring).

    `nu` is per-particle where the kernel takes a scalar; the kernel's `nu`
    enters linearly, so scaling the output by `nu_i / nu_mean` afterwards is
    exact for the `nu_i` factor. (`nu_j` does not appear -- Eq. 25 has a single
    `nu`.) The rescale is skipped entirely when `nu` is uniform, which it is
    unless it was derived from a varying `h`."""
    rho0 = float(view.densities[0]) if view.densities.numel() else 1.0
    nuScalar = float(nu.mean()) if isinstance(nu, torch.Tensor) else float(nu)
    out = computeVelocityDiffusion(view, config, _ViscosityShim(nuScalar * rho0),
                                   adjacency, approachOnly=False)
    if isinstance(nu, torch.Tensor) and float(nu.std()) > 0.0:
        out = out * (nu / nuScalar).unsqueeze(-1)
    return out


def artificialCompressible_step(
    system: ArtificialCompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: ArtificialCompressibleSPHConfig,
    verbose: bool = False,
):
    validateIntegrationScheme(config)
    acParams = schemeConfig.acParams

    if acParams.useTildeVAdvection:
        raise NotImplementedError(
            "acParams.useTildeVAdvection (Eqs. 27-31) is not implemented -- the "
            "paper's own conclusion is to leave it off (Sec. 4.2). "
            "ACSPH_PLAN.md step 10.")
    if acParams.shiftInsidePseudoLoop:
        raise NotImplementedError(
            "acParams.shiftInsidePseudoLoop (Eq. 60) is not implemented; Sec. 4.2 "
            "tested it and chose external shifting. ACSPH_PLAN.md step 10.")
    if acParams.k3 != 0.0:
        raise NotImplementedError(
            "acParams.k3 (Eqs. 9/22) is not implemented -- the paper zeroes it. "
            "ACSPH_PLAN.md step 10.")
    if acParams.rkStages not in _TABLEAUS:
        raise ValueError(f"acParams.rkStages must be one of {sorted(_TABLEAUS)}, "
                         f"got {acParams.rkStages}")

    currentSystem = system
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency

    # --- per-real-step setup ------------------------------------------------
    with record_function("[warpSPH] - [acsph - 01] - compute adjacency"):
        adjacency = buildVerletList(
            currentState, config.domain, verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=adjacency, verbose=False)
        currentSystem.adjacency = adjacency

    with record_function("[warpSPH] - [acsph - 02] - boundary velocities"):
        currentState.velocities = computeBoundaryVelocities(
            currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [acsph - 03] - enforce BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)

    with record_function("[warpSPH] - [acsph - 04] - surface detection"):
        # Frozen for the whole real step. The paper re-evaluates the *diffusion*
        # every dual-time iteration but says nothing about re-detecting the
        # surface; the surface set is a geometric property of a configuration
        # that moves by `tilde v -> 0` during the loop, and re-running Marrone
        # detection per iteration would dominate the cost.
        fs, fsm, n, renormalizationState, lMin = detectFreeSurface(
            currentState, config, schemeConfig, schemeConfig.surfaceDetectionConfig,
            adjacency, returnNormals=True)
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin

    dtau, beta, k1, k2, nu = acParameters(currentState, config, schemeConfig, dt)
    alphaT, betaT, gammaT, bdfOrder = currentSystem.bdfCoefficients(dt)
    tableau = getButcherTableau(_TABLEAUS[acParams.rkStages])
    fluid = currentState.kinds == 0

    x0 = currentState.positions
    v0 = currentState.velocities
    p0 = currentState.pressures
    xPrev = currentSystem.positionsPrev if currentSystem.positionsPrev is not None else x0
    vPrev = currentSystem.velocitiesPrev if currentSystem.velocitiesPrev is not None else v0
    xPrev2 = currentSystem.positionsPrev2 if currentSystem.positionsPrev2 is not None else xPrev
    vPrev2 = currentSystem.velocitiesPrev2 if currentSystem.velocitiesPrev2 is not None else vPrev

    with record_function("[warpSPH] - [acsph - 05] - forcing"):
        bodyForce = computeGravity(currentState, config, schemeConfig, adjacency) \
            + computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)

    # --- the dual-time loop -------------------------------------------------
    x, v, p = x0, v0, p0
    epsV = float('inf')
    iterations = 0
    with record_function("[warpSPH] - [acsph - 06] - dual-time loop"):
        for m in range(acParams.maxPseudoIterations):
            iterations = m + 1
            xStage0, vStage0, pStage0 = x, v, p

            # The BDF source is evaluated at the FROZEN stage-0 value, not at
            # the current stage (Eq. 41's `u^{n+1,m+1,0}`).
            dxdtBdf = alphaT * xStage0 + betaT * xPrev + gammaT * xPrev2
            dvdtBdf = alphaT * vStage0 + betaT * vPrev + gammaT * vPrev2

            stage0 = _workingState(currentState, xStage0, vStage0, pStage0)
            diffusion = computePressureSmoothing(
                stage0, config, schemeConfig, adjacency, renormalizationState,
                pressures=pStage0)

            kx, kv, kp = [], [], []
            for s in range(acParams.rkStages):
                xs, vs, ps = xStage0, vStage0, pStage0
                for l in range(s):
                    a = float(tableau.a[s, l])
                    if a == 0.0:
                        continue
                    xs = xs + dtau * a * kx[l]
                    vs = vs + dtau * a * kv[l]
                    ps = ps + dtau * a * kp[l]

                view = _workingState(currentState, xs, vs, ps)
                rP, rV = _spatialResidual(view, config, schemeConfig, adjacency,
                                          k1, k2, diffusion, bodyForce, nu)
                rX = vs

                # alpha_PI = 1 + alpha_s dtau alpha_t (Eqs. 43-45), applied to
                # all three rows for temporal consistency. `alpha_s` in Eq. (40)
                # is the fraction of `dtau` at which stage `s`'s residual is
                # applied; in Butcher terms that is the node of the stage it
                # produces, i.e. `c[s+1]`, and 1 for the last (which feeds the
                # `b` accumulation). For the RK2 midpoint tableau this is
                # exactly Eq. (40)'s `alpha = {1/2, 1}`. For RK3/RK4 the mapping
                # is ambiguous because Eq. (40) and Fig. 1 disagree there at all
                # (ACSPH_PLAN.md Sec. 5.2) -- and it barely matters, since the
                # paper reports `alpha_PI = 1` works fine here anyway
                # (`usePointImplicit=False`).
                alphaS = float(tableau.c[s + 1]) if s + 1 < acParams.rkStages else 1.0
                alphaPI = 1.0 + alphaS * dtau * alphaT if acParams.usePointImplicit else 1.0

                # I_c = diag{0, 1, 1}: the pressure row has no real-time
                # derivative to subtract. That is what makes r* -> 0 enforce
                # div v = 0 at time level n+1 rather than in pseudo-time.
                kp.append(rP / alphaPI)
                kx.append((rX - dxdtBdf) / alphaPI)
                kv.append((rV - dvdtBdf) / alphaPI)

            x, v, p = xStage0, vStage0, pStage0
            for l in range(acParams.rkStages):
                b = float(tableau.b[l])
                if b == 0.0:
                    continue
                x = x + dtau * b * kx[l]
                v = v + dtau * b * kv[l]
                p = p + dtau * b * kp[l]

            # `tilde v = v - Dx/Dt` is simultaneously the position-row residual
            # and the convergence metric (Eq. 26 / Sec. 1.6): it is zero exactly
            # when the position row is satisfied at time level n+1.
            tildeV = v - (alphaT * x + betaT * xPrev + gammaT * xPrev2)
            epsV = convergenceMetric(tildeV, v, fluid, schemeConfig)
            if m + 1 >= acParams.minPseudoIterations and epsV < acParams.epsilonV:
                break

    if verbose:
        print(f"[acsph] t={currentSystem.t:.6g} dt={dt:.4g} BDF{bdfOrder} "
              f"{iterations} pseudo-iterations, eps_v={epsV:.3f} "
              f"(target {acParams.epsilonV})")

    # --- exact-delta hand-off ----------------------------------------------
    with record_function("[warpSPH] - [acsph - 07] - build update"):
        update = ArtificialCompressibleSystemUpdate(
            dxdt=(x - x0) / dt,
            dvdt=(v - v0) / dt,
            dpdt=(p - p0) / dt,
            passive=torch.zeros(p0.shape, device=p0.device, dtype=torch.bool),
        )
        update.pseudoIterations = iterations
        update.epsilonV = epsV
        update.bdfOrder = bdfOrder

    with record_function("[warpSPH] - [acsph - 08] - enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config,
                       schemeConfig)
        nonFluid = (currentState.kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluid, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluid, torch.zeros_like(update.dvdt), update.dvdt)
        update.dpdt = torch.where(nonFluid.squeeze(-1), torch.zeros_like(update.dpdt),
                                  update.dpdt)

    return update, adjacency, currentState
