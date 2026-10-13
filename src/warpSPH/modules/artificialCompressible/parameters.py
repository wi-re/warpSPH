"""Per-step parameters and the convergence metric of the ACSPH dual-time loop
(De Courcy et al. 2024, Eqs. 24 and 46-48)."""

import torch
from warpSPHCore import sphKernel_xi
from warpSPHIntegrators.integration import IntegrationSchemeType


#: `rkStages` -> the tableau name in `warpSPHIntegrators.getButcherTableau`.
#: These three ARE Fig. 1 of the paper: explicit midpoint, SSPRK3 (Shu-Osher),
#: classical RK4.
TABLEAUS = {2: 'midpoint', 3: 'SSPRK3', 4: 'RK4'}

#: `CFL_tau` the paper pairs with each stage count (Sec. 3.1.3). Not enforced --
#: `acParams.cflTau` is the knob -- but reported so a mismatch is visible.
RECOMMENDED_CFL_TAU = {2: 0.5, 3: 1.0, 4: 1.5}

#: The only integrators whose action on an exact delta is the identity.
EXACT_DELTA_INTEGRATORS = {IntegrationSchemeType.forwardEuler,
                           IntegrationSchemeType.explicitEuler}


def validateIntegrationScheme(config) -> None:
    """Refuse any integrator that would evaluate the step more than once, or
    scale its result: the step returns an exact per-step delta, and a
    multi-stage integrator would run the dual-time solve once per stage and
    blend two converged states -- plausible-looking and wrong."""
    scheme = getattr(config, 'integrationScheme', None)
    if scheme in EXACT_DELTA_INTEGRATORS:
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


def validateAcParams(acParams) -> None:
    """Refuse options that are declared but not implemented, and inconsistent
    combinations, loudly at step entry instead of silently no-opping."""
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
    if acParams.rkStages not in TABLEAUS:
        raise ValueError(f"acParams.rkStages must be one of {sorted(TABLEAUS)}, "
                         f"got {acParams.rkStages}")
    if acParams.wallBounceTarget not in ('position', 'stage', 'both'):
        raise ValueError(f"acParams.wallBounceTarget must be 'position', 'stage' or 'both', "
                         f"got {acParams.wallBounceTarget!r}")
    if acParams.wallPressureEdgeRelax != 'off' and acParams.cavitationProjection != 'off':
        raise ValueError("wallPressureEdgeRelax is an alternative to cavitationProjection; "
                         "set one of them")


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


def convergenceMetricDevice(tildeV: torch.Tensor, velocities: torch.Tensor,
                            fluidIndex: torch.Tensor, n: int, schemeConfig) -> torch.Tensor:
    """`convergenceMetric` as a 0-d device tensor -- no host sync, so the
    dual-time loop reads it only at its check schedule's checkpoints.

    `fluidIndex` / `n` are the fluid rows and their count, computed once per
    step. Bitwise the same number: the host version does its arithmetic in
    Python doubles on float32-exact values and rounds the ratio to float32
    before the `log10`; this does exactly that on the device."""
    if n == 0:
        return torch.full((), float('-inf'), device=tildeV.device, dtype=tildeV.dtype)
    acParams = schemeConfig.acParams
    vMax = velocities.index_select(0, fluidIndex).norm(dim=-1).max().double()
    uChar = vMax if acParams.uChar is None else torch.clamp(vMax, max=float(acParams.uChar))
    uEps = torch.clamp(uChar, min=float(acParams.epsilonS))
    norm = tildeV.index_select(0, fluidIndex).pow(2).sum().sqrt().double()
    ratio = (norm / (n * uEps)).to(tildeV.dtype)
    # the host takes log10 on the CPU; the correctly rounded value (log10 in
    # float64 of the float32 ratio, rounded once) is what that returns --
    # CUDA's single-precision log10f can be one ulp off
    eps = torch.log10(ratio.double()).to(tildeV.dtype)
    return torch.where(norm <= 0.0, torch.full_like(eps, float('-inf')), eps)
