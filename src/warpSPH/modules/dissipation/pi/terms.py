"""The pairwise artificial-viscosity formulations, one function each, plus the dispatch on `ViscosityTerms`.

Every formulation returns `val = rho_j * K / rho * v_sig * scalingFactor`: the Pi term times `rho_j`, so that the
viscosity kernels (`wp_diffusion.py`, ...) can apply it with `m_j / rho_j` as the apparent volume and
`w = (v_ij . x_ij) / r` as the remaining factor -- the acceleration of i gets `(m_j / rho_j) * val * w * grad_i W`.
That is `-m_j Pi_ab grad_i W` in Monaghan's sign convention, which is what `tests/test_piFormulations.py` checks against the
papers. (diffSPH's Pi does not carry the `rho_j`.) A formulation owns its own choice of pair means or one-sided values
(`pick`), its mu form and its signal velocity -- adding one is one function and one line in `evaluateTerm`.

Two mu forms exist. The *raw* form `mu = w` is the signal-velocity family (Monaghan 1997, Price 2012 Eqs. 101/103,
Dukowicz); the *scaled* form `mu = h w / (xi r)` with `scalingFactor = h / (xi r)` is the Monaghan (1992) family
(Monaghan & Gingold 1983, Eq. (98) of Price 2012, Wadsley 2008, delta-SPH), whose Pi is `-nu (v.r) / r^2` -- hence the `1/r`.
"""

from warpSPHCore import *
import warp as wp
from ....configurations.moduleConfigurations.diffusionParameters import ViscosityTerms
from ....configurations.moduleConfigurations.diffusionParameters import DiffusionParameters
from .coefficients import switchedCoefficients
from .pair import PairData, pick
from ...riemann.solvers import riemannStarState

__all__ = ['evaluateTerm']


@wp.func
def _val(pair: PairData, K: scalar_t, rho: scalar_t, v_sig: scalar_t, scalingFactor: scalar_t):
    return pair.rho_j * K / rho * v_sig * scalingFactor


@wp.func
def _muRaw(pair: PairData):
    """mu = (u_ij . x_ij) / r : the approach speed along the line of centres (negative for an approaching pair)."""
    return pair.ux / (pair.r + scalar_t(1e-14) * pair.h_bar)


@wp.func
def _scalingFactor(pair: PairData):
    """h / (xi r): turns `w` into `h (v.r) / (xi r^2)`, the Monaghan (1992) mu, when multiplied with the kernel's `w`."""
    return pair.h_bar / pair.kernelXi / (pair.r + scalar_t(1e-14) * pair.h_bar)


@wp.func
def _muScaled(pair: PairData):
    return _muRaw(pair) * _scalingFactor(pair)


# --- the Monaghan (1992) mu family: Pi = (-alpha c mu + beta mu^2) / rho, mu = h (v.r) / (r^2 + eps h^2) -------------

@wp.func
def monaghanGingold1983(pair: PairData):
    # Monaghan (2005) Eqs. (8.3)-(8.4): Pi = -nu (v.r) / r^2, nu = alpha h_bar c_bar / rho_bar, alpha = C_l
    return _val(pair, scalar_t(1.0), pair.rho_bar, pair.C_l * pair.c_bar, _scalingFactor(pair))


@wp.func
def monaghan1992(pair: PairData):
    # Monaghan (2005) Eq. (8.10) / Price (2012) Eq. (98), alpha = C_l, beta = C_q. c is one-sided (c_i, or c_j for
    # `useJ`): the two-call schemes (CompSPH, CRK) average the two sides, which gives c_bar
    c = pick(pair.c_i, pair.c_j, pair.c_bar, False, pair.useJ)
    v_sig = pair.C_l * c - pair.C_q * _muScaled(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, _scalingFactor(pair))


@wp.func
def wadsley2008(pair: PairData):
    # Wadsley et al. (2017) Eqs. (17)-(18): Monaghan (1992)'s Pi with all pair means
    v_sig = pair.C_l * pair.c_bar - pair.C_q * _muScaled(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, _scalingFactor(pair))


@wp.func
def deltaSPH(pair: PairData):
    # Marrone et al. (2011) Eq. (5): alpha h c0 (rho0 / rho_i) pi_ij, pi_ij = (u_ij . x_ij) / r^2; rho_i (one-sided) in
    # place of the reference density's ratio
    rho = pick(pair.rho_i, pair.rho_j, pair.rho_bar, False, pair.useJ)
    v_sig = pair.C_l * pair.c_bar - pair.C_q * _muScaled(pair)
    return _val(pair, scalar_t(1.0), rho, v_sig, _scalingFactor(pair))


@wp.func
def defaultTerm(pair: PairData):
    # "Default to Monaghan1992": the same, with every quantity a pair mean
    v_sig = pair.C_l * pair.c_bar - pair.C_q * _muScaled(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, _scalingFactor(pair))


# --- the signal-velocity family: Pi = -K v_sig w / rho_bar ---------------------------------------------------------

@wp.func
def monaghan1997a(pair: PairData):
    # Monaghan (2005) Eqs. (8.11)-(8.12) with K = 1/2: K v_sig = c_bar - (beta / 2) w, so alpha = C_l, beta / 2 = C_q
    v_sig = pair.C_l * pair.c_bar - pair.C_q * _muRaw(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


@wp.func
def price2012Eq98Label(pair: PairData):
    # `Price2012_98`: Price (2012) Eqs. (101)+(103), not Eq. (98) -- same arithmetic as Monaghan1997a
    v_sig = pair.C_l * pair.c_bar - pair.C_q * _muRaw(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


@wp.func
def price2012(pair: PairData):
    # Price (2012) Eq. (103): v_sig = (c_a + c_b - beta w) / 2 with beta = C_q
    v_sig = pair.C_l * pair.c_bar - pair.C_q / scalar_t(2.0) * _muRaw(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


@wp.func
def monaghan1997b(pair: PairData):
    # Monaghan (1997) Eq. (4.7) as transcribed (no source on disk to check against). For C_q -> 0 it collapses to
    # c_i + c_j = 2 c_bar, i.e. C_l = 2, C_q = 1 of `Monaghan1997a`.
    mu = _muRaw(pair)
    v_sig = safe_sqrt(pair.c_i * pair.c_i + pair.C_q * mu * mu) + safe_sqrt(pair.c_j * pair.c_j + pair.C_q * mu * mu) - mu
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


@wp.func
def dukowicz(pair: PairData):
    # Monaghan (1997) Eq. (4.8) as transcribed (no source on disk): the 1997a term with a 3/4 on the quadratic part
    v_sig = pair.C_l * pair.c_bar - scalar_t(3.0)/scalar_t(4.0) * pair.C_q * _muRaw(pair)
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


# --- the others --------------------------------------------------------------------------------------------------

@wp.func
def cleary1998(pair: PairData):
    # Monaghan (2005) Eqs. (8.8)-(8.9): mu_a = alpha_a h_a c_a rho_a / (2 (d + 2)),
    # Pi = -19.8 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) (v.r) / r^2   (16 -> 19.8, Cleary & Ha 2002); all one-sided
    rho = pick(pair.rho_i, pair.rho_j, pair.rho_bar, False, pair.useJ)
    h = pick(pair.h_i, pair.h_j, pair.h_bar, False, pair.useJ)
    f = scalar_t(1.0)/(scalar_t(2.0)*(scalar_t(pair.dim)+scalar_t(2.0)))   # 1/8 in 2D, 1/10 in 3D; 1D not given
    # alpha_a is each particle's own switched alpha times the base coefficient (not the pair-mean `C_l`)
    mu_i = f * pair.alpha_i * pair.C_l_ * pair.h_i * pair.c_i * pair.rho_i / pair.kernelXi
    mu_j = f * pair.alpha_j * pair.C_l_ * pair.h_j * pair.c_j * pair.rho_j / pair.kernelXi
    # `_val` divides by `rho`, and Pi (8.9) has no 1/rho beyond the 1/(rho_a rho_b) inside: multiply it back in
    v_sig = scalar_t(19.8) * mu_i * mu_j / (pair.rho_i * pair.rho_j * (mu_i + mu_j)) / (pair.r + scalar_t(1e-14) * h) * rho
    return _val(pair, scalar_t(1.0), rho, v_sig, scalar_t(1.0))


@wp.func
def price2008(pair: PairData):
    # Price (2008) conductivity signal speed v_sig^u = sqrt(|P_a - P_b| / rho_bar); P = rho c^2 (ideal-gas shortcut)
    # unless pressures were passed explicitly
    P_i = pair.rho_i * pair.c_i * pair.c_i
    P_j = pair.rho_j * pair.c_j * pair.c_j
    if pair.explicitPressure:
        P_i = pair.P_i
        P_j = pair.P_j
    rho_bar = (pair.rho_i + pair.rho_j) / scalar_t(2.0)
    v_sig = pair.C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + scalar_t(1e-14) * pair.h_bar))
    return _val(pair, scalar_t(1.0), pair.rho_bar, v_sig, scalar_t(1.0))


@wp.func
def frontiere2017(pair: PairData, viscosityParams: DiffusionParameters):
    """CRKSPH's one-sided viscous pressure Q_i = rho_i (-C_l c_i mu + C_q mu^2) (Frontiere et al. 2017 Eq. (69)),
    mu = min(0, v.eta / (eta.eta + eps^2)), eta = x_ij / h_i, eps^2 = 1e-2 -- Monaghan's (1992) mu with the particle's
    own smoothing length `h_i / kernelScale`, the dimensionless regulariser (without it a near-coincident approaching pair
    has an unbounded mu: OPEN_PROBLEMS §15) and `min(0, .)` built in, whatever `monaghanSwitch` says. Everything is
    one-sided -- rho, c, h and the switched alpha (and beta, per `BetaMode`) -- so a caller wanting Q_j passes `useJ`.
    The value is in Pi's convention (the `rho_j` and the `w` that the kernel supplies), see `computeFrontiereQ`."""
    rho = pick(pair.rho_i, pair.rho_j, pair.rho_bar, False, pair.useJ)
    c = pick(pair.c_i, pair.c_j, pair.c_bar, False, pair.useJ)
    h = pick(pair.h_i, pair.h_j, pair.h_bar, False, pair.useJ) / pair.kernelScale
    alpha = pick(pair.alpha_i, pair.alpha_j, scalar_t(1.0)/scalar_t(2.0) * (pair.alpha_i + pair.alpha_j), False, pair.useJ)
    C_l, C_q = switchedCoefficients(alpha, pair.C_l_, pair.C_q_, viscosityParams)
    denom = pair.r * pair.r + scalar_t(1.0e-2) * h * h
    mu = h * pair.ux / denom
    val = scalar_t(0.0)
    if pair.ux <= scalar_t(0.0):
        # `scalingFactor * w == mu`: the same shape as the Monaghan (1992) family
        val = _val(pair, scalar_t(1.0), rho, C_l * c - C_q * mu, pair.r * h / denom)
    return val


# the adiabatic index used when the caller gives no pressures (`explicitPressure` False): P = rho c^2 / gamma
_GAMMA_NO_PRESSURE = 5.0 / 3.0


@wp.func
def riemannDissipation(pair: PairData, viscosityParams: DiffusionParameters):
    """AV_PLAN Phase 7b: the dissipative part of the pair's Godunov pressure. Left state j, right state i along the
    normal n = x_ij / r (pointing from j to i), with the (possibly reconstructed) pair velocity along n,
    `w = u_ij . n`, as the velocity jump (`uL = 0`, `uR = w`; the solvers are Galilean invariant). The Godunov-SPH
    momentum equation replaces `P_i/rho_i^2 + P_j/rho_j^2` by `p* (1/rho_i^2 + 1/rho_j^2)`; the part of that which exists
    because of the velocity jump,

        Pi_ij = (p*(w) - p*(0)) (1/rho_i^2 + 1/rho_j^2),

    is the pairwise viscous term: symmetric in i <-> j (momentum conserved, the usual Pi heating applies), zero for a
    pair at relative rest whatever the pressures (the conservative pressure gradient stays the symmetric SPH one) and
    zero for equal pressures and densities, positive for an approaching pair and negative for a receding one
    (`monaghanSwitch` removes the latter). For `Acoustic` with equal states it is `c |w| / rho`, Monaghan (1997)'s
    linear viscosity with alpha = 1; the shock solvers add the quadratic term. The pair-mean switched alpha scales it
    (1 without a switch). The adiabatic index is `rho c^2 / P` of the two particles (the ideal gas', exactly), or
    5/3 without explicit pressures. Returned in `Pi`'s convention (`rho_j Pi / |w|`, see `_val`)."""
    P_i = pair.P_i
    P_j = pair.P_j
    gamma = scalar_t(_GAMMA_NO_PRESSURE)
    if pair.explicitPressure:
        gamma = (pair.rho_i * pair.c_i * pair.c_i + pair.rho_j * pair.c_j * pair.c_j) / (P_i + P_j + scalar_t(1.0e-30))
        gamma = wp.min(wp.max(gamma, scalar_t(1.0001)), scalar_t(3.0))
    else:
        P_i = pair.rho_i * pair.c_i * pair.c_i / gamma
        P_j = pair.rho_j * pair.c_j * pair.c_j / gamma
    w = pair.ux / (pair.r + scalar_t(1e-14) * pair.h_bar)
    pStar, uStar = riemannStarState(viscosityParams.riemannSolver, pair.rho_j, scalar_t(0.0), P_j, pair.rho_i, w, P_i, gamma)
    pRest, uRest = riemannStarState(viscosityParams.riemannSolver, pair.rho_j, scalar_t(0.0), P_j, pair.rho_i, scalar_t(0.0), P_i, gamma)
    dp = pStar - pRest
    inv = scalar_t(1.0) / (pair.rho_i * pair.rho_i) + scalar_t(1.0) / (pair.rho_j * pair.rho_j)
    return pair.C_l * pair.rho_j * inv * dp / (wp.abs(w) + scalar_t(1.0e-12) * pair.c_bar + scalar_t(1.0e-20))


@wp.func
def evaluateTerm(viscosityTerm: wp.int32, pair: PairData, viscosityParams: DiffusionParameters):
    """`val` of the selected formulation (before the Monaghan switch); 0 for an unknown member."""
    val = scalar_t(0.0)
    if viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value):
        val = monaghanGingold1983(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value):
        val = cleary1998(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value):
        val = monaghan1992(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value):
        val = monaghan1997a(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value):
        val = monaghan1997b(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value):
        val = dukowicz(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value):
        val = price2012Eq98Label(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value):
        val = price2012(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value):
        val = price2008(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value):
        val = wadsley2008(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value):
        val = deltaSPH(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Default.value):
        val = defaultTerm(pair)
    elif viscosityTerm == wp.static(ViscosityTerms.Frontiere2017.value):
        val = frontiere2017(pair, viscosityParams)
    elif viscosityTerm == wp.static(ViscosityTerms.RiemannDissipation.value):
        val = riemannDissipation(pair, viscosityParams)
    return val
