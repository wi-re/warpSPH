"""Read & Hayfield (2012) SPHS shock capturer (artificial conductivity +
entropy dissipation).

Faithful transcription of the ``rnh2012_sphs`` block of
``warpSPHCore/.../dehnen2012_convergence-without-pairing-instability/data/
da2012_reference.yaml`` (the Dehnen & Aly 2012 replication reference). The
method has three parts:

1. **Switch (eq. 21).** A local shock indicator driven by the *gradient of
   the velocity divergence*:
       alpha_loc,i = h_i^2 |grad(div v_i)| / (h_i^2 |grad(div v_i)| + h_i |div v_i| + ns c_s)
   for ``div v_i < 0`` (compression) and 0 otherwise. ``h`` is the code
   particle support (``particleState.supports``) which equals the paper's
   support radius H. The Balsara limiter (eq. 32),
   ``f = |div v| / (|div v| + |curl v| + balsara_const c_s / h)``, then
   multiplies ``alpha_loc`` to suppress it in shear/rotational flow.

2. **Relaxation (eqs. 22-25).** ``alpha0`` follows ``alpha_loc`` with an
   instantaneous raise and a decaying approach at rate ``1/tau`` where
   ``tau = h / v_max`` and ``v_max,i = max_j (c_i + c_j - 3 w_ij)``.

3. **Entropy dissipation (eqs. 33-35).** A pairwise entropy-smoothing source
   that drives the specific entropy ``A = P/rho^gamma`` across a pair toward
   one another, weighted by a pressure-difference limiter ``L_ij`` and the
   radial kernel gradient ``K_ij = r_hat_ij . grad_i W_ij = dW/dr < 0``.
   The negative ``K_ij`` is what makes the term a *diffusion* (it reduces
   ``A`` where ``A_i > A_j``). It is the term that suppresses the contact-
   discontinuity thermal-energy/pressure overshoot.

The artificial-viscosity *momentum* term (eqs. 29-31) is supplied by the
existing Monaghan viscosity in ``schemes/monaghan.py`` driven by this
switch's ``alpha`` (``queryAlphas``); the only R&H-specific addition to the
scheme is the entropy-dissipation internal-energy rate returned here as
``switchState.dudt_diss``.

The entropy rate ``A_dot_diss`` is converted to an internal-energy rate via
the ideal-gas relation ``u = A rho^(gamma-1)/(gamma-1)``, holding the
density-dependent (adiabatic) part -- already present in the Monaghan ``dudt``
-- fixed: ``dudt_diss = rho^(gamma-1)/(gamma-1) * A_dot_diss``.
"""

import torch
import numpy as np

from .common import *
from .switchState import *
from warpSPHCore import *
from ...systems.compressibleMonaghan import CompressibleState
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...math.scatter import scatter_sum
from typing import Optional, Union

__all__ = ['computeReadHayfieldTerms', 'computeReadHayfieldUpdate']


# --------------------------------------------------------------------------- #
# Kernel radial gradient  K_ij = r_hat_ij . grad_i W_ij = dW/dr  (torch).
#
# For W(r, h) = C_d / h^dim * k(q), q = r/h, the radial gradient is
# dW/dr = C_d / h^(dim+1) * dk/dq.  A compactly supported kernel is decreasing
# for q > 0, so dk/dq < 0 and hence K_ij = dW/dr < 0 -- the sign that makes the
# entropy dissipation a diffusion.
#
# Shape-function derivatives (matching warpSPHCore/kernels/kernelFunctions):
#   B7       k(q) = (1-q)_+^6 - 7(5/7-q)_+^6 + 21(3/7-q)_+^6 - 35(1/7-q)_+^6
#            dk/dq = -6(1-q)_+^5 + 42(5/7-q)_+^5 - 126(3/7-q)_+^5 + 210(1/7-q)_+^5
#   Wendland2 k(q) = (1-q)^4(1+4q)      (dim >= 2),  dk/dq = -20 q (1-q)^3
#            k(q) = (1-q)^3(1+3q)      (dim == 1),  dk/dq = -12 q (1-q)^2
# --------------------------------------------------------------------------- #
def _pos(x: torch.Tensor, p: int) -> torch.Tensor:
    """(x)_+^p = max(x, 0)^p."""
    return torch.clamp(x, min=0.0) ** p


def _b7_dkdq(q: torch.Tensor) -> torch.Tensor:
    return (
        -6.0 * _pos(1.0 - q, 5)
        + 42.0 * _pos(5.0 / 7.0 - q, 5)
        - 126.0 * _pos(3.0 / 7.0 - q, 5)
        + 210.0 * _pos(1.0 / 7.0 - q, 5)
    )


def _wendland2_dkdq(q: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 1:
        return -12.0 * q * _pos(1.0 - q, 2)
    return -20.0 * q * _pos(1.0 - q, 3)


# Normalisation constants C_d (matching warpSPHCore kernel definitions).
_B7_CD = {1: 823543.0 / 92160.0, 2: 5764801.0 / (113149.0 * np.pi), 3: 5764801.0 / (61440.0 * np.pi)}
_WENDLAND2_CD = {1: 1.25, 2: 7.0 / np.pi, 3: 21.0 / (2.0 * np.pi)}


def _kernel_dWdr(q: torch.Tensor, h: torch.Tensor, kernel_int: int, dim: int) -> torch.Tensor:
    """Radial kernel gradient dW/dr = C_d / h^(dim+1) * dk/dq(q)."""
    if kernel_int == 34:  # B7
        cd = _B7_CD[dim]
        dk_dq = _b7_dkdq(q)
    elif kernel_int == 0:  # Wendland2
        cd = _WENDLAND2_CD[dim]
        dk_dq = _wendland2_dkdq(q, dim)
    else:  # fallback: Wendland2-like (compact, decreasing)
        cd = _WENDLAND2_CD[dim]
        dk_dq = _wendland2_dkdq(q, dim)
    return cd / h ** (dim + 1) * dk_dq


def computeReadHayfieldTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):

    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    alpha_min = switchConfig.alpha_min
    alpha_max = switchConfig.alpha_max
    ns = switchConfig.ns
    balsara_const = switchConfig.balsara_const
    gamma = schemeConfig.gamma

    h = particleState.supports
    c = particleState.soundspeeds
    N = particleState.positions.shape[0]
    dim = particleState.positions.shape[1]
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    op = lambda operation: OperationProperties(
        kernel=simulationConfig.kernel,
        operation=operation,
        supportMode=supportMode,
        gradientMode=GradientScheme.Difference,
    )

    # --- velocity divergence, its gradient, and the vorticity magnitude ---- #
    div = warpOperation(
        particleState, op(WarpOperation.Divergence),
        domain=simulationConfig.domain, adjacency=adjacency,
        queryValues=particleState.velocities)
    grad_div = warpOperation(
        particleState, op(WarpOperation.Gradient),
        domain=simulationConfig.domain, adjacency=adjacency,
        queryValues=div)
    grad_div_mag = grad_div.norm(dim=-1)
    curl = warpOperation(
        particleState, op(WarpOperation.Curl),
        domain=simulationConfig.domain, adjacency=adjacency,
        queryValues=particleState.velocities)
    curl_mag = curl.norm(dim=-1)

    # --- local shock indicator (eq. 21) ------------------------------------ #
    h2_gdiv = h ** 2 * grad_div_mag
    denom = h2_gdiv + h * div.abs() + ns * c + 1e-14 * h
    alpha_loc = torch.where(div < 0, h2_gdiv / denom, torch.zeros_like(div))

    # --- Balsara limiter (eq. 32) ------------------------------------------ #
    f_balsara = div.abs() / (div.abs() + curl_mag + balsara_const * c / h + 1e-14)
    alpha_loc = (alpha_loc * f_balsara).clamp(min=alpha_min, max=alpha_max)

    # --- pairwise signal velocity for the relaxation (eqs. 24-25) ---------- #
    i = adjacency.i
    j = adjacency.j
    x_i = particleState.positions[i]
    x_j = particleState.positions[j]
    x_ij = x_i - x_j
    r_ij = x_ij.norm(dim=-1)
    r_ij_safe = torch.clamp(r_ij, min=1e-14)
    v_ij = particleState.velocities[i] - particleState.velocities[j]
    w_ij = (v_ij * x_ij).sum(dim=-1) / r_ij_safe
    v_sig = c[i] + c[j] - 3.0 * w_ij

    v_max = torch.full((N,), float('-inf'), device=device, dtype=dtype)
    v_max.scatter_reduce_(0, i, v_sig, reduce='amax', include_self=False)
    v_max = torch.where(torch.isinf(v_max), c, v_max)
    v_max = torch.clamp(v_max, min=1e-6)
    tau = h / v_max

    # --- relaxation (eqs. 22-23) ------------------------------------------- #
    alpha0s = particleState.alpha0s.clone()
    # instantaneous raise (eq. 22)
    alpha0s = torch.where(alpha_loc > alpha0s, alpha_loc, alpha0s)
    # decaying approach toward max(alpha_loc, alpha_min) (eq. 23)
    target = torch.maximum(alpha_loc, torch.full_like(alpha_loc, alpha_min))
    alpha0s = (alpha0s + (target - alpha0s) / tau * dt).clamp(min=alpha_min, max=alpha_max)

    # --- entropy dissipation (eqs. 33-35) ---------------------------------- #
    rho_i = particleState.densities[i]
    rho_j = particleState.densities[j]
    rho_ij = 0.5 * (rho_i + rho_j)
    alpha_ij = 0.5 * (alpha0s[i] + alpha0s[j])
    P_i = particleState.pressures[i]
    P_j = particleState.pressures[j]
    A_i = particleState.entropies[i]
    A_j = particleState.entropies[j]
    m_j = particleState.masses[j]

    # v_sig^p (eq. 34): positive signal velocity for approaching pairs
    v_sig_p = torch.where(3.0 * w_ij < c[i] + c[j], c[i] + c[j] - 3.0 * w_ij, torch.zeros_like(w_ij))
    # pressure-difference limiter (eq. 35)
    L_ij = (P_i - P_j).abs() / (P_i + P_j + 1e-14)
    # radial kernel gradient K_ij = dW/dr < 0 (eq. 35)
    q_ij = r_ij / h[i]
    K_ij = _kernel_dWdr(q_ij, h[i], simulationConfig.kernel.value, dim)

    rho_ratio = (rho_j / torch.clamp(rho_i, min=1e-14)) ** (gamma - 1.0)
    term = (m_j / rho_ij) * alpha_ij * v_sig_p * L_ij * (A_i - A_j) * rho_ratio * K_ij
    A_dot_diss = scatter_sum(term, i, dim=0, dim_size=N)

    # convert the specific-entropy rate to an internal-energy rate
    dudt_diss = (particleState.densities ** (gamma - 1.0) / (gamma - 1.0)) * A_dot_diss

    return alpha0s, ViscositySwitchState(
        alpha0s=alpha0s,
        alphas=alpha_loc,
        M=None,
        M_inv=None,
        div=div,
        ddivdt=None,
        Shear=None,
        Rot=None,
        R=None,
        Xi=None,
        v_sig=v_sig,
        dvdt_diss=None,
        dudt_diss=dudt_diss,
    )


def computeReadHayfieldUpdate(
        switchState: ViscositySwitchState,
        dt: float,
        dvdt: torch.Tensor,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    # The relaxation is performed in computeReadHayfieldTerms (the R&H
    # convention), so the update is a passthrough that carries the entropy-
    # dissipation rate forward.
    return switchState.alpha0s, ViscositySwitchState(
        alpha0s=switchState.alpha0s,
        alphas=switchState.alphas,
        M=None,
        M_inv=None,
        div=switchState.div,
        ddivdt=None,
        Shear=None,
        Rot=None,
        R=None,
        Xi=None,
        v_sig=switchState.v_sig,
        dvdt_diss=None,
        dudt_diss=switchState.dudt_diss,
    )
