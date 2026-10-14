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

The two pair loops (the relaxation signal velocity and the entropy dissipation) run in warp kernels
(``wp_readHayfield.py``) with the core minimum-image distance and kernel-gradient functions; an earlier
host (torch) version took raw ``x_i - x_j`` differences, wrong across a periodic boundary, and carried
its own B7 / Wendland2 ``dW/dr`` that silently fell back to Wendland2 for other kernels.

The entropy rate ``A_dot_diss`` is converted to an internal-energy rate via
the ideal-gas relation ``u = A rho^(gamma-1)/(gamma-1)``, holding the
density-dependent (adiabatic) part -- already present in the Monaghan ``dudt``
-- fixed: ``dudt_diss = rho^(gamma-1)/(gamma-1) * A_dot_diss``.
"""

import torch

from .common import *
from .Balsara1995 import balsaraFactor
from .wp_readHayfield import computeReadHayfieldVmaxWarp, computeReadHayfieldEntropyDissipationWarp
from .switchState import *
from warpSPHCore import *
from ...systems.compressibleMonaghan import CompressibleState
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from typing import Optional, Union

__all__ = ['computeReadHayfieldTerms', 'computeReadHayfieldUpdate']


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
    f_balsara = balsaraFactor(div, curl_mag, c, h, balsara_const)
    alpha_loc = (alpha_loc * f_balsara).clamp(min=alpha_min, max=alpha_max)

    # --- signal velocity for the relaxation (eqs. 24-25), on the device (minimum image) ---------- #
    v_max = computeReadHayfieldVmaxWarp(
        particleState, op(WarpOperation.Divergence), simulationConfig.domain, adjacency=adjacency)
    v_max = torch.clamp(v_max, min=1e-6)
    tau = h / v_max

    # --- relaxation (eqs. 22-23) ------------------------------------------- #
    alpha0s = particleState.alpha0s.clone()
    # instantaneous raise (eq. 22)
    alpha0s = torch.where(alpha_loc > alpha0s, alpha_loc, alpha0s)
    # decaying approach toward max(alpha_loc, alpha_min) (eq. 23)
    target = torch.maximum(alpha_loc, torch.full_like(alpha_loc, alpha_min))
    alpha0s = (alpha0s + (target - alpha0s) / tau * dt).clamp(min=alpha_min, max=alpha_max)

    # --- entropy dissipation (eqs. 33-35), on the device ----------------------------------------- #
    A_dot_diss = computeReadHayfieldEntropyDissipationWarp(
        particleState, op(WarpOperation.Divergence), simulationConfig.domain, gamma,
        queryAlphas=alpha0s, adjacency=adjacency)

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
        v_sig=v_max,
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
        dudt_diss=switchState.dudt_diss,
    )
