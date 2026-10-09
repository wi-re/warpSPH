"""Wadsley, Keller & Quinn (2017) Gasoline2 shock detector (MNRAS 471, 2357; `wadsley2017`), Eqs. (21)-(29):

    grad P  = (gamma - 1) sum_j m_j u_j grad_i W                                     (21)
    n       = grad P / |grad P|                                                      (22)
    dv/dn   = n_a V_ab n_b                                                           (23)
    D       = 3/2 [dv/dn + 1/3 max(-div v, 0)]                                       (24)
    alpha_loc = alpha_max A / (A + v_sig^2),  A = 2 h^2 xi max(-dD/dt, 0)              (25)-(26)
    dalpha/dt = (alpha_loc - alpha) / tau,  tau = h / (0.2 c);  instant raise          (27)
    xi = ((1 - R) / 2)^4,  R = sum_j m_j (D_j / |T|_j) W_R,ij / sum_j m_j W_R,ij        (28)-(29)
    T = (V + V^T) / 2  (keeps the trace),  W_R,ij = 1 - (r_ij / (2 h_i))^4,  alpha_max = 2

**The point** (AV_PLAN §1.5): `D` follows the velocity gradient *along the pressure gradient*, so it reaches
`div v` in a shock but vanishes in uniform compression, where every divergence-based detector (Cullen-Dehnen
included) switches on.

**The h convention -- AV_PLAN §6.2, derived before any tuning.** Gasoline2's `h` is "half the distance to the
furthest neighbour": its kernels reach out to `2h`. Cullen & Dehnen's `h` is the support itself (their footnote 2:
W = 0 for r > h), which is also what this repo stores (`supports`, call it H). So, for *every* kernel,

    H = h_CD = 2 h_G2,

independent of `sphKernelScale` (that is Dehnen & Aly's smoothing scale, a third convention: B7 2.33 / Wendland C2
1.90 in 2D -- using it here would be wrong). Eq. (26)'s factor 2 was Gasoline2's boost for "the different h
definition in CD" (a factor 4 in h^2) together with D being a 1D derivative; it is a Gasoline2-h constant. In this
repo's units

    A = 2 (H/2)^2 xi max(-dD/dt, 0) = 0.5 H^2 xi max(-dD/dt, 0)     (prefactor 0.5, not 2),
    tau = (H/2) / (0.2 c) = H / (0.4 c),    W_R = 1 - (r / H)^4.

`ViscositySwitchConfig.wadsley_prefactor` overrides the 0.5 (<= 0: derived); `wadsley_tau` is the 0.2.

**Dimension.** Eq. (24)'s 1/3 and 3/2 are 3D: blindness to uniform compression `V = -k I` needs the `1/d` and the
shock normalisation `D = div v` needs `d/(d-1)`. Here `D = d/(d-1) [dv/dn + max(-div v, 0)/d]`, the paper's form for
d = 3; in 1D every compression is planar and `D = div v` (no blindness to have).

**Estimators.** `V` is the volume-consistent `M^-1`-corrected difference gradient (`reconstruction.
computeVelocityJacobian`, exact for any linear field -- Gasoline2's Eq. (6) is exact for isotropic contraction only);
`grad P` the difference gradient of P (only its direction enters; exact zero for uniform P, where `n = 0` and
`dv/dn = 0`); `v_sig` the Cullen-Dehnen signal speed (`wp_vsig`). The R sum runs over the Verlet list with the
plain polynomial weight, deterministic (`segment_reduce`, no atomics).

**Step boundaries.** `dD/dt` is a previous-step finite difference, so like `Rosswog2020` / `Sphenix2022` alpha is
set once per step in the system's `finalize` (`advanceWadsley2017`) from the stage-0 detector values the stage
terms stored on the state (`wadsleyD`, `wadsleyXi`, `wadsleyVsig`); alpha^n is used from step n+1 on.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from warpSPHCore import *
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...systems.compressibleMonaghan import CompressibleState
from .switchState import ViscositySwitchState

__all__ = ['computeWadsley2017Terms', 'computeWadsley2017Update', 'advanceWadsley2017',
           'wadsleyD', 'wadsleyXi', 'wadsleyLocalAlpha', 'wadsleyDetector', 'WADSLEY_PREFACTOR']

#: Eq. (26)'s prefactor in this repo's h (= support) units: 2 (h_G2 / H)^2 = 2 / 4 (see the module docstring).
WADSLEY_PREFACTOR = 0.5


def wadsleyD(V: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Eq. (24), dimension-general: `d/(d-1) [n.V.n + max(-tr V, 0)/d]` (the paper's 3/2, 1/3 for d = 3); `tr V` in 1D."""
    dim = V.shape[-1]
    div = torch.einsum('...ii->...', V)
    if dim == 1:
        return div
    dvdn = torch.einsum('...a,...ab,...b->...', n, V, n)
    return dim / (dim - 1) * (dvdn + (-div).clamp(min=0.0) / dim)


def wadsleyXi(R: torch.Tensor) -> torch.Tensor:
    """Eq. (28), R clamped to [-1, 1] first (Eq. 29's caveat: D modifies dv/dn and noise pushes it out of range)."""
    return ((1.0 - R.clamp(-1.0, 1.0)) / 2.0) ** 4


def wadsleyLocalAlpha(A: torch.Tensor, vsig: torch.Tensor, alpha_max: float) -> torch.Tensor:
    """Eq. (25)."""
    return alpha_max * A / (A + vsig * vsig + torch.finfo(A.dtype).tiny)


def _minimumImage(dx: torch.Tensor, domain) -> torch.Tensor:
    L = (domain.max - domain.min).to(dx.dtype)
    periodic = domain.periodic.to(torch.bool)
    wrapped = dx - L * torch.round(dx / L)
    return torch.where(periodic, wrapped, dx)


def wadsleyDetector(particleState, simulationConfig, adjacency, gamma: float):
    """`(D, |T|, R, n)` for the current state (Eqs. 21-24, 29)."""
    from ..reconstruction import computeVelocityJacobian
    st = particleState
    V = computeVelocityJacobian(st, simulationConfig, SupportScheme.SuperSymmetric, adjacency, corrected=True)
    gradP = warpOperation(
        st, OperationProperties(kernel=simulationConfig.kernel, operation=WarpOperation.Gradient,
                                supportMode=SupportScheme.Gather, gradientMode=GradientScheme.Difference),
        domain=simulationConfig.domain, adjacency=adjacency, queryValues=st.pressures)
    gnorm = gradP.norm(dim=-1, keepdim=True)
    n = torch.where(gnorm > 0, gradP / gnorm.clamp(min=torch.finfo(gradP.dtype).tiny), torch.zeros_like(gradP))
    D = wadsleyD(V, n)
    T = 0.5 * (V + V.mT)
    Tnorm = torch.linalg.matrix_norm(T)              # Frobenius, trace kept
    ratio = torch.where(Tnorm > 0, D / Tnorm.clamp(min=torch.finfo(D.dtype).tiny), torch.zeros_like(D))

    # Eq. (29): polynomial-weighted mean of D/|T| over the neighbours within i's support
    i, j = adjacency.i.to(torch.int64), adjacency.j.to(torch.int64)
    dx = _minimumImage(st.positions[i] - st.positions[j], simulationConfig.domain)
    q = dx.norm(dim=-1) / st.supports[i]
    w = torch.where(q < 1, 1.0 - q ** 4, torch.zeros_like(q)) * st.masses[j]
    counts = torch.bincount(i, minlength=st.positions.shape[0])
    num = torch.segment_reduce(w * ratio[j], 'sum', lengths=counts)
    den = torch.segment_reduce(w, 'sum', lengths=counts)
    R = torch.where(den > 0, num / den.clamp(min=torch.finfo(den.dtype).tiny), torch.zeros_like(den))
    return D, Tnorm, R.clamp(-1.0, 1.0), n


def advanceWadsley2017(state, stage0State, t: float, schemeConfig, simulationConfig=None) -> None:
    """Step-boundary update, in place on the step's final `state`, from the stage-0 detector values."""
    cfg = schemeConfig.viscositySwitchParams
    D = stage0State.wadsleyD
    if D is None:
        return
    D = D.detach()
    if state.wadsleyDPrev is None or state.wadsleyDPrevTime is None:
        alpha = torch.full_like(D, cfg.alpha_min)
    else:
        dt = t - state.wadsleyDPrevTime
        if dt <= 0:
            return
        H = stage0State.supports
        c = stage0State.soundspeeds.clamp(min=torch.finfo(D.dtype).tiny)
        prefactor = cfg.wadsley_prefactor if cfg.wadsley_prefactor > 0 else WADSLEY_PREFACTOR
        A = prefactor * H * H * stage0State.wadsleyXi * (-(D - state.wadsleyDPrev) / dt).clamp(min=0.0)
        alphaLoc = wadsleyLocalAlpha(A, stage0State.wadsleyVsig, cfg.alpha_max)
        tau = 0.5 * H / (cfg.wadsley_tau * c)          # h_G2 / (0.2 c), h_G2 = H / 2
        decayed = alphaLoc + (state.alpha0s - alphaLoc) * torch.exp(-dt / tau)   # Eq. (27) integrated exactly
        alpha = torch.where(state.alpha0s < alphaLoc, alphaLoc, decayed).clamp(min=cfg.alpha_min, max=cfg.alpha_max)
    state.alpha0s = alpha
    state.alphas = alpha.clone()
    state.wadsleyDPrev = D.clone()
    state.wadsleyDPrevTime = float(t)


def computeWadsley2017Terms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    """Per stage: evaluate the detector (D, xi, v_sig) on this stage's state and store it for the step-boundary
    update; alpha itself is passed through unchanged (set in `advanceWadsley2017`)."""
    from .CullenDehnen2010 import compute_vsig
    D, Tnorm, R, _ = wadsleyDetector(particleState, simulationConfig, adjacency, schemeConfig.gamma)
    particleState.wadsleyD = D.detach()
    particleState.wadsleyXi = wadsleyXi(R).detach()
    particleState.wadsleyVsig = compute_vsig(particleState, simulationConfig, schemeConfig,
                                             supportScheme, adjacency).detach()
    alphas = particleState.alpha0s
    return alphas, ViscositySwitchState(
        alpha0s=alphas, alphas=alphas, M=None, M_inv=None, div=None, ddivdt=None,
        Shear=None, Rot=None, R=R, Xi=particleState.wadsleyXi, v_sig=particleState.wadsleyVsig)


def computeWadsley2017Update(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                             supportScheme=None, adjacency=None):
    return switchState.alpha0s, switchState
