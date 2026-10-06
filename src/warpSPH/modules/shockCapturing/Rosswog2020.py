"""Rosswog (2020) entropy-based dissipation trigger, ApJ 898, 60 (`rosswog2020entropy`), Eqs. (15)-(20):

    s_a      = P_a / rho_a^Gamma                                               (15)
    epsdot_a = |s^n_a - s^{n-1}_a| / s^{n-1}_a * tau_a / dt,   tau_a = h_a / c_a  (16)
    d alpha_a / dt = -(alpha_a - alpha_0) / (30 tau_a)                         (17)
    alpha_des = alpha_max S(x),  S(x) = 6x^5 - 15x^4 + 10x^3,                  (18)-(19)
    x = clamp((log epsdot - log eps_0) / (log eps_1 - log eps_0), 0, 1)        (20)

with the paper's `eps_0 = 1e-4`, `eps_1 = 5e-2`, `alpha_max = 1`, `alpha_0 = 0` and the decay time
`30 tau` (`entropy_eps0`, `entropy_eps1`, `entropy_decay`; `alpha_0` is the config's `alpha_min`).
"Instant up, exponential down": each step alpha decays and is raised to `alpha_des` if that is
larger (the paper borrows this from Cullen & Dehnen 2010).

**Step boundaries, not RK stages.** Eq. (16) differences entropy between time steps. The only place
the entropy of the step-boundary state is known exactly is the RHS evaluation at stage 0 (the stage
that runs on `(x^n, u^n)`; later stages run on predictor states), so the trigger is evaluated once
per step in the system's `finalize`, from stage 0's `entropies` (`advanceRosswog2020`, called through
`wrapper.advanceViscositySwitchStep`). The stage-level terms function only passes the stored alpha
through, so alpha is constant across a step's stages. Consequence: alpha^n (from s^n and s^{n-1}) is
used from step n+1 on -- a one-step lag against the paper, which applies it at step n. The first step
only records s^0 and sets alpha to `alpha_0` (the case ICs start alpha at 1).

`h` follows the paper's convention, the kernel reaching out to 2h (its footnote 2), i.e. half the
support radius stored here; the thresholds `eps_0`, `eps_1` were calibrated with it. The decay is
taken exactly, `alpha_0 + (alpha - alpha_0) exp(-dt / (30 tau))`, stable for any `dt / tau`.
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

__all__ = ['computeRosswog2020Terms', 'computeRosswog2020Update', 'advanceRosswog2020',
           'smoothStep', 'entropyRate', 'desiredAlpha', 'decayAlpha']


def smoothStep(x: torch.Tensor) -> torch.Tensor:
    """Eq. (19), the quintic smoothstep on [0, 1]; x is clamped first (Eq. 20)."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def entropyRate(s: torch.Tensor, sPrev: torch.Tensor, tau: torch.Tensor, dt: float) -> torch.Tensor:
    """Eq. (16): relative entropy change over one step, in units of the dynamical time tau."""
    return (s - sPrev).abs() / sPrev.abs().clamp(min=torch.finfo(s.dtype).tiny) * tau / dt


def desiredAlpha(epsdot: torch.Tensor, eps0: float, eps1: float, alpha_max: float) -> torch.Tensor:
    """Eqs. (18)-(20). `log(0) = -inf` clamps to x = 0, so an exactly conserved entropy gives 0."""
    x = (torch.log(epsdot) - math.log(eps0)) / (math.log(eps1) - math.log(eps0))
    return alpha_max * smoothStep(torch.nan_to_num(x, nan=0.0, neginf=0.0, posinf=1.0))


def decayAlpha(alpha: torch.Tensor, alpha0: float, tau: torch.Tensor, dt: float, decay: float) -> torch.Tensor:
    """Eq. (17) integrated exactly over dt."""
    return alpha0 + (alpha - alpha0) * torch.exp(-dt / (decay * tau))


def _tau(state) -> torch.Tensor:
    # h = support / 2: the paper's kernels reach out to 2h (footnote 2)
    return 0.5 * state.supports / state.soundspeeds.clamp(min=torch.finfo(state.supports.dtype).tiny)


def advanceRosswog2020(state, stage0State, t: float, schemeConfig) -> None:
    """Step-boundary update, in place on the step's final `state`: `stage0State` is the RHS state
    of the step's first stage, whose `entropies` are s^n at time `t = t^n`."""
    cfg = schemeConfig.viscositySwitchParams
    s = stage0State.entropies.detach()
    if state.entropiesPrev is None or state.entropiesPrevTime is None:
        alpha = torch.full_like(s, cfg.alpha_min)
        state.entropyRates = torch.zeros_like(s)
    else:
        dt = t - state.entropiesPrevTime
        if dt <= 0:
            return
        tau = _tau(stage0State)
        epsdot = entropyRate(s, state.entropiesPrev, tau, dt)
        alpha = torch.maximum(desiredAlpha(epsdot, cfg.entropy_eps0, cfg.entropy_eps1, cfg.alpha_max),
                              decayAlpha(state.alpha0s, cfg.alpha_min, tau, dt, cfg.entropy_decay))
        state.entropyRates = epsdot
    state.alpha0s = alpha
    state.alphas = alpha.clone()
    state.entropiesPrev = s.clone()
    state.entropiesPrevTime = float(t)


def computeRosswog2020Terms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    # alpha is set at the step boundary (`advanceRosswog2020`); every stage uses it unchanged
    alphas = particleState.alpha0s
    return alphas, ViscositySwitchState(
        alpha0s=alphas, alphas=alphas, M=None, M_inv=None, div=None, ddivdt=None,
        Shear=None, Rot=None, R=None, Xi=None, v_sig=None)


def computeRosswog2020Update(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                             supportScheme=None, adjacency=None):
    return switchState.alpha0s, switchState
