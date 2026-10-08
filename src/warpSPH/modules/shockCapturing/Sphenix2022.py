"""Sphenix (Borrow et al. 2022, MNRAS 511, 2367; `borrow2022`) artificial-viscosity switch, Eqs. (21)-(24):

    S_i        = h_i^2 max(-ddiv_i/dt, 0)   if div v_i <= 0,  0 otherwise            (21)
    ddiv_i/dt  = (div v_i(t + dt) - div v_i(t)) / dt                                (22)
    alpha_loc  = alpha_max S_i / (c_i^2 + S_i),   alpha_max = 2                       (23)
    alpha_i   <- alpha_loc                                  if alpha_i < alpha_loc
                 (alpha_i + alpha_loc dt/tau) / (1 + dt/tau) otherwise,  tau = gamma_K ell h / c   (24)

**Eq. (21)'s sign.** The PDF's typesetting reads `-h^2 max(ddiv/dt, 0)`, which would make S <= 0 and alpha_loc
<= 0; the text ("high in pre-shock regions", where div v is falling) and Eq. (23) need S >= 0, i.e. the minus
inside the max -- SWIFT's implementation. **Eq. (24) is implicit** -- the decay is stable for any dt / tau and
never undershoots zero (the paper's point, and AV_PLAN 5B's stability test); it is implemented as written, not as an
explicit relaxation.

**The h convention.** Sphenix's `h` is the smoothing length and `gamma_K` the kernel's cut-off-to-smoothing-length
ratio (Dehnen & Aly 2012). This repo stores the cut-off `H = gamma_K h`, and `sphKernelScale` *is* Dehnen & Aly's
gamma (cubic 3D 1.825742, quartic 3D 2.018932 -- the paper's own example --, Wendland C2 3D 1.936492). So
`tau = gamma_K ell h / c = ell H / c` (gamma_K cancels) and `h^2` in Eq. (21) is `(H / sphKernelScale)^2`.

**The Balsara factor is not here.** Eq. (19) puts it into the pair coefficient, `alpha_ij = (alpha_i + alpha_j)/4
(B_i + B_j)`, which is `DiffusionParameters.balsaraPairLimiter` (Pi scaled by Bbar_ij, `modules/dissipation`).
The paper's operator is Eqs. (15)-(17), `v_sig = c_i + c_j - beta_V mu`, beta_V = 3, the whole term scaled by
alpha_ij: the `Price2012` term with `C_l = 1`, `C_q = 3`, `BetaMode.Coupled` (scripts/av_report.py `sphenix`).

**Step boundaries.** Eq. (22) differences div v between kicks, and alpha "is only ever updated at the kick steps":
like `Rosswog2020`, alpha is set once per step in the system's `finalize` (`advanceSphenix2022`) from the stage-0
`divergence` (the RHS sets it to `-drho/dt / rho`, the field the Cullen-Dehnen path also reuses), and the
stage-level terms pass it through. Alpha^n is used from step n+1 on (one-step lag); the first step records div v
and starts alpha at `alpha_min` (default 0). No correction matrix, no shear tensor: the switch adds no neighbour
loop of its own (the paper's efficiency claim, AV_PLAN 5B).
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from warpSPHCore import *
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...systems.compressibleMonaghan import CompressibleState
from .switchState import ViscositySwitchState

__all__ = ['computeSphenix2022Terms', 'computeSphenix2022Update', 'advanceSphenix2022',
           'sphenixShockIndicator', 'sphenixLocalAlpha', 'sphenixAlphaUpdate']


def sphenixShockIndicator(div: torch.Tensor, ddivdt: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Eq. (21), `h` the smoothing length."""
    return torch.where(div <= 0, h * h * (-ddivdt).clamp(min=0.0), torch.zeros_like(div))


def sphenixLocalAlpha(S: torch.Tensor, c: torch.Tensor, alpha_max: float) -> torch.Tensor:
    """Eq. (23): 0 for S -> 0, alpha_max for S -> inf. Guarded for cold gas (c = 0, e.g. Sedov's ambient medium):
    S = 0 there is 0 / 0 (a clamped c still underflows c^2), and the formula's own limit is 0."""
    return torch.where(S > 0, alpha_max * S / (c * c + S).clamp(min=torch.finfo(S.dtype).tiny), torch.zeros_like(S))


def sphenixAlphaUpdate(alpha: torch.Tensor, alphaLoc: torch.Tensor, dt: float, tau: torch.Tensor) -> torch.Tensor:
    """Eq. (24): instant raise to alpha_loc, otherwise the implicit (unconditionally stable) decay towards it."""
    x = dt / tau
    return torch.where(alpha < alphaLoc, alphaLoc, (alpha + alphaLoc * x) / (1.0 + x))


def advanceSphenix2022(state, stage0State, t: float, schemeConfig, simulationConfig=None) -> None:
    """Step-boundary update, in place on the step's final `state`: `stage0State.divergence` is div v at `t = t^n`."""
    cfg = schemeConfig.viscositySwitchParams
    div = stage0State.divergence.detach()
    if state.divergencePrevStep is None or state.divergencePrevStepTime is None:
        alpha = torch.full_like(div, cfg.alpha_min)
    else:
        dt = t - state.divergencePrevStepTime
        if dt <= 0:
            return
        kernelScale = _kernelScale(stage0State, simulationConfig)
        c = stage0State.soundspeeds.clamp(min=torch.finfo(div.dtype).tiny)
        H = stage0State.supports
        S = sphenixShockIndicator(div, (div - state.divergencePrevStep) / dt, H / kernelScale)
        alphaLoc = sphenixLocalAlpha(S, c, cfg.alpha_max)
        tau = cfg.sphenix_ell * H / c
        alpha = sphenixAlphaUpdate(state.alpha0s, alphaLoc, dt, tau).clamp(min=cfg.alpha_min)
    state.alpha0s = alpha
    state.alphas = alpha.clone()
    state.divergencePrevStep = div.clone()
    state.divergencePrevStepTime = float(t)


def _kernelScale(state, simulationConfig) -> float:
    if simulationConfig is None:
        raise ValueError('Sphenix2022 needs the SimulationConfig (kernel, dim) for gamma_K')
    return float(sphKernelScale(simulationConfig.kernel.value, simulationConfig.dim))


def computeSphenix2022Terms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    # alpha is set at the step boundary (`advanceSphenix2022`); every stage uses it unchanged
    alphas = particleState.alpha0s
    return alphas, ViscositySwitchState(
        alpha0s=alphas, alphas=alphas, M=None, M_inv=None, div=None, ddivdt=None,
        Shear=None, Rot=None, R=None, Xi=None, v_sig=None)


def computeSphenix2022Update(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                             supportScheme=None, adjacency=None):
    return switchState.alpha0s, switchState
