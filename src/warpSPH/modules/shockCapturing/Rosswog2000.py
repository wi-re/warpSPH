"""Rosswog, Davies, Thielemann & Piran (2000) divergence-source viscosity switch, A&A 360, 171,
Appendix A, Eqs. (A.5)-(A.6):

    d alpha / dt = -(alpha - alpha_min) / tau + max(-div v, 0) (alpha_max - alpha),   tau = h / (eps c)

**Checked against the paper (2026-10-01, `rosswog2000`).** The source is Morris-Monaghan's
`max(-div v, 0)` with the headroom factor `(alpha_max - alpha)` that keeps `alpha` inside its
interval; `tau = h / (eps c)` with `eps = 0.2` is `morris_C1` here (`MorrisMonaghan1997.py`);
`div v` is the paper's SPH divergence (A.4). The paper's constants are `alpha_max = 1.5`,
`alpha_min = 0.05`, `eps = 0.2` -- the repo defaults differ, so set them to reproduce its tests.
Not implemented here, because they belong to the pair operator rather than the switch: the paper's
`beta = 2 alpha` (this repo's `BetaMode.Coupled` is `beta = alpha_bar C_q`, `C_q = 2` gives it) and the
Balsara factor inside `mu_ij` (A.1)-(A.2), which `balsaraFactor` computes. The `h` in `tau` is the
smoothing length, converted from the stored support radius as in `MorrisMonaghan1997.py`.
This is NOT Rosswog (2020)'s entropy-based trigger, which is a separate scheme (AV_PLAN Phase 2).
"""

from __future__ import annotations

from typing import Optional, Union

from warpSPHCore import *
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...systems.compressibleMonaghan import CompressibleState
from .common import *
from .MorrisMonaghan1997 import relaxedAlpha
from .switchState import ViscositySwitchState

__all__ = ['computeRosswog2000Terms', 'computeRosswog2000Update']


def computeRosswog2000Terms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    div = computeDivergence(particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    headroom = (schemeConfig.viscositySwitchParams.alpha_max - particleState.alpha0s).clamp(min=0)
    alphas = relaxedAlpha(particleState, div, dt, simulationConfig, schemeConfig, sourceScale=headroom)
    return alphas, ViscositySwitchState(
        alpha0s=alphas, alphas=alphas, M=None, M_inv=None, div=div, ddivdt=None,
        Shear=None, Rot=None, R=None, Xi=None, v_sig=None)


def computeRosswog2000Update(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                             supportScheme=None, adjacency=None):
    return switchState.alpha0s, switchState
