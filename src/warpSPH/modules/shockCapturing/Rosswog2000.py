"""Rosswog, Davies, Thielemann & Piran (2000) divergence-source viscosity switch.

The Morris-Monaghan equation with the source limited so `alpha` cannot exceed `alpha_max`:

    d alpha / dt = -(alpha - alpha_min) / tau + max(-div v, 0) (alpha_max - alpha)

**Transcribed from the diffSPH implementation (`modules/switches/Rosswog2000.py`), not
checked against the original paper -- it is not in `literature/`.** The time scale is the
Morris-Monaghan one (`h / (C_1 c)`, see `MorrisMonaghan1997.py`; diffSPH's `h c / l` was
dimensionally wrong). This is NOT Rosswog (2020)'s entropy-based trigger, which is a
separate scheme (AV_PLAN Phase 2).
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
