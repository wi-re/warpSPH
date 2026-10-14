"""Colagrossi (2004) shear limiter, as a stand-alone viscosity factor.

`f_i = |div v| / (|div v| + ||S|| + eps c / h)` with `S` the trace-free shear (rate-of-strain)
tensor and `||S|| = sqrt(S : S)` -- the Balsara limiter with the vorticity replaced by the
shear, so pure compression gives 1 and pure shear 0. Like Balsara it is instantaneous
(`alpha_i = f_i`, `alpha0s` untouched). The diffSPH version this follows regularised with
`1e-14 h`, which has the wrong dimensions; `eps c / h` (`balsara_const`) is used instead.
Optional extra in AV_PLAN S3; not part of the bake-off.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from warpSPHCore import *
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...systems.compressibleMonaghan import CompressibleState
from .common import *
from .switchState import ViscositySwitchState

__all__ = ['computeColagrossiTerms', 'computeColagrossiUpdate']


def computeColagrossiTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    div, Shear, Rot = computeShearTensor(None, particleState, simulationConfig, schemeConfig,
                                         supportScheme, adjacency)
    shearNorm = torch.sqrt(torch.einsum('...ij,...ij->...', Shear, Shear))
    const = schemeConfig.viscositySwitchParams.balsara_const
    alphas = div.abs() / (div.abs() + shearNorm + const * particleState.soundspeeds / particleState.supports + 1e-14)
    return alphas, ViscositySwitchState(
        alpha0s=particleState.alpha0s, alphas=alphas, M=None, M_inv=None, div=div, ddivdt=None,
        Shear=Shear, Rot=Rot, R=None, Xi=None, v_sig=None)


def computeColagrossiUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                            supportScheme=None, adjacency=None):
    return particleState.alpha0s, switchState
