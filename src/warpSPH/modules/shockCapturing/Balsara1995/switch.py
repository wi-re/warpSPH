"""Balsara (1995) shear limiter as a standalone factor and as a viscosity "switch".

`B_i = |div v| / (|div v| + |curl v| + eps c / h)`: 1 in pure compression, 0 in pure
rotation (Balsara 1995; Read & Hayfield 2012 Eq. 32; Garcia-Senz & Cabezon 2026 Eq. 8;
Sphenix Eq. 20 use the same form with `eps = 1e-4`). `balsaraFactor` is the reusable
multiplier -- Read-Hayfield applies it inside its indicator, AV_PLAN Phase 4 applies it to
the reconstruction and Sphenix to the pair coefficient. `computeBalsaraTerms` is the
classic stand-alone use: the pair viscosity parameter is `alpha_i = B_i` (averaged over the
pair by the operator), with no time evolution.

Unlike the time-dependent switches this is an *instantaneous* factor, so the stored
`alpha0s` passes through untouched.
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

__all__ = ['balsaraFactor', 'computeDivCurl', 'computeBalsaraTerms', 'computeBalsaraUpdate']


def balsaraFactor(div: torch.Tensor, curlMag: torch.Tensor, c: torch.Tensor, h: torch.Tensor,
                  const: float) -> torch.Tensor:
    """`|div| / (|div| + |curl| + const c / h + 1e-14)`; the `c / h` term keeps the
    dimensions consistent and the factor finite when both vanish."""
    return div.abs() / (div.abs() + curlMag + const * c / h + 1e-14)


def computeDivCurl(particleState: CompressibleState, simulationConfig: SimulationConfig,
                   supportScheme=None, adjacency=None):
    """`(div v, |curl v|)` from the difference-gradient operators."""
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    def op(operation):
        return OperationProperties(kernel=simulationConfig.kernel, operation=operation,
                                   supportMode=supportMode, gradientMode=GradientScheme.Difference)

    div = warpOperation(particleState, op(WarpOperation.Divergence), domain=simulationConfig.domain,
                        adjacency=adjacency, queryValues=particleState.velocities)
    curl = warpOperation(particleState, op(WarpOperation.Curl), domain=simulationConfig.domain,
                         adjacency=adjacency, queryValues=particleState.velocities)
    return div, curl.norm(dim=-1)


def computeBalsaraTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    div, curlMag = computeDivCurl(particleState, simulationConfig, supportScheme, adjacency)
    alphas = balsaraFactor(div, curlMag, particleState.soundspeeds, particleState.supports,
                           schemeConfig.viscositySwitchParams.balsara_const)
    return alphas, ViscositySwitchState(
        alpha0s=particleState.alpha0s, alphas=alphas, M=None, M_inv=None, div=div, ddivdt=None,
        Shear=None, Rot=None, R=None, Xi=None, v_sig=None)


def computeBalsaraUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                         supportScheme=None, adjacency=None):
    return particleState.alpha0s, switchState
