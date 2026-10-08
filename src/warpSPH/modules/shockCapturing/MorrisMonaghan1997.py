"""Morris & Monaghan (1997) viscosity switch, J. Comput. Phys. 136, 41, Eqs. (5), (6), (13).

    d alpha / dt = -(alpha - alpha_min) / tau + S,    tau = h / (C_1 c),    S = max(-div v, 0)

`alpha_min` is the paper's `alpha_infinity = 0.1` (the repo's `alpha_min` config; set it to
0.1 to match the paper), `C_1 = 0.2` (`morris_C1`, paper range 0.1-0.2) and the source is
the bare compression rate. The pair parameter is the average `0.5 (alpha_a + alpha_b)`
(their Eq. 26), which the pair operator already forms. The decay is taken implicitly
(`relaxAlpha`), so it is stable for any `dt / tau`.

`h` is the smoothing length, converted from the stored support radius the way
`CullenDehnen2010.py` does (`h * f_kern`, `f_kern = 1 / sphKernel_xi`). The diffSPH
implementation this was started from has `tau = h c / l` (dimensionally wrong); the paper's
`h / (C_1 c)` is used here.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from warpSPHCore import *
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...systems.compressibleMonaghan import CompressibleState
from .common import *
from .switchRelaxation import relaxAlpha
from .switchState import ViscositySwitchState

__all__ = ['computeMorrisMonaghanTerms', 'computeMorrisMonaghanUpdate', 'relaxedAlpha']


def relaxedAlpha(particleState: CompressibleState, div: torch.Tensor, dt: float,
                 simulationConfig: SimulationConfig, schemeConfig: CompressibleSPHConfig,
                 sourceScale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One step of the source-and-decay equation; `sourceScale` (e.g. `alpha_max - alpha`)
    multiplies the source (Rosswog et al. 2000 uses it)."""
    cfg = schemeConfig.viscositySwitchParams
    xi_kern = sphKernel_xi(simulationConfig.kernel.value, particleState.positions.shape[1])
    f_kern = float(type(xi_kern)(1.0) / xi_kern)  # float64 build: xi is a wp.float64 (no int / wp.float64, no Tensor * wp.float64)
    tau = particleState.supports * f_kern / (cfg.morris_C1 * particleState.soundspeeds + 1e-14)
    source = (-div).clamp(min=0)
    if sourceScale is not None:
        source = source * sourceScale
    return relaxAlpha(particleState.alpha0s, source, tau, dt, cfg.alpha_min, cfg.alpha_max)


def computeMorrisMonaghanTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    div = computeDivergence(particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    alphas = relaxedAlpha(particleState, div, dt, simulationConfig, schemeConfig)
    return alphas, ViscositySwitchState(
        alpha0s=alphas, alphas=alphas, M=None, M_inv=None, div=div, ddivdt=None,
        Shear=None, Rot=None, R=None, Xi=None, v_sig=None)


def computeMorrisMonaghanUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                                supportScheme=None, adjacency=None):
    # the relaxation happens in `computeMorrisMonaghanTerms` (Read-Hayfield's convention)
    return switchState.alpha0s, switchState
