"""Config of the Godunov SPH scheme (`schemes/gsph.py`, GODUNOV_SPH_PLAN): a `CompressibleSPHConfig` whose `diffusionParams` carry the Riemann
settings instead of the artificial-viscosity ones -- `riemannSolver`, `velocityPairPolicy` (`Raw`: first-order states; `Limited`: the
second-order reconstruction of the velocity), `riemannReconstruction` (the same for `rho` and `P`), `limiterType` and the taper constants. There is no
viscosity switch and no viscosity term in the scheme; the fields that only the AV operator reads are unused.

`reconstructionOrder` is derived: 2 when `velocityPairPolicy` is `Limited` / `BalsaraLimited`, else 1.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig
from .moduleConfigurations.diffusionParameters import (DiffusionParameters, LimiterType, RiemannSolver, StateLimiter, VelocityPairPolicy,
                                                       buildDefaultDiffusionParamsCompressibleSPH)
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig

__all__ = ['GSPHConfig', 'InutsukaGSPHConfig', 'buildDefaultDiffusionParamsGSPH', 'buildDefaultDiffusionParamsInutsuka', 'gsphConfigToDict',
           'dictToGSPHConfig', 'dictToInutsukaGSPHConfig']


def buildDefaultDiffusionParamsGSPH() -> DiffusionParameters:
    """Adaptive solver, second-order states: the limited velocity and the reconstructed `(rho, P)`, limited by van Leer's harmonic mean
    (`StateLimiter.VanLeerHarmonic`, Murante et al. 2011; the user's default decision 2026-10-08, was `PairRatio`)."""
    p = buildDefaultDiffusionParamsCompressibleSPH()
    p.riemannSolver = RiemannSolver.Adaptive.value
    p.velocityPairPolicy = VelocityPairPolicy.Limited.value
    p.riemannReconstruction = True
    p.limiterType = LimiterType.VanLeerFrontiere.value
    p.stateLimiter = StateLimiter.VanLeerHarmonic.value
    return p


def buildDefaultDiffusionParamsInutsuka() -> DiffusionParameters:
    """Murante et al. (2011)'s reference (Adaptive solver, harmonic-mean limiter) with Inutsuka (2002) Eq. 75's first-order shock switch, `C = 3`
    (the user's default decision 2026-10-08: without it the strong-shock energy is not conserved, docs/av/godunov_l3_2026-10-08)."""
    p = buildDefaultDiffusionParamsGSPH()
    p.shockSwitchC = 3.0
    return p


@dataclass
class GSPHConfig(CompressibleSPHConfig):
    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsGSPH)
    # no viscosity switch: the scheme has no artificial viscosity
    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)
    schemeName: str = field(default='GSPH', metadata={'description': 'Godunov SPH'})


def gsphConfigToDict(config: GSPHConfig) -> Dict[str, Any]:
    return compressibleConfigToDict(config)


def dictToGSPHConfig(configDict: Dict[str, Any]) -> GSPHConfig:
    base = dictToCompressibleConfig(configDict)
    return GSPHConfig(**base.__dict__)


@dataclass
class InutsukaGSPHConfig(GSPHConfig):
    """Inutsuka (2002) Godunov SPH (`schemes/gsphInutsuka.py`, layer 3): Gaussian kernel (`supports / 3` is its h), interpolated specific volume, interface `s*`.
    `cubicVolume`: the cubic Hermite interpolant of `V = 1/rho` (Murante's reference) instead of the linear one; `verletScale` of the neighbour list is
    raised to reach `sqrt2` times the support."""
    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsInutsuka)
    schemeName: str = field(default='GSPH (Inutsuka)', metadata={'description': 'Godunov SPH, kernel-convolution form'})


def dictToInutsukaGSPHConfig(configDict: Dict[str, Any]) -> InutsukaGSPHConfig:
    base = dictToCompressibleConfig(configDict)
    return InutsukaGSPHConfig(**base.__dict__)
