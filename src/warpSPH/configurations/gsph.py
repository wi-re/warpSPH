"""Config of the Godunov SPH scheme (`schemes/gsph.py`, GODUNOV_SPH_PLAN): a `CompressibleSPHConfig` whose `diffusionParams` carry the Riemann
settings instead of the artificial-viscosity ones -- `riemannSolver`, `velocityPairPolicy` (`Raw`: first-order states; `Limited`: the
second-order reconstruction of the velocity), `riemannReconstruction` (the same for `rho` and `P`), `limiterType` and the taper constants. There is no
viscosity switch and no viscosity term in the scheme; the fields that only the AV operator reads are unused.

`reconstructionOrder` is derived: 2 when `velocityPairPolicy` is `Limited` / `BalsaraLimited`, else 1.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig
from .moduleConfigurations.diffusionParameters import (DiffusionParameters, LimiterType, RiemannSolver, VelocityPairPolicy,
                                                       buildDefaultDiffusionParamsCompressibleSPH)
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig

__all__ = ['GSPHConfig', 'buildDefaultDiffusionParamsGSPH', 'gsphConfigToDict', 'dictToGSPHConfig']


def buildDefaultDiffusionParamsGSPH() -> DiffusionParameters:
    """Adaptive solver, second-order states: the limited velocity and the reconstructed `(rho, P)`."""
    p = buildDefaultDiffusionParamsCompressibleSPH()
    p.riemannSolver = RiemannSolver.Adaptive.value
    p.velocityPairPolicy = VelocityPairPolicy.Limited.value
    p.riemannReconstruction = True
    p.limiterType = LimiterType.VanLeerFrontiere.value
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
