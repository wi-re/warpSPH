"""Two config surfaces for the CRK-SPH scheme (`schemes/crkSPH.py`,
`schemes/builder.py`'s CRK `SchemeBundle`, `modules/crk/{accel,dudt}.py`):
`CRKViscosity` (a `wp.struct` of CRK-limiter / van-Leer-limiter tunables) and
`CRKSPHConfig`, a `CompressibleSPHConfig` subclass that adds `crkViscosityParams`
plus CRK-tuned `diffusionParams` defaults. Note: `schemes/crkSPH.py`'s
`crkSPH_step` type-hints its `schemeConfig` parameter as `CompSPHConfig`, but
`schemes/builder.py` actually passes it a `CRKSPHConfig` (accessing
`.crkViscosityParams`, which `CompSPHConfig` lacks) -- a stale type hint,
harmless since Python doesn't enforce it, but confusing to a reader who trusts it.
"""

__all__ = ['CRKViscosity', 'CRKSPHConfig', 'crkSPHConfigToDict', 'dictToCRKSPHConfig']

from dataclasses import dataclass
import warp as wp
import torch

import warp as wp
from enum import Enum
from warpSPHCore import *
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from dataclasses import dataclass, field


@wp.struct
class CRKViscosity:
    # Limiter constants in units of r/H (H = support radius). A value <= 0 means "derive from n_h" and is
    # resolved per step by `resolveCRKLimiter`: eta_crit = 1/n_h, eta_fold = 0.2/n_h, i.e. one and 0.2 nominal
    # particle spacings (Frontiere et al. 2017 Eqs. 51-53 with (eta_crit, eta_fold) = (1/n_h, 0.2) in units of the
    # smoothing scale; Spheral: etaCritFrac/nPerh, etaFoldFrac/nPerh). CRKSPH_LIMITER_PLAN O2. Set an explicit
    # positive value to override.
    eta_fold: scalar_t = field(default = scalar_t(-1.0))
    eta_crit: scalar_t = field(default = scalar_t(-1.0))

    enableCRKLimiter: bool = field(default = True)
    enableVanLeerLimiter: bool = field(default = True)

    forceVanLeerOff: bool = field(default = False)
    forceVanLeerOn: bool = field(default = False)

    # Pair weight of the CRK force / energy terms: False = V_i V_j (Frontiere 2017 Eq. 38), True = ((V_i + V_j)/2)^2 (Spheral's w_ij^2,
    # CRKSPH_LIMITER_PLAN note (f)). Both are symmetric in i <-> j, so both conserve.
    meanVolumeWeights: bool = field(default = False)

def resolveCRKLimiter(params, n_h):
    """`params` with any non-positive `eta_crit` / `eta_fold` replaced by the n_h-derived value
    (1/n_h, 0.2/n_h). Returns `params` itself when both are explicit."""
    if params.eta_crit > 0 and params.eta_fold > 0:
        return params
    out = CRKViscosity()
    for name in ('enableCRKLimiter', 'enableVanLeerLimiter', 'forceVanLeerOff', 'forceVanLeerOn', 'meanVolumeWeights'):
        setattr(out, name, getattr(params, name))
    out.eta_crit = float(params.eta_crit) if params.eta_crit > 0 else 1.0 / float(n_h)
    out.eta_fold = float(params.eta_fold) if params.eta_fold > 0 else 0.2 / float(n_h)
    return out

def buildDefaultCRKViscosityParams():
    crkViscosityParams = CRKViscosity()
    crkViscosityParams.eta_fold = -1.0   # derived from n_h, see CRKViscosity
    crkViscosityParams.eta_crit = -1.0
    crkViscosityParams.enableCRKLimiter = True
    crkViscosityParams.enableVanLeerLimiter = True
    crkViscosityParams.forceVanLeerOff = False
    crkViscosityParams.forceVanLeerOn = False
    crkViscosityParams.meanVolumeWeights = False
    
    return crkViscosityParams


from .moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch
from ..enumTypes import EnergyScheme

# from ..modules import *
from warpSPHCore import *

from dataclasses import dataclass, field

from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig

def buildDefaultDiffusionParamsCRKSPH():
    diffusionParams = DiffusionParameters()
        
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 1
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 1
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Monaghan1992.value
    diffusionParams.thermalConductivityTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    
    return diffusionParams

@dataclass
class CRKSPHConfig(CompressibleSPHConfig):
    energyScheme: EnergyScheme = field(default=EnergyScheme.CRK, metadata={'description': 'Energy scheme for the simulation'})

    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsCRKSPH)
    # no switch by default (the Monaghan host's Rosswog default is not this scheme's)
    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)
    crkViscosityParams: CRKViscosity = field(default_factory=buildDefaultCRKViscosityParams)
    schemeName: str = field(default='CRKSPH', metadata={'description': 'Name of the CRK SPH scheme to use'})
    
    compatibleEnergy: bool = field(default=True, metadata={'description': 'Whether to use a compatible energy discretization (e.g. evolve total energy and compute internal energy from it) or not (e.g. evolve internal energy directly)'})

from typing import Dict, Any

def crkSPHConfigToDict(config: CRKSPHConfig) -> Dict[str, Any]:
    baseDict = compressibleConfigToDict(config)
    baseDict.update({
        'energyScheme': config.energyScheme.name,
        'compatibleEnergy': config.compatibleEnergy,
        'crkViscosityParams': {
            'eta_fold': config.crkViscosityParams.eta_fold,
            'eta_crit': config.crkViscosityParams.eta_crit,
            'enableCRKLimiter': config.crkViscosityParams.enableCRKLimiter,
            'enableVanLeerLimiter': config.crkViscosityParams.enableVanLeerLimiter,
            'forceVanLeerOff': config.crkViscosityParams.forceVanLeerOff,
            'forceVanLeerOn': config.crkViscosityParams.forceVanLeerOn,
            'meanVolumeWeights': config.crkViscosityParams.meanVolumeWeights
        }
    })
    return baseDict

def dictToCRKSPHConfig(configDict: Dict[str, Any]) -> CRKSPHConfig:
    compressibleConfig = dictToCompressibleConfig(configDict)
    crkSPHConfig = CRKSPHConfig(**compressibleConfig.__dict__)
    crkSPHConfig.energyScheme = EnergyScheme[configDict['energyScheme']] if isinstance(configDict['energyScheme'], str) else configDict['energyScheme']
    crkSPHConfig.compatibleEnergy = configDict['compatibleEnergy']
    crkViscosityParamsDict = configDict['crkViscosityParams']
    crkSPHConfig.crkViscosityParams = CRKViscosity()
    crkSPHConfig.crkViscosityParams.eta_fold = crkViscosityParamsDict['eta_fold']
    crkSPHConfig.crkViscosityParams.eta_crit = crkViscosityParamsDict['eta_crit']
    crkSPHConfig.crkViscosityParams.enableCRKLimiter = crkViscosityParamsDict['enableCRKLimiter']
    crkSPHConfig.crkViscosityParams.enableVanLeerLimiter = crkViscosityParamsDict['enableVanLeerLimiter']
    crkSPHConfig.crkViscosityParams.forceVanLeerOff = crkViscosityParamsDict['forceVanLeerOff']
    crkSPHConfig.crkViscosityParams.forceVanLeerOn = crkViscosityParamsDict['forceVanLeerOn']
    crkSPHConfig.crkViscosityParams.meanVolumeWeights = crkViscosityParamsDict.get('meanVolumeWeights', False)
    
    return crkSPHConfig