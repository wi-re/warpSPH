"""Two config surfaces for the CRK-SPH scheme (`schemes/compressible/crkSPH.py`,
`schemes/builder.py`'s CRK `SchemeBundle`, `modules/crk/{accel,dudt}.py`):
`CRKViscosity` (a `wp.struct` of CRK-limiter / van-Leer-limiter tunables) and
`CRKSPHConfig`, a `CompressibleSPHConfig` subclass that adds `crkViscosityParams`
plus CRK-tuned `diffusionParams` defaults. Note: `schemes/compressible/crkSPH.py`'s
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
    # particle spacings, matching current Spheral's etaCritFrac/nPerh and etaFoldFrac/nPerh after unit conversion.
    # Frontiere 2017 Eqs. 51-53 instead state eta_fold=0.2 in smoothing-length units: in r/R this is 0.2/K,
    # K=kernel support extent, and agrees with this default when nPerh=n_h/K=1. See the limiter-threshold audit.
    # CRKSPH_LIMITER_PLAN O2 records the existing default decision. Set an explicit
    # positive value to override.
    eta_fold: scalar_t = field(default = scalar_t(-1.0))
    eta_crit: scalar_t = field(default = scalar_t(-1.0))

    enableCRKLimiter: bool = field(default = True)
    enableVanLeerLimiter: bool = field(default = True)
    # `LimiterType`: which slope limiter `enableVanLeerLimiter` switches on (VanLeerFrontiere = 0, the one CRKSPH has always used)
    limiterType: wp.int32 = field(default = 0)

    forceVanLeerOff: bool = field(default = False)
    forceVanLeerOn: bool = field(default = False)

    # Pair weight: False = V_i V_j (Frontiere 2017 Eq. 38 and pinned Spheral
    # 116c71f2), True = ((V_i + V_j)/2)^2 experimental alternative.
    # Both are symmetric in i <-> j, so both conserve pair momentum.
    meanVolumeWeights: bool = field(default = False)
    # Experimental AV length parity control: zero retains sphKernelScale;
    # positive uses h=compact support/this value in Frontiere Q only.
    viscosityKernelScale: scalar_t = field(default=scalar_t(0.0))
    oneSidedWork: bool = field(default=False)

def resolveCRKLimiter(params, n_h):
    """`params` with any non-positive `eta_crit` / `eta_fold` replaced by the n_h-derived value
    (1/n_h, 0.2/n_h). Returns `params` itself when both are explicit."""
    if params.eta_crit > 0 and params.eta_fold > 0:
        return params
    out = CRKViscosity()
    for name in ('enableCRKLimiter', 'enableVanLeerLimiter', 'limiterType', 'forceVanLeerOff', 'forceVanLeerOn', 'meanVolumeWeights', 'viscosityKernelScale', 'oneSidedWork'):
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
    crkViscosityParams.limiterType = LimiterType.VanLeerFrontiere.value
    crkViscosityParams.forceVanLeerOff = False
    crkViscosityParams.forceVanLeerOn = False
    crkViscosityParams.meanVolumeWeights = False
    
    return crkViscosityParams


from .moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms, LimiterType
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
    crkDensityPolicy: str = field(default='legacy', metadata={'description': 'Opt-in CRK material density rule: legacy, sameMaterial (Spheral raw-kernel sum), or massReplacement (paper corrected-kernel mass rule). Volumes/moments retain all neighbours.'})
    crkEvolution: str = field(default='summation', metadata={'description': 'Default summation: density/volume every RHS. Experimental spheral: m/rho reset, same-material density, bounded continuity volume. Experimental paper: Eq76 reset once/step, continuity density, geometric Eq75 volume every RHS. Stage modes require midpoint.'})
    crkPairStorage: str = field(default='recompute', metadata={'description': 'Recompute shared pair physics directly into X6 work by default on native eager single-gas steps; expanded selects reference storage and unsupported contexts fall back automatically.'})
    crkViscosityGradient: str = field(default='current', metadata={'description': 'Experimental AV timing control: current stage (default) or cached previous RHS gradient; cached requires spheral midpoint evolution.'})
    crkEnergyPartition: str = field(default='native', metadata={'description': 'Midpoint work partition: native entropy estimator, entropy with final compatible-work sign, or Spheral node dudt weights.'})
    crkEnergyPredictor: str = field(default='raw', metadata={'description': 'Experimental midpoint internal energy predictor: raw dudt (native) or compatible pair work.'})

    def validate(self, config=None):
        super().validate(config)
        if self.crkPairStorage not in ('expanded','recompute'):
            raise ValueError('crkPairStorage must be expanded or recompute')
        if self.crkEnergyPartition not in ('native', 'entropy', 'spheral') or self.crkEnergyPredictor not in ('raw', 'compatible'):
            raise ValueError('Invalid CRK energy partition/predictor')
        if self.crkEnergyPartition != 'native' or self.crkEnergyPredictor != 'raw':
            if not self.compatibleEnergy or self.energyScheme != EnergyScheme.CRK:
                raise ValueError('Experimental CRK energy controls require compatibleEnergy and energyScheme=CRK')
        if self.crkViscosityGradient not in ('current', 'cached'):
            raise ValueError('crkViscosityGradient must be current or cached')
        import math
        if not math.isfinite(float(self.crkViscosityParams.viscosityKernelScale)) or self.crkViscosityParams.viscosityKernelScale < 0:
            raise ValueError('viscosityKernelScale must be finite and nonnegative')
        if self.crkDensityPolicy not in ('legacy', 'sameMaterial', 'massReplacement'):
            raise ValueError('crkDensityPolicy must be legacy, sameMaterial or massReplacement')
        if self.crkDensityPolicy != 'legacy' and self.materials is None:
            raise ValueError('material CRK density policies require an explicit materials table')
        if self.crkEvolution not in ('summation', 'spheral', 'paper'):
            raise ValueError('crkEvolution must be summation, spheral or paper')
        if self.crkEvolution in ('spheral', 'paper'):
            required='sameMaterial' if self.crkEvolution=='spheral' else 'massReplacement'
            if self.crkDensityPolicy != required:
                raise ValueError(f'crkEvolution={self.crkEvolution} requires crkDensityPolicy={required}')
            if self.boundaryConditions:
                raise ValueError('Experimental CRK stage modes do not yet support Dirichlet boundary conditions')
        midpoint=(self.crkEvolution!='summation' or self.crkEnergyPartition!='native' or self.crkEnergyPredictor!='raw' or self.crkViscosityGradient=='cached')
        if midpoint:
            if config is not None:
                if config.integrationScheme.name != 'rungeKutta2':
                    raise ValueError('Experimental CRK controls currently require midpoint rungeKutta2')
                from ..modules.crk.evolution import kernelExtent
                kernelExtent(config.kernel)
                if not bool(config.domain.periodic.all()):
                    raise ValueError('Experimental CRK controls are currently qualified only for periodic fluid cases')

from typing import Dict, Any

def crkSPHConfigToDict(config: CRKSPHConfig) -> Dict[str, Any]:
    baseDict = compressibleConfigToDict(config)
    baseDict.update({
        'energyScheme': config.energyScheme.name,
        'compatibleEnergy': config.compatibleEnergy,
        'crkDensityPolicy': config.crkDensityPolicy,
        'crkEvolution': config.crkEvolution,
        'crkViscosityGradient': config.crkViscosityGradient,
        'crkPairStorage': config.crkPairStorage,
        'crkEnergyPartition': config.crkEnergyPartition,
        'crkEnergyPredictor': config.crkEnergyPredictor,
        'crkViscosityParams': {
            'eta_fold': config.crkViscosityParams.eta_fold,
            'eta_crit': config.crkViscosityParams.eta_crit,
            'enableCRKLimiter': config.crkViscosityParams.enableCRKLimiter,
            'enableVanLeerLimiter': config.crkViscosityParams.enableVanLeerLimiter,
            'limiterType': LimiterType(config.crkViscosityParams.limiterType).name,
            'forceVanLeerOff': config.crkViscosityParams.forceVanLeerOff,
            'forceVanLeerOn': config.crkViscosityParams.forceVanLeerOn,
            'meanVolumeWeights': config.crkViscosityParams.meanVolumeWeights,
            'viscosityKernelScale': float(config.crkViscosityParams.viscosityKernelScale),
            'oneSidedWork': bool(config.crkViscosityParams.oneSidedWork)
        }
    })
    return baseDict

def dictToCRKSPHConfig(configDict: Dict[str, Any]) -> CRKSPHConfig:
    compressibleConfig = dictToCompressibleConfig(configDict)
    crkSPHConfig = CRKSPHConfig(**compressibleConfig.__dict__)
    crkSPHConfig.energyScheme = EnergyScheme[configDict['energyScheme']] if isinstance(configDict['energyScheme'], str) else configDict['energyScheme']
    crkSPHConfig.compatibleEnergy = configDict['compatibleEnergy']
    crkSPHConfig.crkDensityPolicy = configDict.get('crkDensityPolicy', 'legacy')
    crkSPHConfig.crkEvolution = configDict.get('crkEvolution', 'summation')
    crkSPHConfig.crkViscosityGradient = configDict.get('crkViscosityGradient', 'current')
    crkSPHConfig.crkPairStorage = configDict.get('crkPairStorage', 'recompute')
    crkSPHConfig.crkEnergyPartition = configDict.get('crkEnergyPartition', 'native')
    crkSPHConfig.crkEnergyPredictor = configDict.get('crkEnergyPredictor', 'raw')
    crkViscosityParamsDict = configDict['crkViscosityParams']
    crkSPHConfig.crkViscosityParams = CRKViscosity()
    crkSPHConfig.crkViscosityParams.eta_fold = crkViscosityParamsDict['eta_fold']
    crkSPHConfig.crkViscosityParams.eta_crit = crkViscosityParamsDict['eta_crit']
    crkSPHConfig.crkViscosityParams.enableCRKLimiter = crkViscosityParamsDict['enableCRKLimiter']
    crkSPHConfig.crkViscosityParams.enableVanLeerLimiter = crkViscosityParamsDict['enableVanLeerLimiter']
    limiter = crkViscosityParamsDict.get('limiterType', LimiterType.VanLeerFrontiere.name)   # absent before Phase 7b
    crkSPHConfig.crkViscosityParams.limiterType = (LimiterType[limiter] if isinstance(limiter, str) else LimiterType(limiter)).value
    crkSPHConfig.crkViscosityParams.forceVanLeerOff = crkViscosityParamsDict['forceVanLeerOff']
    crkSPHConfig.crkViscosityParams.forceVanLeerOn = crkViscosityParamsDict['forceVanLeerOn']
    crkSPHConfig.crkViscosityParams.meanVolumeWeights = crkViscosityParamsDict.get('meanVolumeWeights', False)
    crkSPHConfig.crkViscosityParams.viscosityKernelScale = crkViscosityParamsDict.get('viscosityKernelScale', 0.)
    crkSPHConfig.crkViscosityParams.oneSidedWork = crkViscosityParamsDict.get('oneSidedWork', False)
    
    return crkSPHConfig
