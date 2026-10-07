"""`DiffusionParameters` (a `wp.struct` of artificial-viscosity / thermal-
conductivity coefficients) and `ViscosityTerms` (which formulation each
coefficient feeds, e.g. Monaghan1992, Price2012_98), embedded as
`diffusionParams` on `CompressibleSPHConfig`/`CompSPHConfig`/`CRKSPHConfig` and
read by the compressible dissipation modules. Each of the three scheme configs
supplies its own `buildDefaultDiffusionParamsCompressibleSPH`/`CompSPH`/`CRKSPH`
default-factory with slightly different tuned values (e.g. `C_q`/`Cu_q` are 0
for the base compressible scheme's defaults here, but 1 or 2 for CompSPH/CRKSPH)
-- only this module's `buildDefaultDiffusionParamsCompressibleSPH` lives here;
the CompSPH/CRKSPH variants live in their own config files.
"""

__all__ = ['ViscosityTerms', 'BetaMode', 'DiffusionParameters', 'buildDefaultDiffusionParamsCompressibleSPH', 'diffusionParamsToDict', 'dictToDiffusionParams']

from typing import Dict, Any
from warpSPHCore import *
from dataclasses import dataclass, field
import warp as wp
from enum import Enum


class ViscosityTerms(Enum):
    Default = 0
    MonaghanGingold1983 = 1
    Cleary1998 = 2
    Monaghan1992 = 3
    Monaghan1997a = 4
    Monaghan1997b = 5
    Dukowicz = 6
    Price2012_98 = 7
    Price2012 = 8
    Price2008 = 9
    Wadsley2008 = 10
    DeltaSPH = 11
    Frontiere2017 = 12   # CRKSPH's one-sided Q_i: Monaghan (1992) mu, own h, rho, c, alpha, eps^2 = 1e-2 (Frontiere et al. 2017)

class BetaMode(Enum):
    """How the quadratic coefficient `beta` (`C_q`) responds to the switched alpha.

    The two halves of AV_PLAN's `CoefficientPolicy`: `SwitchedAlphaCoupledBeta` and
    `SwitchedAlphaFixedBeta` (an un-switched alpha is `NoneSwitch`, alpha = 1, in
    either mode).

    * `Coupled` (default, the historical behaviour): `beta = alpha_bar * C_q`, i.e.
      beta follows the switch -- Chen & Nixon (2025)'s recommendation.
    * `Fixed`: `beta = C_q` whatever alpha is -- what Garcia-Senz & Cabezon (2026)
      (`beta = 2`), Sphenix (`beta_V = 3`) and Wadsley (`beta = 2`) specify, and what
      PHANTOM does. Before this existed that setting was inexpressible.

    `DiffusionParameters.scaleBeta` is a third, legacy behaviour (`beta = alpha_bar^2 *
    C_l * C_q`, a second alpha factor that matches no paper); it only applies in
    `Coupled` mode and is ignored in `Fixed`.
    """
    Coupled = 0
    Fixed = 1


@wp.struct
class DiffusionParameters:
    c_s: scalar_t = field(default=scalar_t(1.0)) # Speed of sound, used in some formulations to compute the signal velocity
    C_l: scalar_t = field(default=scalar_t(1.0)) # Linear viscosity coefficient, also referred to as alpha in some formulations
    C_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic viscosity coefficient, also referred to as beta in some formulations
    Cu_l: scalar_t = field(default=scalar_t(1.0)) # Linear thermal conductivity coefficient, also referred to as alpha_u in some formulations
    Cu_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic thermal conductivity coefficient, also referred to as beta_u in some formulations
    
    K: scalar_t = field(default=scalar_t(1.0)) # Overall viscosity scaling factor
    thermalConductivity: scalar_t = field(default=scalar_t(0.5)) # Overall thermal conductivity scaling factor
    viscosityTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    thermalConductivityTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Thermal conductivity formulation to use, e.g. Monaghan1997 thermal conductivity term, Cleary1998 thermal conductivity term etc.
    betaMode: wp.int32 = field(default=BetaMode.Coupled.value) # `BetaMode`: Coupled (beta = alpha_bar * C_q, the default) or Fixed (beta = C_q, independent of the switched alpha).
    scaleBeta: wp.bool = field(default=False) # LEGACY, `Coupled` mode only: historically `beta = alpha_bar^2 * C_l * C_q` (a second alpha factor -- the comment here used to describe only the first half). Matches no paper; kept so stored configs reproduce.  # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    monaghanSwitch: wp.bool = field(default=True) # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    correctXi: wp.bool = field(default=True) # Divide the viscosity by the kernel-dependent length factor `sphKernel_xi` (packing ratio x kernel scale) so the coefficients are comparable across kernels. Unrelated to Cullen & Dehnen's limiter Xi (`ViscositySwitchConfig.limitXi`). The name is kept because it is serialised in stored configs.

    
def buildDefaultDiffusionParamsCompressibleSPH():
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 0
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 0
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.thermalConductivityTerm = ViscosityTerms.Price2008.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    
    return diffusionParams



def diffusionParamsToDict(diffusionParams: DiffusionParameters) -> Dict[str, Any]:
    return {
        'c_s': diffusionParams.c_s,
        'C_l': diffusionParams.C_l,
        'C_q': diffusionParams.C_q,
        'Cu_l': diffusionParams.Cu_l,
        'Cu_q': diffusionParams.Cu_q,
        'K': diffusionParams.K,
        'thermalConductivity': diffusionParams.thermalConductivity,
        'viscosityTerm': diffusionParams.viscosityTerm.name if isinstance(diffusionParams.viscosityTerm, ViscosityTerms) else diffusionParams.viscosityTerm,
        'thermalConductivityTerm': diffusionParams.thermalConductivityTerm.name if isinstance(diffusionParams.thermalConductivityTerm, ViscosityTerms) else diffusionParams.thermalConductivityTerm,
        'scaleBeta': diffusionParams.scaleBeta,
        'betaMode': BetaMode(diffusionParams.betaMode).name,
        'monaghanSwitch': diffusionParams.monaghanSwitch,
        'correctXi': diffusionParams.correctXi
    }

def dictToDiffusionParams(diffusionParamsDict: Dict[str, Any]) -> DiffusionParameters:
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = diffusionParamsDict['c_s']
    diffusionParams.C_l = diffusionParamsDict['C_l']
    diffusionParams.C_q = diffusionParamsDict['C_q']
    diffusionParams.Cu_l = diffusionParamsDict['Cu_l']
    diffusionParams.Cu_q = diffusionParamsDict['Cu_q']
    diffusionParams.K = diffusionParamsDict['K']
    diffusionParams.thermalConductivity = diffusionParamsDict['thermalConductivity']
    diffusionParams.viscosityTerm = ViscosityTerms[diffusionParamsDict['viscosityTerm']] if isinstance(diffusionParamsDict['viscosityTerm'], str) else diffusionParamsDict['viscosityTerm']
    diffusionParams.thermalConductivityTerm = ViscosityTerms[diffusionParamsDict['thermalConductivityTerm']] if isinstance(diffusionParamsDict['thermalConductivityTerm'], str) else diffusionParamsDict['thermalConductivityTerm']
    diffusionParams.scaleBeta = diffusionParamsDict['scaleBeta']
    # absent in configs stored before the mode existed -> the historical Coupled behaviour
    mode = diffusionParamsDict.get('betaMode', BetaMode.Coupled.name)
    diffusionParams.betaMode = (BetaMode[mode] if isinstance(mode, str) else BetaMode(mode)).value
    diffusionParams.monaghanSwitch = diffusionParamsDict['monaghanSwitch']
    diffusionParams.correctXi = diffusionParamsDict['correctXi']
    
    return diffusionParams