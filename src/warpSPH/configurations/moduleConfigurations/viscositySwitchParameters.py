"""`ViscositySwitchConfig`: Cullen-Dehnen-style viscosity-switch tuning
(alpha bounds, beta parameters, divergence scheme), embedded as
`.viscositySwitchParams` on `CompressibleSPHConfig`/`CompSPHConfig`/
`CRKSPHConfig`/`WeaklyCompressibleSPHConfig`/`IncompressibleSPHConfig` and read
by `modules/shockCapturing/CullenDehnen2010.py`. Note: `limitXi` is declared
twice below (default `False` at the top, default `True` further down) --
dataclass field redefinition means only the second (`True`, "limit the xi
parameter in the Cullen-Dehnen switch") is actually live; the first is dead.
"""

__all__ = ['ViscositySwitchConfig', 'viscositySwitchConfigToDict', 'dictToViscositySwitchConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum



@dataclass
class ViscositySwitchConfig:
    scheme: ViscositySwitch = field(default=ViscositySwitch.NoneSwitch, metadata={'description': 'Viscosity switch to use'})
    limitXi: bool = field(default=False, metadata={'description': 'Whether to limit the viscosity switch based on the xi parameter'})
    correctVelocityGradient: bool = field(default=False, metadata={'description': 'Whether to apply the correction matrix to the velocity gradient in the viscosity switch computation'})
    divergenceScheme: Optional[str] = field(default='naive', metadata={'description': 'Scheme to compute the divergence for the viscosity switch. Options are "naive" for the standard SPH divergence and "cullen"'})

    alpha_min : float = field(default=0.02, metadata={'description': 'Minimum alpha value for the viscosity switch'})
    alpha_max : float = field(default=2.0, metadata={'description': 'Maximum alpha value for the viscosity switch'})

    beta_c: float = field(default=0.7, metadata={'description': 'Beta parameter for the Cullen-Dehnen switch'})
    beta_d: float = field(default=0.05, metadata={'description': 'Beta parameter for the Cullen-Dehnen switch'})
    beta_xi: float = field(default=2.0, metadata={'description': 'Beta parameter for the xi limiter in the Cullen-Dehnen switch'})
    limitXi: bool = field(default=True, metadata={'description': 'Whether to limit the xi parameter in the Cullen-Dehnen switch'})

    ns: float = field(default=0.05, metadata={'description': "Noise parameter for the ReadHayfield2012 switch (eq. 21 denominator)"})
    balsara_const: float = field(default=1.0e-4, metadata={'description': "Balsara limiter constant for the ReadHayfield2012 switch (eq. 32)"})



def viscositySwitchConfigToDict(viscositySwitchConfig: ViscositySwitchConfig) -> Dict[str, Any]:
    return {
        'scheme': viscositySwitchConfig.scheme.name if isinstance(viscositySwitchConfig.scheme, ViscositySwitch) else viscositySwitchConfig.scheme,
        'limitXi': viscositySwitchConfig.limitXi,
        'correctVelocityGradient': viscositySwitchConfig.correctVelocityGradient,
        'divergenceScheme': viscositySwitchConfig.divergenceScheme,
        'alpha_min': viscositySwitchConfig.alpha_min,
        'alpha_max': viscositySwitchConfig.alpha_max,
        'beta_c': viscositySwitchConfig.beta_c,
        'beta_d': viscositySwitchConfig.beta_d,
        'beta_xi': viscositySwitchConfig.beta_xi,
        'ns': viscositySwitchConfig.ns,
        'balsara_const': viscositySwitchConfig.balsara_const
    }

def dictToViscositySwitchConfig(viscositySwitchConfigDict: Dict[str, Any]) -> ViscositySwitchConfig:
    viscositySwitchConfig = ViscositySwitchConfig()
    viscositySwitchConfig.scheme = ViscositySwitch[viscositySwitchConfigDict['scheme']] if isinstance(viscositySwitchConfigDict['scheme'], str) else viscositySwitchConfigDict['scheme']
    viscositySwitchConfig.limitXi = viscositySwitchConfigDict['limitXi']
    viscositySwitchConfig.correctVelocityGradient = viscositySwitchConfigDict['correctVelocityGradient']
    viscositySwitchConfig.divergenceScheme = viscositySwitchConfigDict['divergenceScheme']
    viscositySwitchConfig.alpha_min = viscositySwitchConfigDict['alpha_min']
    viscositySwitchConfig.alpha_max = viscositySwitchConfigDict['alpha_max']
    viscositySwitchConfig.beta_c = viscositySwitchConfigDict['beta_c']
    viscositySwitchConfig.beta_d = viscositySwitchConfigDict['beta_d']
    viscositySwitchConfig.beta_xi = viscositySwitchConfigDict['beta_xi']
    viscositySwitchConfig.ns = viscositySwitchConfigDict.get('ns', 0.05)
    viscositySwitchConfig.balsara_const = viscositySwitchConfigDict.get('balsara_const', 1.0e-4)

    return viscositySwitchConfig