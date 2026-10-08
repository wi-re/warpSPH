"""`ViscositySwitchConfig`: Cullen-Dehnen-style viscosity-switch tuning
(alpha bounds, beta parameters, divergence scheme), embedded as
`.viscositySwitchParams` on `CompressibleSPHConfig`/`CompSPHConfig`/
`CRKSPHConfig`/`WeaklyCompressibleSPHConfig`/`IncompressibleSPHConfig` and read
by `modules/shockCapturing/CullenDehnen2010.py`. `limitXi` is C&D's limiter
Xi, unrelated to the kernel length factor `DiffusionParameters.correctXi` /
`sphKernel_xi`. (It used to be declared twice -- a dead `False` default and the live
`True` one; only the live one is kept.)
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
    correctVelocityGradient: bool = field(default=False, metadata={'description': 'Whether to apply the correction matrix to the velocity gradient in the viscosity switch computation'})
    divergenceScheme: Optional[str] = field(default='naive', metadata={'description': 'Scheme to compute the divergence for the viscosity switch. Options are "naive" for the standard SPH divergence and "cullen"'})

    alpha_min : float = field(default=0.02, metadata={'description': 'Minimum alpha value for the viscosity switch'})
    alpha_max : float = field(default=2.0, metadata={'description': 'Maximum alpha value for the viscosity switch'})

    beta_c: float = field(default=0.7, metadata={'description': 'Beta parameter for the Cullen-Dehnen switch'})
    beta_d: float = field(default=0.05, metadata={'description': 'Beta parameter for the Cullen-Dehnen switch'})
    beta_xi: float = field(default=2.0, metadata={'description': 'Beta parameter for the xi limiter in the Cullen-Dehnen switch'})
    limitXi: bool = field(default=True, metadata={'description': 'Whether to limit the xi parameter in the Cullen-Dehnen switch'})

    ns: float = field(default=0.05, metadata={'description': "Noise parameter for the ReadHayfield2012 switch (eq. 21 denominator)"})
    balsara_const: float = field(default=1.0e-4, metadata={'description': "Balsara limiter constant `eps` in `|div| / (|div| + |curl| + eps c / h)` (ReadHayfield2012 eq. 32; also the Balsara1995 and Colagrossi2004 switches)"})
    morris_C1: float = field(default=0.2, metadata={'description': "Decay-rate constant C_1 of the source-and-decay switches (tau = h / (C_1 c)): MorrisMonaghan1997 Eq. 6 (paper range 0.1-0.2) and Rosswog2000"})

    entropy_eps0: float = field(default=1.0e-4, metadata={'description': "Rosswog2020: entropy rate below which alpha_des = 0 (the paper's l_0 = log(1e-4))"})
    entropy_eps1: float = field(default=5.0e-2, metadata={'description': "Rosswog2020: entropy rate at and above which alpha_des = alpha_max (l_1 = log(5e-2))"})
    entropy_decay: float = field(default=30.0, metadata={'description': "Rosswog2020: alpha decay time in units of tau = h / c (Eq. 17)"})
    sphenix_ell: float = field(default=0.05, metadata={'description': "Sphenix2022: alpha decay length ell_V; tau = gamma_K ell_V h / c = ell_V H / c with H the support radius stored here (Borrow et al. 2022 Eq. 24)"})
    wadsley_tau: float = field(default=0.2, metadata={'description': "Wadsley2017: alpha decay tau = h / (wadsley_tau c) (Eq. 27, 0.2 in the paper; h the smoothing length)"})
    wadsley_prefactor: float = field(default=-1.0, metadata={'description': "Wadsley2017: the prefactor of A_i = prefactor h^2 xi max(-dD/dt, 0) (Eq. 26's 2, re-derived for this repo's h convention in Wadsley2017.py); <= 0 uses the derived value"})



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
        'balsara_const': viscositySwitchConfig.balsara_const,
        'morris_C1': viscositySwitchConfig.morris_C1,
        'entropy_eps0': viscositySwitchConfig.entropy_eps0,
        'entropy_eps1': viscositySwitchConfig.entropy_eps1,
        'entropy_decay': viscositySwitchConfig.entropy_decay,
        'sphenix_ell': viscositySwitchConfig.sphenix_ell,
        'wadsley_tau': viscositySwitchConfig.wadsley_tau,
        'wadsley_prefactor': viscositySwitchConfig.wadsley_prefactor,
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
    viscositySwitchConfig.morris_C1 = viscositySwitchConfigDict.get('morris_C1', 0.2)
    viscositySwitchConfig.entropy_eps0 = viscositySwitchConfigDict.get('entropy_eps0', 1.0e-4)
    viscositySwitchConfig.entropy_eps1 = viscositySwitchConfigDict.get('entropy_eps1', 5.0e-2)
    viscositySwitchConfig.entropy_decay = viscositySwitchConfigDict.get('entropy_decay', 30.0)
    viscositySwitchConfig.sphenix_ell = viscositySwitchConfigDict.get('sphenix_ell', 0.05)
    viscositySwitchConfig.wadsley_tau = viscositySwitchConfigDict.get('wadsley_tau', 0.2)
    viscositySwitchConfig.wadsley_prefactor = viscositySwitchConfigDict.get('wadsley_prefactor', -1.0)

    return viscositySwitchConfig