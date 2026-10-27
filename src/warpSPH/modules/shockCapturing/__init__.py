"""Time-varying artificial-viscosity switches (Cullen & Dehnen 2010, Hopkins' modified form).

Dispatches between the two per-particle alpha-viscosity schemes on
``schemeConfig.viscositySwitchParams.scheme``, plus their shared support code
(shear/rotation tensor, CRK-style gradient correction matrix, signal velocity).

Layout: one subpackage per switch (`Balsara1995/`, `CullenDehnen2010/`, `CullenHopkins/`, `Colagrossi2004/`, `MorrisMonaghan1997/`,
`ReadHayfield2012/`, `Rosswog2000/`, `Rosswog2020/`, `Sphenix2022/`, `Wadsley2017/`), each with its frontend in `switch.py` (terms, update,
step hook) and, where it has them, its warp kernels in `kernels.py`; the package re-exports the switch's names, so
`shockCapturing.Wadsley2017` is the import path. Only what several switches share stays at this level: `common.py` (M, shear tensor,
divergence), `wp_computeM.py`, `wp_vsig.py`, `switchState.py`, `switchRelaxation.py`, and the registry `wrapper.py`.
"""

from .wrapper import computeViscositySwitchTerms, updateViscositySwitch
from .switchState import ViscositySwitchState

__all__ = ['computeViscositySwitchTerms', 'updateViscositySwitch', 'ViscositySwitchState']