"""`fluidProperties`: EOS type plus rest density / sound-speed parameters,
embedded as `.fluid` on `WeaklyCompressibleSPHConfig` and `IncompressibleSPHConfig`
and read by `initializers/weaklyCompressible.py` and the EOS modules. Note: this
is a *different* dataclass from `modules.eos.props.fluidProperties` -- that one
is a smaller, unrelated bundle used only by the currently-dead
`modules/eos/gas.py` path; this is the one actually referenced as
`schemeConfig.fluid` elsewhere in the codebase.
"""

__all__ = ['fluidProperties', 'buildDefaultFluidProperties']

from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum


@dataclass
class fluidProperties:
    eosType: EquationOfState = field(default=EquationOfState.isoThermal,metadata={"description": "Type of equation of state"})

    restDensity: float = field(default = 1.0, metadata={"description": "Rest density of the fluid"})

    polytropicExponent: Optional[float] = field(default=7.0, metadata={"description": "Polytropic exponent for polytropic EOS"})
    kappa: Optional[float] = field(default=1.3, metadata={"description": "Kappa"})
    gas_constant: Optional[float] = field(default=8.314, metadata={"description": "Gas constant"})
    molarMass: Optional[float] = field(default=0.02897, metadata={"description": "Molar mass of the gas"})

    fixedSoundSpeed: Optional[float] = field(default=10.0, metadata={"description": "Fixed sound speed"})
    backgroundPressure: float = field(default=0.0, metadata={"description": "Uniform background pressure p_b added to the weakly-compressible EOS output (p <- p_EOS + p_b). A small positive p_b pushes particles together near a free surface (where p_EOS ~ 0), opposing the delta+ shift's outward drift and the tensile instability, at the cost of a spurious surface-tension-like rounding of sharp features. 0.0 = off (the default; no change to any existing case)."})

def buildDefaultFluidProperties() -> fluidProperties:
    return fluidProperties(
        eosType=EquationOfState.isoThermal,
        restDensity=1.0,
        polytropicExponent=7.0,
        kappa=1.3,
        gas_constant=8.314,
        molarMass=0.02897,
        fixedSoundSpeed=10.0,
        backgroundPressure=0.0,
    )