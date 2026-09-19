"""Dispatch layer selecting a viscosity switch scheme (Cullen-Dehnen, Hopkins, or none) per ``schemeConfig.viscositySwitchParams.scheme``.

``computeViscositySwitchTerms`` computes the per-step alpha/switch state;
``updateViscositySwitch`` advances it over ``dt``. The ``NoneSwitch`` branch
is a no-op that passes ``particleState.alphas``/``alpha0s`` straight through
with ``None`` switch state.
"""

from ...enumTypes import ViscositySwitch

from ...systems.compressibleMonaghan import CompressibleState
from typing import Optional, Union
import torch
from ...configurations import *
from warpSPHCore import *

from .switchState import ViscositySwitchState
from .CullenDehnen2010 import computeCullenTerms, computeCullenUpdate
from .CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate
from .ReadHayfield2012 import computeReadHayfieldTerms, computeReadHayfieldUpdate

__all__ = ['computeViscositySwitchTerms', 'updateViscositySwitch']

def computeViscositySwitchTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    
    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    if switchConfig.scheme == ViscositySwitch.CullenDehnen2010:
        return computeCullenTerms(dt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    elif switchConfig.scheme == ViscositySwitch.CullenHopkins:
        return computeHopkinsTerms(dt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    elif switchConfig.scheme == ViscositySwitch.NoneSwitch:
        return particleState.alphas, None
    elif switchConfig.scheme == ViscositySwitch.ReadHayfield2012:
        return computeReadHayfieldTerms(dt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    else:
        raise ValueError(f"Unsupported viscosity switch scheme: {switchConfig.scheme}")


def updateViscositySwitch(
        switchState: ViscositySwitchState,
        dt: float,
        dvdt: torch.Tensor,

        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    if switchConfig.scheme == ViscositySwitch.CullenDehnen2010:
        return computeCullenUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    elif switchConfig.scheme == ViscositySwitch.CullenHopkins:
        return computeHopkinsUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    elif switchConfig.scheme == ViscositySwitch.NoneSwitch:
        return particleState.alpha0s, None
    elif switchConfig.scheme == ViscositySwitch.ReadHayfield2012:
        return computeReadHayfieldUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
    else:
        raise ValueError(f"Unsupported viscosity switch scheme: {switchConfig.scheme}")