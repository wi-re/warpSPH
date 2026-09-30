"""Registry-based dispatch of the viscosity switch chosen by ``schemeConfig.viscositySwitchParams.scheme``.

``computeViscositySwitchTerms`` computes the per-step alpha/switch state;
``updateViscositySwitch`` advances it over ``dt``. Both look the scheme up in
``SWITCHES`` (one ``(termsFn, updateFn)`` entry per scheme), so a new detector is one
registry entry, not two ``elif`` branches that can drift apart. ``NoneSwitch`` is a
no-op that passes ``particleState.alphas``/``alpha0s`` straight through with ``None``
switch state. Enum members without an implementation are listed in ``PLANNED`` with the
AV_PLAN phase that provides them and raise a ``NotImplementedError`` naming it.
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

__all__ = ['computeViscositySwitchTerms', 'updateViscositySwitch', 'SWITCHES', 'PLANNED',
           'registerViscositySwitch']


def _noneTerms(dt, particleState, simulationConfig, schemeConfig, supportScheme=None, adjacency=None):
    return particleState.alphas, None


def _noneUpdate(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                supportScheme=None, adjacency=None):
    return particleState.alpha0s, None


#: ``scheme -> (termsFn, updateFn)``. ``termsFn(dt, particleState, simulationConfig,
#: schemeConfig, supportScheme, adjacency) -> (alphas, ViscositySwitchState | None)``;
#: ``updateFn(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
#: supportScheme, adjacency) -> (alpha0s, ViscositySwitchState | None)`` (AV_PLAN,
#: "Registering a new switch", touch point 3).
SWITCHES = {
    ViscositySwitch.NoneSwitch: (_noneTerms, _noneUpdate),
    ViscositySwitch.CullenDehnen2010: (computeCullenTerms, computeCullenUpdate),
    ViscositySwitch.CullenHopkins: (computeHopkinsTerms, computeHopkinsUpdate),
    ViscositySwitch.ReadHayfield2012: (computeReadHayfieldTerms, computeReadHayfieldUpdate),
}

#: Enum members with no implementation yet -> where AV_PLAN.md provides them.
PLANNED = {
    ViscositySwitch.Balsara1995: 'AV_PLAN S3: the standalone Balsara multiplier (a limiter in [0,1], not an alpha)',
    ViscositySwitch.Colagrossi2004: 'AV_PLAN S3 (optional): the Colagrossi shear limiter',
    ViscositySwitch.MorrisMonaghan1997: 'AV_PLAN S3: the Morris & Monaghan (1997) switch',
    ViscositySwitch.Rosswog2000: 'AV_PLAN S3: the divergence-source switch (the 2020 entropy trigger is a separate, new Rosswog2020 member, Phase 2)',
}


def registerViscositySwitch(scheme: ViscositySwitch, termsFn, updateFn) -> None:
    """Add (or replace) a switch. A scheme may not be both registered and planned."""
    SWITCHES[scheme] = (termsFn, updateFn)
    PLANNED.pop(scheme, None)


def _resolve(scheme: ViscositySwitch):
    entry = SWITCHES.get(scheme)
    if entry is None:
        where = PLANNED.get(scheme, 'not registered in modules/shockCapturing/wrapper.py SWITCHES')
        raise NotImplementedError(f'Viscosity switch {scheme} is not implemented yet ({where}).')
    return entry


def computeViscositySwitchTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    termsFn, _ = _resolve(schemeConfig.viscositySwitchParams.scheme)
    return termsFn(dt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)


def updateViscositySwitch(
        switchState: ViscositySwitchState,
        dt: float,
        dvdt: torch.Tensor,

        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    _, updateFn = _resolve(schemeConfig.viscositySwitchParams.scheme)
    return updateFn(switchState, dt, dvdt, particleState, simulationConfig, schemeConfig,
                    supportScheme, adjacency)
