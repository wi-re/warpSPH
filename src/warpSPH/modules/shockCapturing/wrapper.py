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
from .Balsara1995 import computeBalsaraTerms, computeBalsaraUpdate
from .Colagrossi2004 import computeColagrossiTerms, computeColagrossiUpdate
from .MorrisMonaghan1997 import computeMorrisMonaghanTerms, computeMorrisMonaghanUpdate
from .Rosswog2000 import computeRosswog2000Terms, computeRosswog2000Update
from .Rosswog2020 import computeRosswog2020Terms, computeRosswog2020Update, advanceRosswog2020
from .Sphenix2022 import computeSphenix2022Terms, computeSphenix2022Update, advanceSphenix2022
from .Wadsley2017 import computeWadsley2017Terms, computeWadsley2017Update, advanceWadsley2017

__all__ = ['computeViscositySwitchTerms', 'updateViscositySwitch', 'SWITCHES', 'PLANNED',
           'STEP_HOOKS', 'registerViscositySwitch', 'advanceViscositySwitchStep']


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
    ViscositySwitch.Balsara1995: (computeBalsaraTerms, computeBalsaraUpdate),
    ViscositySwitch.Colagrossi2004: (computeColagrossiTerms, computeColagrossiUpdate),
    ViscositySwitch.MorrisMonaghan1997: (computeMorrisMonaghanTerms, computeMorrisMonaghanUpdate),
    ViscositySwitch.Rosswog2000: (computeRosswog2000Terms, computeRosswog2000Update),
    ViscositySwitch.Rosswog2020: (computeRosswog2020Terms, computeRosswog2020Update),
    ViscositySwitch.Sphenix2022: (computeSphenix2022Terms, computeSphenix2022Update),
    ViscositySwitch.Wadsley2017: (computeWadsley2017Terms, computeWadsley2017Update),
}

#: Switches that update at step boundaries rather than per RHS stage:
#: ``scheme -> hook(state, stage0State, t, schemeConfig, simulationConfig)``, called from the compressible
#: systems' ``finalize`` (``advanceViscositySwitchStep``) on the step's final state, with the
#: RHS state of the step's first stage (the one evaluated at ``t = t^n``).
STEP_HOOKS = {
    ViscositySwitch.Rosswog2020: advanceRosswog2020,
    ViscositySwitch.Sphenix2022: advanceSphenix2022,
    ViscositySwitch.Wadsley2017: advanceWadsley2017,
}

#: Enum members with no implementation yet -> where AV_PLAN.md provides them (empty since
#: AV_PLAN S3).
PLANNED: dict = {}


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


def advanceViscositySwitchStep(system, initialSystem, returnValues, schemeConfig, simulationConfig=None) -> None:
    """Step-boundary switch update (``STEP_HOOKS``); a no-op for every per-stage switch."""
    params = getattr(schemeConfig, 'viscositySwitchParams', None)
    hook = STEP_HOOKS.get(params.scheme) if params is not None else None
    if hook is not None:
        hook(system.state, returnValues[0][1], float(initialSystem.t), schemeConfig, simulationConfig)
