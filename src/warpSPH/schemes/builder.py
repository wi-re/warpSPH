"""Resolve a scheme name/enum member to its `SchemeBundle`: the state, config,
and update classes plus step/export/import functions for one of the six
registered SPH schemes (Monaghan, CompSPH, CRKSPH, deltaSPH,
divergence-free/DFSPH, and the non-fluid wave equation). `_divergenceFree`
imports `dfsph` lazily to avoid a circular import with `schemes/__init__.py`,
which imports `dfsph` first.
"""

from dataclasses import dataclass
from typing import Callable, Union

from ..systems import (
    CompSPHState, CompSPHSystem, CompressibleState,
    CompressibleSystem, CompressibleSystemUpdate, WeaklyCompressibleState,
    WeaklyCompressibleSystem, WaveSystemStatev3, WaveSystemUpdatev3, WaveSystemv3,
)
from ..configurations import (
    CRKSPHConfig, CompSPHConfig, CompressibleSPHConfig,
    IncompressibleSPHConfig, WeaklyCompressibleSPHConfig, WaveEquationConfig,
    compSPHConfigToDict, compressibleConfigToDict, crkSPHConfigToDict,
    dictToCRKSPHConfig, dictToCompSPHConfig, dictToCompressibleConfig,
    dictToIncompressibleSPHConfig, dictToWeaklyCompressibleConfig,
    dictToWaveEquationConfig, incompressibleConfigToDict,
    waveEquationConfigToDict, weaklyCompressibleConfigToDict,
)
from .compSPH import compSPH_step
from .deltaSPH import deltaSPH_step
from .crkSPH import crkSPH_step
from .monaghan import compressibleSPH_Monaghan
from .waveEquation import f_wave_equation
from ..enumTypes import (
    CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme,
    WaveEquationScheme,
)

__all__ = ['SchemeBundle', 'buildScheme']


#: The order `buildScheme` used to return as a bare tuple. Frozen at these seven
#: members on purpose: `SchemeBundle.__iter__` walks this list, not
#: `dataclasses.fields()`, so a new field can be added to the bundle without
#: shifting what positional unpacking at an old call site binds.
_LEGACY_TUPLE_ORDER = (
    'SimulationSystem', 'SimulationState', 'SimulationConfig', 'SimulationUpdate',
    'stepFunction', 'exportFunction', 'importFunction',
)


@dataclass(frozen=True)
class SchemeBundle:
    """Everything that varies between SPH schemes, named.

    `buildScheme` returned these seven as a positional tuple, unpacked verbatim
    at every call site. That made an eighth member -- the tangent-propagation
    function forward-mode AD will need -- a breaking change everywhere at once.
    Named fields make additions additive; `__iter__` keeps the old unpacking
    working meanwhile.
    """

    #: The `SimulationSystem` class: geometry, adjacency, and the state it owns.
    SimulationSystem: type
    #: The per-step state class the system hands out.
    SimulationState: type
    #: The scheme's own config dataclass, instantiated with no arguments.
    SimulationConfig: type
    #: The update class the integrator accumulates into.
    SimulationUpdate: type
    #: `(system, dt, config, schemeConfig, verbose) -> update`.
    stepFunction: Callable
    #: Scheme config -> plain dict, for HDF5/JSON export.
    exportFunction: Callable
    #: The inverse of `exportFunction`.
    importFunction: Callable

    def __iter__(self):
        """Legacy 7-tuple unpacking. Pinned to `_LEGACY_TUPLE_ORDER`."""
        return iter(tuple(getattr(self, name) for name in _LEGACY_TUPLE_ORDER))

    def __len__(self):
        return len(_LEGACY_TUPLE_ORDER)


def _monaghan() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=CompressibleSystem,
        SimulationState=CompressibleState,
        SimulationConfig=CompressibleSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=compressibleSPH_Monaghan,
        exportFunction=compressibleConfigToDict,
        importFunction=dictToCompressibleConfig,
    )


def _compSPH() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=CompSPHSystem,
        SimulationState=CompSPHState,
        SimulationConfig=CompSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=compSPH_step,
        exportFunction=compSPHConfigToDict,
        importFunction=dictToCompSPHConfig,
    )


def _crkSPH() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=CompSPHSystem,
        SimulationState=CompSPHState,
        SimulationConfig=CRKSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=crkSPH_step,
        exportFunction=crkSPHConfigToDict,
        importFunction=dictToCRKSPHConfig,
    )


def _deltaSPH() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=WeaklyCompressibleSystem,
        SimulationState=WeaklyCompressibleState,
        SimulationConfig=WeaklyCompressibleSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=deltaSPH_step,
        exportFunction=weaklyCompressibleConfigToDict,
        importFunction=dictToWeaklyCompressibleConfig,
    )


def _divergenceFree() -> SchemeBundle:
    # Deferred, as the original branch was: it keeps `builder` from importing
    # `dfsph` at its own import time. (`schemes/__init__` imports `dfsph` first
    # in practice, so this is about the dependency edge, not about saving the
    # load.) The config codecs come from `..configurations`, which is where they
    # are defined -- `dfsph` used to relay them only as a side effect of star
    # importing that package.
    from .dfsph import dfsph_step
    from ..systems.incompressible import (IncompressibleSystem, IncompressibleState,
                                          IncompressibleSystemUpdate)
    return SchemeBundle(
        SimulationSystem=IncompressibleSystem,
        SimulationState=IncompressibleState,
        SimulationConfig=IncompressibleSPHConfig,
        SimulationUpdate=IncompressibleSystemUpdate,
        stepFunction=dfsph_step,
        exportFunction=incompressibleConfigToDict,
        importFunction=dictToIncompressibleSPHConfig,
    )


def _dfsphReference() -> SchemeBundle:
    # Same lazy-import rationale as `_divergenceFree`. Reuses the incompressible
    # state/config/update and their codecs -- the only thing that differs is the
    # step function and the system's `finalize` (see
    # `systems/incompressible.py::DFSPHReferenceSystem` and
    # `schemes/dfsphReference.py`).
    from .dfsphReference import dfsphReference_step
    from ..systems.incompressible import (DFSPHReferenceSystem, IncompressibleState,
                                          IncompressibleSystemUpdate)
    return SchemeBundle(
        SimulationSystem=DFSPHReferenceSystem,
        SimulationState=IncompressibleState,
        SimulationConfig=IncompressibleSPHConfig,
        SimulationUpdate=IncompressibleSystemUpdate,
        stepFunction=dfsphReference_step,
        exportFunction=incompressibleConfigToDict,
        importFunction=dictToIncompressibleSPHConfig,
    )


def _iisph() -> SchemeBundle:
    # Plain IISPH ([I], Ihmsen et al. 2014): shares `dfsphReference`'s
    # incompressible state/config/update, codecs and `DFSPHReferenceSystem`
    # (the whole time integration happens inside the step), differing only in
    # the step function -- the divergence-free pass is switched off. See
    # `schemes/dfsphReference.py::iisph_step` and DFSPH_IMPROVEMENT_PLAN.md
    # Part 33.
    from .dfsphReference import iisph_step
    from ..systems.incompressible import (DFSPHReferenceSystem, IncompressibleState,
                                          IncompressibleSystemUpdate)
    return SchemeBundle(
        SimulationSystem=DFSPHReferenceSystem,
        SimulationState=IncompressibleState,
        SimulationConfig=IncompressibleSPHConfig,
        SimulationUpdate=IncompressibleSystemUpdate,
        stepFunction=iisph_step,
        exportFunction=incompressibleConfigToDict,
        importFunction=dictToIncompressibleSPHConfig,
    )


def _waveEquation() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=WaveSystemv3,
        SimulationState=WaveSystemStatev3,
        SimulationConfig=WaveEquationConfig,
        SimulationUpdate=WaveSystemUpdatev3,
        stepFunction=f_wave_equation,
        exportFunction=waveEquationConfigToDict,
        importFunction=dictToWaveEquationConfig,
    )


#: enum member -> the factory that builds its bundle.
_SCHEMES = {
    CompressibleSPHScheme.Monaghan: _monaghan,
    CompressibleSPHScheme.CompSPH: _compSPH,
    CompressibleSPHScheme.CRKSPH: _crkSPH,
    WeaklyCompressibleSPHScheme.deltaSPH: _deltaSPH,
    IncompressibleSPHScheme.divergenceFree: _divergenceFree,
    IncompressibleSPHScheme.dfsphReference: _dfsphReference,
    IncompressibleSPHScheme.iisph: _iisph,
    WaveEquationScheme.waveEquation: _waveEquation,
}

#: Lower-cased string spellings. Every enum member answers to its own name; the
#: extra entries are the spellings the notebooks and older scripts already use,
#: which do not all match the member name (`Monaghan` vs the string the examples
#: pass, `'MonaghanCompressibleSPH'`).
_ALIASES = {member.name.lower(): member for member in _SCHEMES}
_ALIASES['monaghancompressiblesph'] = CompressibleSPHScheme.Monaghan


def buildScheme(
    schemeName: Union[str, CompressibleSPHScheme, WeaklyCompressibleSPHScheme,
                      IncompressibleSPHScheme, WaveEquationScheme]
) -> SchemeBundle:
    """Resolve a scheme name or enum member to its :class:`SchemeBundle`."""
    if isinstance(schemeName, str):
        member = _ALIASES.get(schemeName.lower())
    else:
        member = schemeName if schemeName in _SCHEMES else None

    if member is None:
        raise ValueError(
            f'Scheme {schemeName!r} not recognized. Valid schemes are: '
            f'{sorted(_ALIASES)}.'
        )
    return _SCHEMES[member]()
