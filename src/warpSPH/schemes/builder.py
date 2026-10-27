"""Resolve supported schemes to state/config/update, driver and codec bundles."""

from warpSPHIntegrators.schemes import ProjectedStep, IntegrationSchemeType
from dataclasses import dataclass, replace
from typing import Callable, Union

from ..systems import (
    CompSPHState, CompSPHSystem, CompressibleState,
    CompressibleSystem, CompressibleSystemUpdate, MeshlessState,
    MeshlessSystem, MeshlessSystemUpdate, WeaklyCompressibleState,
    WeaklyCompressibleSystem, WaveSystemStatev3, WaveSystemUpdatev3, WaveSystemv3,
)
from ..configurations import (
    CRKSPHConfig, CompSPHConfig, CompressibleSPHConfig, GSPHConfig, InutsukaGSPHConfig,
    IncompressibleSPHConfig, MFMConfig, MFVConfig, MeshlessConfig,
    WeaklyCompressibleSPHConfig, Sun2017DeltaSPHConfig,
    WaveEquationConfig,
    compSPHConfigToDict, compressibleConfigToDict, crkSPHConfigToDict,
    dictToCRKSPHConfig, dictToCompSPHConfig, dictToCompressibleConfig, dictToGSPHConfig, dictToInutsukaGSPHConfig, gsphConfigToDict,
    dictToIncompressibleSPHConfig, dictToMeshlessConfig, dictToWeaklyCompressibleConfig,
    dictToWaveEquationConfig, incompressibleConfigToDict, meshlessConfigToDict,
    waveEquationConfigToDict, weaklyCompressibleConfigToDict,
)
from .compressible.compSPH import compSPH_step
from .weaklyCompressible.deltaSPH import deltaSPH_step
from .compressible.crkSPH import crkSPH_step
from .compressible.monaghan import compressibleSPH_Monaghan
from .compressible.gsph import compressibleSPH_GSPH
from .compressible.gsphInutsuka import compressibleSPH_InutsukaGSPH
from .compressible.meshless import mfmStep, mfvStep
from .pde.waveEquation import f_wave_equation
from ..enumTypes import (
    ArtificialCompressibleSPHScheme, CompressibleSPHScheme,
    WeaklyCompressibleSPHScheme, IncompressibleSPHScheme, WaveEquationScheme,
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
    #: Optional driver constructor or registered scheme enum; legacy unpacking is unchanged.
    integrationMethod: Callable | IntegrationSchemeType | None = None
    #: Resolve an integrated-field layout after scheme config overrides.
    stateFactory: Callable | None = None

    def configured(self, config):
        return self if self.stateFactory is None else replace(self, SimulationState=self.stateFactory(config))

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


def _pesph() -> SchemeBundle:
    from ..configurations.pesph import PESPHConfig, pesphConfigToDict, dictToPESPHConfig
    from ..systems.pesph import PESPHSystem, PressureEnergyState, PESPHSystemUpdate, pesphStateType
    from .compressible.pesph import pesphStep
    return SchemeBundle(PESPHSystem, PressureEnergyState, PESPHConfig, PESPHSystemUpdate,
                        pesphStep, pesphConfigToDict, dictToPESPHConfig,
                        stateFactory=lambda config: pesphStateType(config.variant))


def _gsph() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=CompressibleSystem,
        SimulationState=CompressibleState,
        SimulationConfig=GSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=compressibleSPH_GSPH,
        exportFunction=gsphConfigToDict,
        importFunction=dictToGSPHConfig,
    )


def _inutsukaGSPH() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=CompressibleSystem,
        SimulationState=CompressibleState,
        SimulationConfig=InutsukaGSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=compressibleSPH_InutsukaGSPH,
        exportFunction=gsphConfigToDict,
        importFunction=dictToInutsukaGSPHConfig,
    )


def _mfm() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=MeshlessSystem,
        SimulationState=MeshlessState,
        SimulationConfig=MFMConfig,
        SimulationUpdate=MeshlessSystemUpdate,
        stepFunction=mfmStep,
        exportFunction=meshlessConfigToDict,
        importFunction=dictToMeshlessConfig,
    )


def _mfv() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=MeshlessSystem,
        SimulationState=MeshlessState,
        SimulationConfig=MFVConfig,
        SimulationUpdate=MeshlessSystemUpdate,
        stepFunction=mfvStep,
        exportFunction=meshlessConfigToDict,
        importFunction=dictToMeshlessConfig,
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


def _sun2017DeltaSPH() -> SchemeBundle:
    # Same state/update/step/codecs as `deltaSPH` -- Sun et al. 2017's
    # prescription is a parameter choice (`freezeDiffusionAcrossStages=True`),
    # not different physics code. See `Sun2017DeltaSPHConfig`'s docstring and
    # `DELTASPH_VALIDATION_PLAN.md` Part 5.1.
    return SchemeBundle(
        SimulationSystem=WeaklyCompressibleSystem,
        SimulationState=WeaklyCompressibleState,
        SimulationConfig=Sun2017DeltaSPHConfig,
        SimulationUpdate=CompressibleSystemUpdate,
        stepFunction=deltaSPH_step,
        exportFunction=weaklyCompressibleConfigToDict,
        importFunction=dictToWeaklyCompressibleConfig,
    )


def _dfsph(ordering='df_np_cd') -> SchemeBundle:
    from .incompressible.dfsph import dfsph_step
    from ..configurations.incompressible import PredictorDFSPHConfig, ShiftDFSPHConfig
    configType = {'df_np_cd': IncompressibleSPHConfig, 'np_df_cd': PredictorDFSPHConfig,
                  'np_df_shift': ShiftDFSPHConfig}[ordering]
    from ..systems.incompressible import DFSPHState, DFSPHSystem, IncompressibleSystemUpdate
    return SchemeBundle(
        SimulationSystem=DFSPHSystem,
        SimulationState=DFSPHState,
        SimulationConfig=configType,
        SimulationUpdate=IncompressibleSystemUpdate,
        integrationMethod=ProjectedStep,
        stepFunction=dfsph_step,
        exportFunction=incompressibleConfigToDict,
        importFunction=dictToIncompressibleSPHConfig,
    )


def _iisph() -> SchemeBundle:
    # Native IISPH (Ihmsen et al. 2014), one rest-volume density projection.
    # Shares state/time infrastructure, not DFSPH's coefficients or policy.
    from .incompressible.iisph import iisph_step
    from ..systems.incompressible import (IISPHSystem, IncompressibleState,
                                          IncompressibleSystemUpdate)
    return SchemeBundle(
        SimulationSystem=IISPHSystem,
        SimulationState=IncompressibleState,
        SimulationConfig=IncompressibleSPHConfig,
        SimulationUpdate=IncompressibleSystemUpdate,
        integrationMethod=IntegrationSchemeType.semiImplicitEuler,
        stepFunction=iisph_step,
        exportFunction=incompressibleConfigToDict,
        importFunction=dictToIncompressibleSPHConfig,
    )


def _artificialCompressible() -> SchemeBundle:
    # Import the dual-time implementation lazily. See
    # `schemes/incompressible/artificialCompressible.py` -- this scheme owns its whole real
    # step (a dual-time solve) and hands the integrator an exact delta, which
    # is why it pins `forwardEuler` and validates it at step entry.
    from .incompressible.artificialCompressible import artificialCompressible_step
    from ..systems.artificialCompressible import (ArtificialCompressibleState,
                                                  ArtificialCompressibleSystem,
                                                  ArtificialCompressibleSystemUpdate)
    from ..configurations.artificialCompressible import (
        ArtificialCompressibleSPHConfig, artificialCompressibleConfigToDict,
        dictToArtificialCompressibleConfig)
    return SchemeBundle(
        SimulationSystem=ArtificialCompressibleSystem,
        SimulationState=ArtificialCompressibleState,
        SimulationConfig=ArtificialCompressibleSPHConfig,
        SimulationUpdate=ArtificialCompressibleSystemUpdate,
        stepFunction=artificialCompressible_step,
        exportFunction=artificialCompressibleConfigToDict,
        importFunction=dictToArtificialCompressibleConfig,
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
    CompressibleSPHScheme.GSPH: _gsph,
    CompressibleSPHScheme.InutsukaGSPH: _inutsukaGSPH,
    CompressibleSPHScheme.PESPH: _pesph,
    CompressibleSPHScheme.MFM: _mfm,
    CompressibleSPHScheme.MFV: _mfv,
    WeaklyCompressibleSPHScheme.deltaSPH: _deltaSPH,
    WeaklyCompressibleSPHScheme.sun2017DeltaSPH: _sun2017DeltaSPH,
    IncompressibleSPHScheme.dfsph: _dfsph,
    IncompressibleSPHScheme.dfsphPredictor: lambda: _dfsph('np_df_cd'),
    IncompressibleSPHScheme.dfsphShift: lambda: _dfsph('np_df_shift'),
    IncompressibleSPHScheme.iisph: _iisph,
    ArtificialCompressibleSPHScheme.artificialCompressible: _artificialCompressible,
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
                      IncompressibleSPHScheme, ArtificialCompressibleSPHScheme,
                      WaveEquationScheme]
) -> SchemeBundle:
    """Resolve a scheme name or enum member to its :class:`SchemeBundle`."""
    rejectRetiredScheme(schemeName)
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


RETIRED_SCHEMES = {
    'divergencefree': 'dfsphShift (native VD+PS)',
    'dfsphreference': 'dfsph (classic DFSPH)',
    'omniincompressible': 'dfsphPredictor (predictor-first DFSPH)',
    'band2018pb': "dfsph with dfsph.pressureBoundary='band2018' (experimental, static sampled walls only)",
}


def rejectRetiredScheme(name):
    replacement = RETIRED_SCHEMES.get(str(getattr(name, 'name', name)).lower())
    if replacement is not None:
        raise ValueError(f"Scheme {name!r} is retired; select {replacement} explicitly. "
                         "Native operators/policies differ; old checkpoints cannot be silently migrated. "
                         "See docs/incompressible/MIGRATION.md.")
