"""Public namespace for every config dataclass in warpSPH: the scheme-independent
`SimulationConfig` plus each scheme's own config (`CompressibleSPHConfig`,
`CompSPHConfig`, `CRKSPHConfig`, `WeaklyCompressibleSPHConfig`,
`IncompressibleSPHConfig`, `WaveEquationConfig`), the wave-equation case config, and `region.py`/
`rigidBody.py`. Also re-exports everything from `moduleConfigurations` (the
shared sub-configs those scheme configs embed, e.g. `DiffusionParameters`,
`ShiftProperties`, `SurfaceDetectionConfig`). This is the namespace most of the
rest of the package imports from (`from warpSPH.configurations import ...`),
including `schemes/builder.py`'s per-scheme `SchemeBundle` registry.
"""

from .simulationConfig import SimulationConfig, buildConfig, configurationToDict, dictToConfig
from .waveEquationConfig import ShapeSpec as WaveShapeSpec
from .waveEquationConfig import WaveSource, WaveBoundary
from .waveEquationConfig import CaseConfig as WaveCaseConfig
from .waveEquationConfig import WaveEquationConfig, waveEquationConfigToDict, dictToWaveEquationConfig
from .acousticCoreConfig import AcousticCoreConfig, acousticCoreConfigToDict, dictToAcousticCoreConfig
from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig
from .compSPHConfig import CompSPHConfig, compSPHConfigToDict, dictToCompSPHConfig
from .crkSPH import CRKViscosity, CRKSPHConfig, crkSPHConfigToDict, dictToCRKSPHConfig
from .moduleConfigurations.boundaryConditions import *
from .weaklyCompressible import WeaklyCompressibleSPHConfig, weaklyCompressibleConfigToDict, dictToWeaklyCompressibleConfig
from .region import RegionType, ParticleRegion
from .rigidBody import RigidBody
from .moduleConfigurations.surfaceDetection import SurfaceDetectionConfig, SurfaceDetectionScheme, NormalSource
from .incompressible import IncompressibleSPHConfig, incompressibleConfigToDict, dictToIncompressibleSPHConfig
from .artificialCompressible import (ArtificialCompressibilityParams,
                                     ArtificialCompressibleSPHConfig,
                                     buildDefaultArtificialCompressibilityParams,
                                     artificialCompressibleConfigToDict,
                                     dictToArtificialCompressibleConfig)



__all__ = [
    'SimulationConfig',
    'buildConfig',
    'configurationToDict',
    'dictToConfig',
    'WaveShapeSpec',
    'WaveSource',
    'WaveBoundary',
    'WaveCaseConfig',
    'WaveEquationConfig',
    'waveEquationConfigToDict',
    'dictToWaveEquationConfig',
    'AcousticCoreConfig',
    'acousticCoreConfigToDict',
    'dictToAcousticCoreConfig',
    'CompressibleSPHConfig',
    'CompSPHConfig',
    'CRKViscosity',
    'CRKSPHConfig',
    'BoundaryConditionType',
    'VectorProjectionType',
    'BoundaryCondition',
    'compressibleConfigToDict',
    'dictToCompressibleConfig',
    'compSPHConfigToDict',
    'dictToCompSPHConfig',
    'crkSPHConfigToDict',
    'dictToCRKSPHConfig',
    'WeaklyCompressibleSPHConfig',
    'weaklyCompressibleConfigToDict',
    'dictToWeaklyCompressibleConfig',
    'ArtificialCompressibilityParams',
    'ArtificialCompressibleSPHConfig',
    'buildDefaultArtificialCompressibilityParams',
    'artificialCompressibleConfigToDict',
    'dictToArtificialCompressibleConfig',
    'RegionType',
    'ParticleRegion',
    'RigidBody',
    'SurfaceDetectionConfig',
    'SurfaceDetectionScheme',
    'NormalSource',
    'IncompressibleSPHConfig',
    'incompressibleConfigToDict',
    'dictToIncompressibleSPHConfig',
]

from .moduleConfigurations import *
__all__.extend(moduleConfigurations.__all__)