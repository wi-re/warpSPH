"""Aggregates every scheme's `dataclass` state/system(/update) triad into one
`warpSPH.systems` namespace, built on `warpSPHIntegrators`'s `BaseState`/
`BaseIntegrationSystem` framework. `baseState`'s `BaseParticleState`/
`BaseSystemUpdate`/`BaseSystem` is the shared skeleton the field layout of
every other module here extends. `systems/waveSystem.py`'s wave-equation
triad is documented separately (not a fluid scheme). `systems/incompressible.py`
(the divergence-free/DFSPH state) is deliberately not re-exported here --
`schemes/divergenceFree.py` and `schemes/builder.py` import it directly from that
submodule instead.
"""

from .baseState import BaseParticleState, BaseSystemUpdate, BaseSystem
from .compressibleMonaghan import CompressibleState, CompressibleSystemUpdate, CompressibleSystem
from .waveSystem import WaveSystemStatev3, WaveSystemUpdatev3, WaveSystemv3
from .compSPH import CompSPHState, CompSPHSystem
from .weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from .acousticCore import AcousticCoreState, AcousticCoreSystemUpdate, AcousticCoreSystem
from .artificialCompressible import (ArtificialCompressibleState,
                                     ArtificialCompressibleSystemUpdate,
                                     ArtificialCompressibleSystem,
                                     bdfCoefficients)

__all__ = [
    'BaseParticleState', 'BaseSystemUpdate', 'BaseSystem',
    'CompressibleState', 'CompressibleSystemUpdate', 'CompressibleSystem',
    'WaveSystemStatev3', 'WaveSystemUpdatev3', 'WaveSystemv3',
    'CompSPHState', 'CompSPHSystem',
    'WeaklyCompressibleState', 'WeaklyCompressibleSystem', 'WeaklyCompressibleSystemUpdate',
    'AcousticCoreState', 'AcousticCoreSystemUpdate', 'AcousticCoreSystem',
    'ArtificialCompressibleState', 'ArtificialCompressibleSystemUpdate',
    'ArtificialCompressibleSystem', 'bdfCoefficients',
]