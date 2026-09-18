"""Particle-sampling layer: builds initial particle lattices/shapes (regular
grids, relaxed "optimal" layouts, spherical shells, split 2D regions) and
per-scheme initial conditions (compressible/weakly-compressible), called from
`warpSPH.caseUtils`/`warpSPH.cases`. `bySamplingScheme.sampleParticles` (not
re-exported here) is a separate dispatcher used only by
`warpSPH.systems.waveSystem`. `waveSystem.py` in this same directory finalizes
the wave-equation system and is documented separately.
"""

from ..geometry import PointCloud, ParticleSet
from ..utils.domain import DomainDescription
from ..utils.support import volumeToSupport
from ..math import getPeriodicPositions

from .regular import sampleRegularParticles
from .optimal import sampleOptimal
from .densest import sampleDensestParticles
from .shell import sampleShell, sampleShellv2
from .regions2D import sampleRegionSystem

__all__ = [
    'sampleRegularParticles', 'sampleOptimal', 'sampleDensestParticles',
    'sampleShell', 'sampleShellv2',

    'PointCloud', 'ParticleSet',

    'DomainDescription', 'volumeToSupport', 'getPeriodicPositions', 'sampleRegionSystem']