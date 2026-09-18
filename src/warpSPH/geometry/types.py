"""Geometry-side particle/point-cloud data types: `ParticleSet` (a sampled
region's positions/supports/masses/densities), `SamplingScheme` (the
regular/jittered/glass/optimal/random enum consumed by
`sample.bySamplingScheme`), and re-exports of `ParticleState`/`PointCloud`
from `warpSPHCore.dataTypes`.
"""

from ..utils.domain import *
import torch
from ..utils.support import volumeToSupport

from warpSPHCore.dataTypes import ParticleState, PointCloud

from typing import NamedTuple
class ParticleSet(NamedTuple):
    positions: torch.Tensor
    supports: torch.Tensor

    masses: torch.Tensor
    densities: torch.Tensor
    
# @torch.jit.script
# @dataclass#(slots=True)
# class PointCloud:
#     """
#     A named tuple containing the positions of the particles and the number of particles.
#     """
#     positions: torch.Tensor
#     supports: torch.Tensor

#     def __ne__(self, other: 'PointCloud') -> bool:
#         return not self.__eq__(other)
    


# from waves.utils.sampling import sampleRegularParticles
# from waves.utils.support import n_h_to_nH

from warpSPHCore import *

# from .wp_deltaShift import computeDeltaShiftWarp
# from waves.utils.sampling import ParticleSet


from ..utils.support import n_h_to_nH
# from ..config import SimulationConfig
from enum import Enum

class SamplingScheme(Enum):
    regular = 1
    jittered = 2
    glass = 3
    optimal = 4
    random = 5
    densest = 6


__all__ = ['ParticleState', 'ParticleSet', 'PointCloud', 'SamplingScheme']
