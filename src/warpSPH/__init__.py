"""warpSPH: a GPU-accelerated, differentiable SPH solver for delta-SPH/mDBC
research, built on NVIDIA Warp and PyTorch.

This file is the public-API aggregation point: it extends `__all__` by
re-exporting every subpackage's own `__all__` (`math`, `configurations`,
`systems`, `schemes`, `modules`, `utils`, `sample`, `geometry`,
`initializers`, `rigidBody`, `regions`, `io`), plus a hand-picked set of
enum types and case-setup helpers named individually below.
"""

__version__ = "0.5.0"

# from .casefile import argparse_defaults_from_casefile, build_configs_from_casefile, load_casefile
# from .shape_generation import populateSourceObstacleGridsStructured, sampleShapeStructured

__all__ = []
from .math import *
__all__.extend(math.__all__)
# from .utils import *
# __all__.extend(utils.__all__)

from .configurations import *
from .systems import *
from .schemes import *
from .modules import *
from .utils import *
from .enumTypes import EnergyScheme, AdaptiveSupportScheme, ViscositySwitch, CompressibleSPHScheme, WeaklyCompressibleSPHScheme, DensityDiffusionScheme, PressureForceScheme, IncompressibleSPHScheme, ArtificialCompressibleSPHScheme, PressureSmoothingScheme
from .sample import *

__all__.extend(configurations.__all__)
__all__.extend(systems.__all__)
__all__.extend(schemes.__all__)
__all__.extend(modules.__all__)
__all__.extend(utils.__all__)
# __all__.extend(enumTypes.__all__)
__all__.extend(sample.__all__)

from .geometry import *
__all__.extend(geometry.__all__)

__all__.extend(['EnergyScheme', 'AdaptiveSupportScheme', 'ViscositySwitch', 'CompressibleSPHScheme', 'WeaklyCompressibleSPHScheme', 'DensityDiffusionScheme', 'PressureForceScheme', 'IncompressibleSPHScheme', 'ArtificialCompressibleSPHScheme', 'PressureSmoothingScheme'])


from .io.parsers import parseKernelFunctions, parseIntegrationScheme, parseViscositySwitch, parseCompressibleSPHScheme, parseAdaptiveSupportScheme
__all__.extend(['parseKernelFunctions', 'parseIntegrationScheme', 'parseViscositySwitch', 'parseCompressibleSPHScheme', 'parseAdaptiveSupportScheme'])


from .sample.compressible import setupBasicCompressibleInitialState, sampleShockRegions1D

__all__.extend(['setupBasicCompressibleInitialState', 'sampleShockRegions1D'])

from .sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
__all__.extend(['setupBasicWeaklyCompressibleInitialState'])

from .initializers import *
from .rigidBody import *
from .regions import *
__all__.extend(initializers.__all__)
__all__.extend(rigidBody.__all__)
__all__.extend(regions.__all__)

from .io import *
__all__.extend(io.__all__)