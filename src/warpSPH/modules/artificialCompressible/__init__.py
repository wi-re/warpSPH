"""Modules specific to artificial-compressibility SPH (De Courcy et al. 2024).

Deliberately thin: most of what ACSPH needs is the delta-SPH machinery with a
different field or a different prefactor, and lives where it already lived
(`modules/deltaSPH`, `modules/density`, `modules/pressure`, `modules/mdbc`).
Only what has no delta-SPH counterpart belongs here.
"""

from .dualTime import DualTimeResult, DualTimeSolver, bdfHistory, exactDeltaUpdate, zeroNonFluidRows
from .forcing import bodyForces
from .parameters import (EXACT_DELTA_INTEGRATORS, RECOMMENDED_CFL_TAU, TABLEAUS, acParameters,
                         convergenceMetric, convergenceMetricDevice, validateAcParams,
                         validateIntegrationScheme)
from .pressureSmoothing import computePressureSmoothing
from .residual import spatialResidual
from .surfaceFlags import SurfaceFlags, detectSurface, diffusionTaper
from .wallClosure import (ContactMasks, WallBounce, contactMasks, makeWallBounce, unilateralWallDivergence,
                          unilateralWallForce, wallAdjacent, wallBounce, wallBounceSetup, wallPressures)
from .workingState import workingState

__all__ = ['computePressureSmoothing', 'DualTimeResult', 'DualTimeSolver', 'exactDeltaUpdate',
           'zeroNonFluidRows', 'bdfHistory', 'bodyForces', 'EXACT_DELTA_INTEGRATORS', 'RECOMMENDED_CFL_TAU',
           'TABLEAUS', 'acParameters', 'convergenceMetric', 'convergenceMetricDevice',
           'validateAcParams', 'validateIntegrationScheme', 'spatialResidual', 'SurfaceFlags',
           'detectSurface', 'diffusionTaper', 'ContactMasks', 'WallBounce', 'contactMasks',
           'makeWallBounce', 'unilateralWallDivergence', 'unilateralWallForce', 'wallAdjacent', 'wallBounce', 'wallBounceSetup', 'wallPressures',
           'workingState']
