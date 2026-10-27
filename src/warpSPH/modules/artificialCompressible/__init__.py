"""Modules specific to artificial-compressibility SPH (De Courcy et al. 2024).

Deliberately thin: most of what ACSPH needs is the delta-SPH machinery with a
different field or a different prefactor, and lives where it already lived
(`modules/deltaSPH`, `modules/density`, `modules/pressure`, `modules/mdbc`).
Only what has no delta-SPH counterpart belongs here.
"""

from .dualTime import (DualTimeProblem, DualTimeResult, bdfHistory, exactDeltaUpdate, iterate,
                       makeDualTimeProblem, solveDualTime, zeroNonFluidRows)
from .forcing import bodyForces
from .parameters import acParameters, convergenceMetric, convergenceMetricDevice
from .pressureSmoothing import computePressureSmoothing
from .residual import spatialResidual
from .surfaceFlags import SurfaceFlags, detectSurface, diffusionTaper
from .wallClosure import (ContactMasks, WallBounce, contactMasks, makeWallBounce, bounceVelocities, projectPressure, unilateralWallDivergence,
                          unilateralWallForce, wallAdjacent, wallBounce, wallBounceSetup, wallPressures)
from .workingState import workingState

__all__ = ['computePressureSmoothing', 'DualTimeProblem', 'DualTimeResult', 'iterate', 'makeDualTimeProblem', 'solveDualTime', 'exactDeltaUpdate',
           'zeroNonFluidRows', 'bdfHistory', 'bodyForces', 'acParameters', 'convergenceMetric', 'convergenceMetricDevice',
           'spatialResidual', 'SurfaceFlags',
           'detectSurface', 'diffusionTaper', 'ContactMasks', 'WallBounce', 'contactMasks',
           'makeWallBounce', 'bounceVelocities', 'projectPressure', 'unilateralWallDivergence', 'unilateralWallForce', 'wallAdjacent', 'wallBounce', 'wallBounceSetup', 'wallPressures',
           'workingState']
