"""Wall terms of the delta+ stages for ANALYTIC boundaries (a boundary provider instead of boundary particles).

The provider (`boundary/provider.py`) returns geometry: the kernel integrals over the solid at the fluid
particles. This package owns the boundary-condition physics that turns them into the terms a boundary particle
contributes to each stage (wall continuity at stage 12, wall viscosity at 11, the pressure force of stage 13,
and, in the later patches, the detector, the shifting and the no-penetration impulse).
"""
from .noPenetration import analyticNoPenShift
from .shifting import wallShiftRaw, kernelAtSpacing, wallUChar, wallConcentrationGradient
from .detector import detectFreeSurfaceAnalytic, sym2LamPinv
from .wallTerms import (WallState, evaluateWall, resolveWall, wallContinuity, wallPressureAcceleration, wallViscousAcceleration, viscousPrefactor, wallLoads)

__all__ = ['analyticNoPenShift', 'wallShiftRaw', 'kernelAtSpacing', 'wallUChar', 'wallConcentrationGradient', 'detectFreeSurfaceAnalytic', 'sym2LamPinv', 'WallState', 'evaluateWall', 'resolveWall', 'wallContinuity', 'wallPressureAcceleration', 'wallViscousAcceleration', 'viscousPrefactor', 'wallLoads']
