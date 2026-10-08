"""CRKSPH: pressure/viscosity forces and thermal energy using CRK-corrected kernels.

Pressure+artificial-viscosity acceleration (`accel`) and the matching internal
energy rate (`dudt`), both using CRK (Conservative Reproducing Kernel)
corrected kernel values/gradients and the van Leer / eta-based slope limiters
in `modules/reconstruction` (re-exported by `limiter`) to build a monotonic pseudo-viscosity.
"""

from .accel import computeCrkSPHAccelWarp
from .dudt import computeCrkSPHdudtWarp

__all__ = ['computeCrkSPHAccelWarp', 'computeCrkSPHdudtWarp']