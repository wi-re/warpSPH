"""Adaptive timestep computation, compressible and weakly-compressible.

`computeTimestep` (from `wrapper`) dispatches to the compressible or
weakly-compressible implementation by the runtime type of `system`, each
combining CFL/viscous/acoustic/acceleration constraints and clamping to
[minDt, maxDt] with a growth-rate limit. `setupWeaklyCompressibleTimestep`
additionally back-solves the weakly-compressible EOS sound speed from a
target dt/dx/neighbor-count.
"""

from .wrapper import computeTimestep, defaultTimestep
from .weaklyCompressible import setupWeaklyCompressibleTimestep

__all__ = [
    "computeTimestep",
    "defaultTimestep",
    "setupWeaklyCompressibleTimestep"
]