"""mDBC (modified Dynamic Boundary Condition) ghost-particle machinery.

Density extrapolation from fluid to boundary/ghost particles
(`computeMdbcDensity`), per-material boundary velocity conditions
(`computeBoundaryVelocities`: zero/constant/no-slip/free-slip/extended), and
the no-penetration velocity-shift correction (`computeMdbcNoPenShift`).

(Pressure extrapolation for the removed `BoundaryPressureMode.mdbcMlsPressure`
used to live here too, as `computeMdbcPressure` -- removed in the pre-merge
cleanup pass, 09-04; `modules/incompressible/wallPressure.py`'s
`wallPressureExtrapolation` supersedes it. See
`DFSPH_IMPROVEMENT_PLAN.md`.)
"""

from .density2025 import computeMdbcDensity
from .velocity import computeBoundaryVelocities
from .wp_nopenshift import computeMdbcNoPenShift
__all__ = [
    "computeMdbcDensity",
    "computeBoundaryVelocities",
    "computeMdbcNoPenShift",
]