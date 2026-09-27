"""Particle shifting technique (PST): anti-clustering position correction,
with optional free-surface projection of the shift. The shift schemes
(`ShiftingScheme` in `configurations.moduleConfigurations.shifting`) are:
explicit delta-SPH (`computeDeltaShiftWarp`/`computeDeltaShift`); implicit
particle shifting (`computeImplicitShift`), a matrix-free Krylov
(BiCGStab/GMRES) solve over the neighbor graph; `dynamic`
(`computeDynamicImplicitShift`), the same implicit path with the opt-in
inner-solve fallback chain (`ShiftProperties.implicitFallback`) enabled by
default -- the improved scheme, while `implicit` stays byte-identical for
legacy users; and `michel2022` (`computeMichelShift`, `PST_ALE_PLAN.md`
Part 5.1), Michel et al. 2022's consistent, Mach-free law -- the only one of
these usable by ACSPH, which has no sound speed.
"""

from .delta import computeDeltaShiftWarp
from .implicitShifting import computeImplicitShift, computeDynamicImplicitShift
from .michel import computeMichelShift
from .wp_michelUChar import computeUCharWarp
from .wrapper import solveShifting

__all__ = [
    'computeDeltaShiftWarp',
    'computeImplicitShift',
    'computeDynamicImplicitShift',
    'computeMichelShift',
    'computeUCharWarp',
    'solveShifting'
]