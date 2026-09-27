"""Modules specific to artificial-compressibility SPH (De Courcy et al. 2024).

Deliberately thin: most of what ACSPH needs is the delta-SPH machinery with a
different field or a different prefactor, and lives where it already lived
(`modules/deltaSPH`, `modules/density`, `modules/pressure`, `modules/mdbc`).
Only what has no delta-SPH counterpart belongs here.
"""

from .pressureSmoothing import computePressureSmoothing

__all__ = ['computePressureSmoothing']
