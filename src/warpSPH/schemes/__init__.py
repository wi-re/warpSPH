"""The step-function orchestration layer: one module per SPH scheme, each
wiring `modules/` compute functions into a single per-step function with the
signature `(system, dt, config, schemeConfig, verbose) -> update, adjacency,
...`. `builder.buildScheme` maps a scheme name/enum member to the matching
`SchemeBundle` (state/config/update classes plus this step function); nothing
here should be called by name without going through it. `waveEquation.py`
(`f_wave_equation`) is the exception among the fluid schemes: a non-fluid
demo scheme (see `cases/waveEquation.py` for its registered `Case`), but it
follows the same `SchemeBundle`/`buildScheme` contract as the rest.
"""

from .monaghan import compressibleSPH_Monaghan
from .compSPH import compSPH_step
from .crkSPH import crkSPH_step
from .divergenceFree import divergenceFree_step
from .waveEquation import f_wave_equation

from .builder import buildScheme, CompressibleSPHScheme

__all__ = ['f_wave_equation', 'compressibleSPH_Monaghan', 'compSPH_step', 'crkSPH_step', 'buildScheme', 'CompressibleSPHScheme', 'divergenceFree_step']