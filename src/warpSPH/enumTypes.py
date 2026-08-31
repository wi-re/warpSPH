"""Central enum registry for warpSPH: the scheme/kernel/BC/viscosity-switch
selectors used throughout `configurations/`, `modules/`, `schemes/`, and
`runner/` to pick a numerical formulation. Values are the strings/ints
stored in casefile YAML and parsed back into these enums by
`io.parsers`.
"""

from enum import Enum
import torch

__all__ = [
    'EnergyScheme',
    'AdaptiveSupportScheme',
    'ViscositySwitch',
    'CompressibleSPHScheme',
    'WeaklyCompressibleSPHScheme',
    'IncompressibleSPHScheme',
    'WaveEquationScheme',
    'EquationOfState',
    'DensityDiffusionScheme',
    'PressureForceScheme',
]

# @torch.jit.script
class EnergyScheme(Enum):
    equalWork = 0
    PdV = 1
    diminishing = 2
    monotonic = 3
    hybrid = 4
    CRK = 5

# @torch.jit.script
class AdaptiveSupportScheme(Enum):
    NoScheme = 0
    Monaghan = 1
    Owen = 2

# @torch.jit.script
class ViscositySwitch(Enum):
    Balsara1995 = 0
    Colagrossi2004 = 1
    CullenDehnen2010 = 2
    CullenHopkins = 3
    MorrisMonaghan1997 = 4
    Rosswog2000 = 5
    NoneSwitch = 6


# @torch.jit.script
class CompressibleSPHScheme(Enum):
    Monaghan = 0
    CompSPH = 1
    CRKSPH = 2

# @torch.jit.script
class WeaklyCompressibleSPHScheme(Enum):
    deltaSPH = 0

# @torch.jit.script
class IncompressibleSPHScheme(Enum):
    divergenceFree = 0
    #: Reference DFSPH (Bender & Koschier 2015/2017) as implemented in
    #: SPlisHSPlasH's `TimeStepDFSPH.cpp`: the constant-density and
    #: divergence-free corrections are applied to the *velocity* as warm-started
    #: pressure impulses, not as the momentum-neutral position shift the
    #: `divergenceFree` (VD+PS) scheme uses. Exists as the ground-truth
    #: comparison for the hydrostatic-column failure -- see
    #: `DFSPH_IMPROVEMENT_PLAN.md` Part 23 and `schemes/dfsphReference.py`.
    dfsphReference = 1
    #: Plain IISPH (Ihmsen et al. 2014): a single constant-density projection
    #: per step, applied as a velocity impulse -- no divergence-free pass, no
    #: VD+PS position shift. Shares `dfsphReference`'s body with the divergence
    #: solve off (`iisph_step`). Part 33 measured it as the first scheme in the
    #: codebase to hold `hydrostaticColumn`; it also holds `staticBlob` where
    #: the two-solve `dfsphReference` diverges.
    iisph = 2

# @torch.jit.script
class WaveEquationScheme(Enum):
    waveEquation = 0


class EquationOfState(Enum):
    stiffTait = "stiffTait"
    Tait = "Tait"
    isoThermal = "isoThermal"
    Polytropic = "polytropic"
    Murnaghan = "murnaghan"


class DensityDiffusionScheme(Enum):
    deltaSPH = 0
    denormalized = 1
    densityOnly = 2
    deltaOnly = 3
    denormalizedOnly = 4

class PressureForceScheme(Enum):
    conservative = 0
    nonConservative = 1
    Antuono = 2
    i = 3
    j = 4
    symmetric = 5