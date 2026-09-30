"""Dissipative SPH terms: the artificial-viscosity "Pi" term and the
momentum-viscosity, thermal-conductivity and thermal-dissipation operators
built on top of it, used by the compressible schemes.
"""

from .wp_conductivity import computeConductivityWarp as computeConductivity
from .wp_diffusion import computeViscosityWarp as computeViscosity
from .wp_dissipation import computeThermalDissipationWarp as computeThermalDissipation
from .pi import computePi_actual
from .avPower import computeAVPowerSplit
from ...configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms


__all__ = ['computeConductivity', 'computeViscosity', 'computeThermalDissipation', 'computePi_actual', 'computeAVPowerSplit', 'DiffusionParameters', 'ViscosityTerms']