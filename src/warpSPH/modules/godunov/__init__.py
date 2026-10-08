"""Godunov SPH pair forces (GODUNOV_SPH_PLAN): the momentum and internal-energy rates from the Riemann problem between each pair.

`computeGodunovWarp` -- the simplified form of Cha & Whitworth (2003) Case 3 / Iwasaki & Inutsuka (2011) Eq. (24) / Puri &
Ramachandran (2014) Eq. (15), any kernel (layer 2 of the plan). Inutsuka's (2002) convolution form (Gaussian kernel, `V_ij^2`, interface
position `s*`) is layer 3.
"""

from .wp_gsph import computeGodunovWarp

__all__ = ['computeGodunovWarp']
