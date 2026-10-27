"""Momentum-equation source term: -rho * div(v), in three variants.

`consistent` applies gradient-renormalized/grad-h-corrected divergence;
`inconsistent` uses the plain (non-renormalized) super-symmetric SPH
divergence; `incompressible` uses a fixed rest density and intermediate
advection velocities instead of the current velocity field.
"""

from .consistent import computeMomentumConsistent_warp as computeMomentumConsistent
from .inconsistent import computeMomentum 

__all__ = ['computeMomentumConsistent', 'computeMomentum']