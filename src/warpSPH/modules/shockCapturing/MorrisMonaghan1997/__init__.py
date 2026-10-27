"""Morris & Monaghan (1997) viscosity switch, J. Comput. Phys. 136, 41, Eqs. (5), (6), (13).

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.MorrisMonaghan1997 import ...` is the one import path.
"""

from .switch import *
from .switch import computeMorrisMonaghanTerms, computeMorrisMonaghanUpdate, relaxedAlpha

from .switch import __all__
