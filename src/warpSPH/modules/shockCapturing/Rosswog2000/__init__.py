"""Rosswog, Davies, Thielemann & Piran (2000) divergence-source viscosity switch, A&A 360, 171,

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Rosswog2000 import ...` is the one import path.
"""

from .switch import *
from .switch import computeRosswog2000Terms, computeRosswog2000Update

from .switch import __all__
