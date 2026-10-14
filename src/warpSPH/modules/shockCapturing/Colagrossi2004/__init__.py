"""Colagrossi (2004) shear limiter, as a stand-alone viscosity factor.

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Colagrossi2004 import ...` is the one import path.
"""

from .switch import *
from .switch import computeColagrossiTerms, computeColagrossiUpdate

from .switch import __all__
