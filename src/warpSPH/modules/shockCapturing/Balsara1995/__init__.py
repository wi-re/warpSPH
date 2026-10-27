"""Balsara (1995) shear limiter as a standalone factor and as a viscosity "switch".

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Balsara1995 import ...` is the one import path.
"""

from .switch import *
from .switch import balsaraFactor, computeDivCurl, computeBalsaraTerms, computeBalsaraUpdate

from .switch import __all__
