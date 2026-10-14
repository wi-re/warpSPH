"""Cullen & Dehnen (2010) time-dependent artificial-viscosity switch.

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.CullenDehnen2010 import ...` is the one import path.
"""

from .switch import *
from .switch import computeSecondOrderV, computeR, computeXi, compute_vsig, computeCullenTerms, computeCullenUpdate

from .switch import __all__
