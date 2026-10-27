"""Hopkins' modified form of the Cullen & Dehnen (2010) viscosity switch (as used in CRKSPH / ATHENA).

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.CullenHopkins import ...` is the one import path.
"""

from .switch import *
from .switch import computeHopkinsTerms, computeHopkinsUpdate

from .switch import __all__
