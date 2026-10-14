"""Sphenix (Borrow et al. 2022, MNRAS 511, 2367; `borrow2022`) artificial-viscosity switch, Eqs. (21)-(24):

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Sphenix2022 import ...` is the one import path.
"""

from .switch import *
from .switch import computeSphenix2022Terms, computeSphenix2022Update, advanceSphenix2022, sphenixShockIndicator, sphenixLocalAlpha, sphenixAlphaUpdate

from .switch import __all__
