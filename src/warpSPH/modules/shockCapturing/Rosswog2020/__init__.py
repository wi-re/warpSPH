"""Rosswog (2020) entropy-based dissipation trigger, ApJ 898, 60 (`rosswog2020entropy`), Eqs. (15)-(20):

The frontend (terms / update / step hook) is `switch.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Rosswog2020 import ...` is the one import path.
"""

from .switch import *
from .switch import computeRosswog2020Terms, computeRosswog2020Update, advanceRosswog2020, smoothStep, entropyRate, desiredAlpha, decayAlpha

from .switch import __all__
