"""Wadsley, Keller & Quinn (2017) Gasoline2 shock detector (MNRAS 471, 2357; `wadsley2017`), Eqs. (21)-(29):

The frontend (terms / update / step hook) is `switch.py`, its warp kernels are `kernels.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.Wadsley2017 import ...` is the one import path.
"""

from .switch import *
from .switch import computeWadsley2017Terms, computeWadsley2017Update, advanceWadsley2017, wadsleyD, wadsleyXi, wadsleyLocalAlpha, wadsleyDetector, wadsleyDetectorReference, WADSLEY_PREFACTOR

from .switch import __all__
