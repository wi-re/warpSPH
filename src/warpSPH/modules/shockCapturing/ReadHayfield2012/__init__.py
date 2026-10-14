"""Read & Hayfield (2012) SPHS shock capturer (artificial conductivity +

The frontend (terms / update / step hook) is `switch.py`, its warp kernels are `kernels.py`; this package re-exports the switch's names so
`from warpSPH.modules.shockCapturing.ReadHayfield2012 import ...` is the one import path.
"""

from .switch import *
from .switch import computeReadHayfieldTerms, computeReadHayfieldUpdate

from .switch import __all__
