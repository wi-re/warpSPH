"""Linear algebra on the GPU. The implementation lives in `warpSPHCore.linalg` (the core's own dense solves and the integrators' hook use it
too); this package re-exports it so `from warpSPH.linalg import ...` keeps working. See `docs/linalg/SOLVER_INVENTORY.md`."""

from warpSPHCore.linalg import *                                    # noqa: F401,F403
from warpSPHCore.linalg import __all__                              # noqa: F401
