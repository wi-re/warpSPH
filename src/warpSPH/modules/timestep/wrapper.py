"""computeTimestep dispatcher.

Picks the weakly-compressible or (fully) compressible adaptive-timestep
implementation based on `isinstance(system, WeaklyCompressibleSystem)`; this
is the single entry point re-exported as `warpSPH.modules.timestep.computeTimestep`.

`ArtificialCompressibleSystem` is rejected rather than falling through to the
compressible branch: ACSPH's Eq. (46) is a different constraint set (advective
rather than acoustic, `0.125 h^2/nu` viscous, and a symmetric [0.8, 1.2]x growth
clamp that exists to protect BDF2 accuracy), and quietly handing it an acoustic
timestep built on a sound speed the scheme does not have would be wrong in a
way nothing downstream would notice. ACSPH_PLAN.md step 6 replaces the raise.
"""

from warpSPH.utils.support import volumeToSupport

from ...systems.artificialCompressible import ArtificialCompressibleSystem
from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Any, Optional
from warpSPHCore import *
from ...modules.eos import idealGasEOS
import torch
import warp as wp

from .compressible import computeTimestep as computeTimestepCompressible
from .weaklyCompressible import computeTimestep as computeTimestepWeaklyCompressible

__all__ = ['computeTimestep']


def computeTimestep(
    system: Any,
    config: SimulationConfig,
    compParams: Any,
    dt: Optional[float] = None,
    systemUpdate: Optional[WeaklyCompressibleSystemUpdate] = None,
):
    if isinstance(system, ArtificialCompressibleSystem):
        raise NotImplementedError(
            "computeTimestep has no artificial-compressibility branch yet -- see "
            "this module's docstring and ACSPH_PLAN.md Sec. 4.5 / step 6. Drive "
            "the case with a fixed `--dt` until it lands.")
    if isinstance(system, WeaklyCompressibleSystem):
        return computeTimestepWeaklyCompressible(
            system = system,
            config = config,
            compParams = compParams,
            dt = dt,
            systemUpdate = systemUpdate
        )
    else:
        return computeTimestepCompressible(
            system = system,
            config = config,
            compParams = compParams,
            dt = dt,
        )