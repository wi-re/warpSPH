"""computeTimestep dispatcher.

Picks the weakly-compressible or (fully) compressible adaptive-timestep
implementation based on `isinstance(system, WeaklyCompressibleSystem)`; this
is the single entry point re-exported as `warpSPH.modules.timestep.computeTimestep`.

`ArtificialCompressibleSystem` gets its own branch (`artificialCompressible.py`,
De Courcy et al. 2024 Eq. 46): a different constraint set -- advective rather
than acoustic, `0.125 h^2/nu` viscous, and a symmetric [0.8, 1.2]x step-ratio
clamp that exists to protect BDF2 accuracy. Handing it the acoustic timestep
would build a `dt` on a sound speed the scheme does not have.
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
from .artificialCompressible import computeTimestep as computeTimestepArtificialCompressible
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
        return computeTimestepArtificialCompressible(
            system=system, config=config, compParams=compParams, dt=dt,
            systemUpdate=systemUpdate)
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