"""computeTimestep dispatcher.

Picks the weakly-compressible or (fully) compressible adaptive-timestep
implementation based on `isinstance(system, WeaklyCompressibleSystem)`; this
is the single entry point re-exported as `warpSPH.modules.timestep.computeTimestep`.

`ArtificialCompressibleSystem` gets its own branch (`artificialCompressible.py`,
De Courcy et al. 2024 Eq. 46): a different constraint set -- advective rather
than acoustic, `0.125 h^2/nu` viscous, and a symmetric [0.8, 1.2]x step-ratio
clamp that exists to protect BDF2 accuracy. Handing it the acoustic timestep
would build a `dt` on a sound speed the scheme does not have.

`MeshlessSystem` gets its own too (`meshless.py`): its CFL is the core backend's
per-particle Riemann signal speed `2 C h / v_sig` (Hopkins 2015 Eq. 24-25), not
the ideal-gas sound speed the compressible branch measures -- the kernel-xi
fallback would step on a speed the meshless flux does not carry.
"""

from warpSPH.utils.support import volumeToSupport

from ...systems.artificialCompressible import ArtificialCompressibleSystem
from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...systems.meshless import MeshlessSystem
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

__all__ = ['computeTimestep', 'defaultTimestep']


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
    if isinstance(system, MeshlessSystem):
        # Imported here: `.meshless` pulls in the `warpSPHCore.mfm` backend,
        # which a compressible / ACSPH / weakly-compressible dt never needs.
        from .meshless import computeTimestep as computeTimestepMeshless
        return computeTimestepMeshless(
            system=system, config=config, compParams=compParams, dt=dt)
    else:
        return computeTimestepCompressible(
            system = system,
            config = config,
            compParams = compParams,
            dt = dt,
        )


def defaultTimestep(ctx, state):
    """The runner's default `timestep` hook: the dispatcher's dt on the current
    state (the core signal dt for a meshless system, the acoustic CFL for the
    rest), so a run re-picks its step every step without the case spelling the
    hook out (COMPRESSIBLE_SPH_PLAN E2: the meshless schemes must track the
    signal speed, and their cases otherwise run at the setup dt for life)."""
    return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)