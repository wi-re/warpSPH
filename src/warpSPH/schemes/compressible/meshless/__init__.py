"""The meshless finite-mass (MFM) and finite-volume (MFV) schemes of Hopkins (2015), on the `warpSPHCore.mfm` backend (COMPRESSIBLE_SPH_PLAN E1 + D4).

* `mfm.py`      `mfmStep`  -- conserved face fluxes, dm = 0.
* `mfv.py`      `mfvStep`  -- the same fluxes plus the GIZMO mass-flux limiter (dm != 0), and the experimental GIZMO primitive update.
* `step.py`     `runMeshlessStep` -- the skeleton both share: integrator guard, geometry, primitives and supports, temporal scheme, the update.
* `temporal.py` the temporal schemes ('hancock', 'ssprk', 'kdk'), the per-mode 'auto' default, and the exact-delta contract they work under.
* `common.py`   geometry build, the start-of-step primitives / diagnostics / supports, the geometry hand-over from the timestep.

Unlike the other compressible steps a meshless step owns its whole time step and hands the integrator the exact per-step deltas, so it must run under a
single-evaluation integrator (`schemes/exactDelta.py`). The geometry is built by the backend from the frontend's own positions / supports, so no
frontend Verlet list is needed (the returned adjacency is None).

`meshlessStep` dispatches on `schemeConfig.mode` for callers that hold a config and not a scheme (the core-parity scripts, the benchmark).
"""

from ...exactDelta import requireExactDeltaIntegrator
from .common import BALL, StepContext, stashGeometry, supports_from_volume, wrap
from .mfm import mfmStep
from .mfv import gizmoPrimitiveUpdate, mfvStep
from .step import runMeshlessStep
from .temporal import AUTO_TEMPORAL_SCHEME, TEMPORAL_SCHEMES, resolve_temporal_scheme

__all__ = ['meshlessStep', 'mfmStep', 'mfvStep', 'supports_from_volume', 'resolve_temporal_scheme', 'validateIntegrationScheme', 'stashGeometry']


def meshlessStep(system, dt, config, schemeConfig, verbose = False):
    """The step of the scheme `schemeConfig.mode` names ('MFM' / 'MFV')."""
    mode = str(schemeConfig.mode).upper()
    if mode == 'MFM':
        return mfmStep(system, dt, config, schemeConfig, verbose)
    if mode == 'MFV':
        return mfvStep(system, dt, config, schemeConfig, verbose)
    raise ValueError(f"meshless mode {schemeConfig.mode!r} is not 'MFM' or 'MFV'")


def validateIntegrationScheme(config) -> None:
    """Refuse any integrator that would run the step more than once per time step (`schemes/exactDelta.py`)."""
    requireExactDeltaIntegrator(config, 'MFM / MFV meshless')
