"""MFM/MFV spatial RHS entry points and direct-step compatibility APIs.

Normal runs use the ConservedLagrangian driver in warpSPHIntegrators. The
MeshlessSystem lifecycle prepares fresh geometry/primitives, applies compatible
corrections and finalizes diagnostics. Shared spatial work lives in modules:
meshless face accumulation, reconstruction/limiting, ideal-gas primitives and
moving-face Riemann policy. Geometry remains in warpSPHCore.mfm.

The temporal/common exports preserve oracle utilities. The explicit historical
map is isolated in modules.meshless.compatibility; registered RHSs refuse calls
without the package driver's stage lifecycle. Returned frontend adjacency remains None because
meshless geometry owns its own neighbor structure.
"""

from ...exactDelta import requireExactDeltaIntegrator
from .common import BALL, StepContext, stashGeometry, supports_from_volume, wrap
from .mfm import mfmStep
from .mfv import mfvStep
from .step import runMeshlessStep
from .temporal import AUTO_TEMPORAL_SCHEME, TEMPORAL_SCHEMES, resolve_temporal_scheme

__all__ = ['meshlessStep', 'mfmStep', 'mfvStep', 'supports_from_volume', 'resolve_temporal_scheme', 'validateIntegrationScheme', 'stashGeometry']


def meshlessStep(system, dt, config, schemeConfig, verbose = False, **kwargs):
    """The step of the scheme `schemeConfig.mode` names ('MFM' / 'MFV')."""
    mode = str(schemeConfig.mode).upper()
    if mode == 'MFM':
        return mfmStep(system, dt, config, schemeConfig, verbose, **kwargs)
    if mode == 'MFV':
        return mfvStep(system, dt, config, schemeConfig, verbose, **kwargs)
    raise ValueError(f"meshless mode {schemeConfig.mode!r} is not 'MFM' or 'MFV'")


def validateIntegrationScheme(config) -> None:
    """Validate the legacy direct-step adapter; normal staging is package-owned."""
    requireExactDeltaIntegrator(config, 'MFM / MFV meshless')
