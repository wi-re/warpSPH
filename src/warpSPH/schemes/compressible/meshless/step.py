"""The meshless step skeleton shared by MFM and MFV: `runMeshlessStep`.

Each mode (`mfm.py`, `mfv.py`) supplies only its name and its Euler update of the conserved vector for the 'hancock' temporal scheme; everything
else -- the exact-delta guard, the geometry, the primitives, the supports, the temporal scheme, the update handed to the integrator -- is here.
"""

from warpSPHCore.profiling import record_function

from ....configurations import SimulationConfig
from ....configurations.meshless import MeshlessConfig
from ....systems.meshless import MeshlessSystem, MeshlessSystemUpdate
from ...exactDelta import requireExactDeltaIntegrator
from .common import StepContext, refreshPrimitives, startOfStepGeometry
from .temporal import HancockUpdate, advance, resolve_temporal_scheme

__all__ = ['runMeshlessStep']


def runMeshlessStep(system: MeshlessSystem, dt, config: SimulationConfig, schemeConfig: MeshlessConfig, *, mode: str,
                    hancockUpdate: HancockUpdate):
    """One full meshless time step in `mode` ('MFM' / 'MFV'); returns ``(update, None, state)`` with the exact per-step deltas."""
    if str(schemeConfig.mode).upper() != mode:
        raise ValueError(f'the {mode} step was given a scheme config with mode={schemeConfig.mode!r}; use {mode}Config')

    with record_function(f"[warpSPH] - [{mode.lower()} - 01] integrator guard"):
        requireExactDeltaIntegrator(config, f'{mode} meshless')           # the step owns the full time step: one evaluation per step

    currentState = system.state
    domain = system.domain
    gamma = float(schemeConfig.gamma)
    scheme = resolve_temporal_scheme(schemeConfig)

    with record_function(f"[warpSPH] - [{mode.lower()} - 02] geometry"):
        geom = startOfStepGeometry(currentState, domain, config, schemeConfig)

    with record_function(f"[warpSPH] - [{mode.lower()} - 03] primitives and supports"):
        rho, vel, P, u = refreshPrimitives(currentState, geom, gamma, schemeConfig.nngb, config.dim)

    ctx = StepContext(dt=dt, gamma=gamma, config=config, schemeConfig=schemeConfig, domain=domain, pos=currentState.positions,
                      supports=currentState.supports, Q=currentState.Q, rho=rho, vel=vel, P=P, u=u, geom=geom)

    with record_function(f"[warpSPH] - [{mode.lower()} - 04] {scheme} step"):
        result = advance(ctx, scheme, hancockUpdate)

    # hand the integrator the exact per-step deltas: it applies pos += dt*dxdt (wrapping) and Q += dt*dQdt. The closure warm start for the
    # next step is the last geometry's lam.
    currentState.lam = result.lam
    return MeshlessSystemUpdate(dxdt=result.dxdt, dQdt=result.dQdt), None, currentState
