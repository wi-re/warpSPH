"""Compatibility map for old direct callers; temporal arithmetic lives in warpSPHIntegrators."""
from warpSPHCore.profiling import record_function
from ...schemes.exactDelta import requireExactDeltaIntegrator
from ...systems.meshless import MeshlessSystemUpdate
from .context import StepContext, refreshPrimitives, startOfStepGeometry
from .temporal import advance, resolve_temporal_scheme


def legacyMeshlessStep(system, dt, config, schemeConfig, verbose=False):
    """Explicit historical map for oracle scripts only; never a registered RHS."""
    from .temporal import conservedHancockUpdate
    from .updates import gizmoPrimitiveUpdate
    update = gizmoPrimitiveUpdate if schemeConfig.mode == 'MFV' and schemeConfig.mfvUpdate == 'primitive' else conservedHancockUpdate
    return exactDeltaStep(system, dt, config, schemeConfig, mode=schemeConfig.mode, hancockUpdate=update)


def exactDeltaStep(system, dt, config, schemeConfig, *, mode, hancockUpdate):
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
