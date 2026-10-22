"""Staged MFM/MFV RHS. System lifecycle hooks prepare and finalize spatial state.

The ConservedLagrangian driver in warpSPHIntegrators owns all temporal stages.
The exact-delta adapter remains available for older direct-step callers.
"""
from warpSPHCore.profiling import record_function
from ....systems.meshless import MeshlessSystemUpdate
from ....modules.meshless.compatibility import exactDeltaStep

__all__ = ['runMeshlessStep']


def runMeshlessStep(system, dt, config, schemeConfig, *, mode, hancockUpdate,
                    conservedIntegration=False, stageIndex=0, temporalScheme=None):
    if str(schemeConfig.mode).upper() != mode:
        raise ValueError(f'the {mode} step was given a scheme config with mode={schemeConfig.mode!r}; use {mode}Config')
    if not conservedIntegration:
        return exactDeltaStep(system, dt, config, schemeConfig, mode=mode, hancockUpdate=hancockUpdate)

    ctx = system.meshlessContext
    primitiveKick = temporalScheme == 'hancock' and mode == 'MFV' and schemeConfig.mfvUpdate == 'primitive'
    with record_function(f"[warpSPH] - [{mode.lower()} - 03] reconstructed face flux"):
        rate = ctx.rates(0.0 if primitiveKick else schemeConfig.massFluxLimit)
    with record_function(f"[warpSPH] - [{mode.lower()} - 04] stage derivative"):
        update = MeshlessSystemUpdate(dxdt=ctx.vel, dQdt=rate)
        system.meshlessUpdate = update
    return update, None, system.state
