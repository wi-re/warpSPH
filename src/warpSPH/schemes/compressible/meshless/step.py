"""Staged MFM/MFV RHS. System lifecycle hooks prepare and finalize spatial state.

The ConservedLagrangian driver in warpSPHIntegrators owns all temporal stages.
Registered RHS calls never invoke the historical exact-delta adapter.
"""
from warpSPHCore.profiling import record_function
from ....systems.meshless import MeshlessSystemUpdate

__all__ = ['runMeshlessStep']


def runMeshlessStep(system, dt, config, schemeConfig, *, mode,
                    conservedIntegration=False, stageIndex=0, temporalScheme=None):
    if str(schemeConfig.mode).upper() != mode:
        raise ValueError(f'the {mode} step was given a scheme config with mode={schemeConfig.mode!r}; use {mode}Config')
    if not conservedIntegration:
        raise ValueError(
            f'{mode} requires its warpSPHIntegrators ConservedLagrangian driver and stage lifecycle; '
            'a whole-step delta cannot be used as an ordinary RHS. Use schemeConfig.integrationMethod().')
    if schemeConfig.mfvUpdate != 'conserved':
        raise ValueError('mfvUpdate must be conserved: the experimental primitive kick is a whole-step map, not a stage derivative')

    ctx = system.meshlessContext
    with record_function(f"[warpSPH] - [{mode.lower()} - 03] reconstructed face flux"):
        rate = ctx.rates(schemeConfig.massFluxLimit)
    with record_function(f"[warpSPH] - [{mode.lower()} - 04] stage derivative"):
        update = MeshlessSystemUpdate(dxdt=ctx.vel, dQdt=rate)
    return update, None, system.state
