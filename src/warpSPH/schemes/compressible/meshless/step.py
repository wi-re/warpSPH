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
    source_energy = None
    if system.sourceTerms is not None:
        if temporalScheme != 'ssprk':
            raise ValueError('meshless source terms currently require solver.temporalScheme=ssprk')
        with record_function('[warpSPH] - [meshless - external sources]'):
            from ....distributed import processGroup, validateSourceClock
            import torch
            communication = processGroup(system)
            if communication is not None:
                validateSourceClock(system, communication)
            evaluate = lambda: system.sourceTerms.fields(ctx.Q[:, 0], ctx.pos, ctx.vel, ctx.u, system.t)
            momentum, work, heat = (evaluate() if communication is None else
                communication.localPhase('meshless:source:fields', evaluate))
            source = torch.cat((torch.zeros_like(ctx.Q[:, :1]), momentum, (work+heat)[:, None]), 1)
            if communication is None:
                source_energy = torch.stack((work.double().sum(), heat.double().sum()))
            else:
                communication.assertReplicated('meshless:source:fields', dict(source=source))
                source_energy = torch.stack((communication.sumOwned(work, system.adjacency, tag='source:work'),
                    communication.sumOwned(heat, system.adjacency, tag='source:heat')))
            rate = rate+source
    with record_function(f"[warpSPH] - [{mode.lower()} - 04] stage derivative"):
        update = MeshlessSystemUpdate(dxdt=ctx.vel, dQdt=rate, sourceEnergyRates=source_energy)
    from warpSPHCore.radiusSearch import ChunkedAdjacency
    return update, (ctx.geom.adjacency if isinstance(ctx.geom.adjacency, ChunkedAdjacency) else None), system.state
