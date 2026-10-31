"""Meshless lifecycle hooks: shared step data, fresh stage geometry and compatible flux updates.

The integrators package owns kicks, drifts and stage combination. As in CRK's
pair-work lifecycle, spatial state is shared explicitly and finalized once.
"""
from dataclasses import dataclass
from .context import StepContext
from .geometry import startOfStepGeometry
from .primitives import primitives, refreshPrimitives, particleGamma
from warpSPHCore.profiling import record_function


@dataclass
class MeshlessStages:
    startState: object = None


def prepareStage(system, dt, config, schemeConfig, stageIndex):
    from .chunked import checkMeshlessChunks
    checkMeshlessChunks(system, config, schemeConfig)
    state = system.state
    if system.drivenParticles is not None:
        system.drivenParticles.position(state, system.t)
    with record_function('[warpSPH] - [meshless stage - 01] geometry'):
        geom = (system.inflowReservoir.build(state, config, schemeConfig, system.t)
                if system.inflowReservoir is not None else
                startOfStepGeometry(state, system.domain, config, schemeConfig))
    if system.drivenParticles is not None:
        system.drivenParticles.conserved(state, geom, system.t, float(schemeConfig.gamma))
    gamma = particleGamma(state, schemeConfig)
    with record_function('[warpSPH] - [meshless stage - 02] primitives'):
        if stageIndex == 0:
            rho, vel, pressure, energy = refreshPrimitives(state, geom, gamma, schemeConfig.nngb, config.dim)
            system.meshlessStages.startState = state
        else:
            # Supports and public diagnostics retain their start-of-step policy.
            rho, vel, pressure, energy = primitives(state.Q, geom.volume, gamma, returnEnergy=True)
    from warpSPHCore.radiusSearch import ChunkedAdjacency
    if isinstance(geom.adjacency, ChunkedAdjacency):
        system.adjacency = geom.adjacency
        checkMeshlessChunks(system, config, schemeConfig)
    state.lam = geom.lam
    system.meshlessContext = StepContext(dt=dt, gamma=gamma, config=config, schemeConfig=schemeConfig,
        domain=system.domain, pos=state.positions, supports=state.supports, Q=state.Q,
        rho=rho, vel=vel, P=pressure, u=energy, geom=geom,
        materials=state.materials if getattr(schemeConfig, 'materials', None) is not None else None,
        reservoir=system.inflowReservoir)
