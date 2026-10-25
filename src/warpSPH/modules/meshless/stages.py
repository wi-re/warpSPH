"""Meshless lifecycle hooks: shared step data, fresh stage geometry and compatible flux updates.

The integrators package owns kicks, drifts and stage combination. As in CRK's
pair-work lifecycle, spatial state is shared explicitly and finalized once.
"""
from dataclasses import dataclass
from .context import StepContext
from .geometry import startOfStepGeometry
from .primitives import primitives, refreshPrimitives
from warpSPHCore.profiling import record_function


@dataclass
class MeshlessStages:
    startState: object = None


def prepareStage(system, dt, config, schemeConfig, stageIndex):
    state = system.state
    with record_function('[warpSPH] - [meshless stage - 01] geometry'):
        geom = startOfStepGeometry(state, system.domain, config, schemeConfig)
    with record_function('[warpSPH] - [meshless stage - 02] primitives'):
        if stageIndex == 0:
            rho, vel, pressure, energy = refreshPrimitives(state, geom, float(schemeConfig.gamma), schemeConfig.nngb, config.dim)
            system.meshlessStages.startState = state
        else:
            # Supports and public diagnostics retain their start-of-step policy.
            rho, vel, pressure, energy = primitives(state.Q, geom.volume, float(schemeConfig.gamma), returnEnergy=True)
    state.lam = geom.lam
    system.meshlessContext = StepContext(dt=dt, gamma=float(schemeConfig.gamma), config=config, schemeConfig=schemeConfig,
        domain=system.domain, pos=state.positions, supports=state.supports, Q=state.Q,
        rho=rho, vel=vel, P=pressure, u=energy, geom=geom)
