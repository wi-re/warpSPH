"""Per-boundary-condition dispatch: for each `bc` in
`compParams.boundaryConditions`, evaluate `bc.sdf` on periodic-wrapped
positions and apply the matching Dirichlet/forcing/update function only to
particles inside the SDF (`d < 0`). Works against either a `system.state`
(state-holding system) or a bare state object depending on
`hasattr(system, 'state')`.
"""

from warpSPHCore import *
from ...math import getPeriodicPositions
# from ...systems.compSPH import CompSPHSystem, CompSPHState
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...systems.compressibleMonaghan import CompressibleSystemUpdate
import torch

from ...systems.baseState import BaseState
from ...configurations import SimulationConfig, CompressibleSPHConfig
from typing import Any

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
__all__ = ['enforceDirichlet', 'computeForcing', 'enforceUpdates']

def enforceDirichlet(
    system: Any,
    t: float,
    dt: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    with record_function("[warpSPH] - enforceDirichlet"):
        for bc in compParams.boundaryConditions:
            if bc.dirichletFunctions is not None and len(bc.dirichletFunctions) > 0:
                for varName, dirichletFn in bc.dirichletFunctions.items():
                    #print(f'Enforcing dirichlet for {varName}')
                    if hasattr(system, 'state'):
                        var = getattr(system.state, varName)
                        periodicPositions = getPeriodicPositions(system.state.positions, config.domain)
                        d, n = bc.sdf(periodicPositions)
                        updatedValues = dirichletFn(system.state, config, compParams, periodicPositions, d, n, t, dt)
                        var[d < 0] = updatedValues[d < 0]
                        # print(f'Enforced dirichlet for {varName}')
                    else:
                        var = getattr(system, varName)
                        periodicPositions = getPeriodicPositions(system.positions, config.domain)
                        d, n = bc.sdf(periodicPositions)
                        updatedValues = dirichletFn(system, config, compParams, periodicPositions, d, n, t, dt)
                        var[d < 0] = updatedValues[d < 0]
                        # print(f'Enforced dirichlet for {varName}')
                    
def computeForcing(
    system: Any,
    dt: float,
    t: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    with record_function("[warpSPH] - computeForcing"):
        totalForcing = torch.zeros_like(system.state.positions) if hasattr(system, 'state') else torch.zeros_like(system.positions)
        for bc in compParams.boundaryConditions:
            if bc.forcingFunctions is not None and len(bc.forcingFunctions) > 0:
                for forcingFn in bc.forcingFunctions:
                    if hasattr(system, 'state'):
                        periodicPositions = getPeriodicPositions(system.state.positions, config.domain)
                        d, n = bc.sdf(periodicPositions)
                        forcingValues = forcingFn(system.state, config, compParams, periodicPositions, d, n, t, dt)
                        totalForcing[d < 0] += forcingValues[d < 0]
                    else:
                        periodicPositions = getPeriodicPositions(system.positions, config.domain)
                        d, n = bc.sdf(periodicPositions)
                        forcingValues = forcingFn(system, config, compParams, periodicPositions, d, n, t, dt)
                        totalForcing[d < 0] += forcingValues[d < 0]
        return totalForcing

def enforceUpdates(
    updates: CompressibleSystemUpdate,
    system: Any,
    dt: float,
    t: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    with record_function("[warpSPH] - enforceUpdates"):
        for bc in compParams.boundaryConditions:
            if bc.updateFunctions is not None and len(bc.updateFunctions) > 0:
                for varName, updateFn in bc.updateFunctions.items():
                    # print(f'Enforcing updates for {varName}')
                    if hasattr(system, 'state'):
                        periodicPositions = getPeriodicPositions(system.state.positions, config.domain)
                        var = getattr(updates, varName)
                        d, n = bc.sdf(periodicPositions)
                        updatedValues = updateFn(system.state, config, compParams, periodicPositions, d, n, t, dt)
                        var[d < 0] = updatedValues[d < 0]
                    else:
                        periodicPositions = getPeriodicPositions(system.positions, config.domain)
                        var = getattr(updates, varName)
                        d, n = bc.sdf(periodicPositions)
                        updatedValues = updateFn(system, config, compParams, periodicPositions, d, n, t, dt)
                        var[d < 0] = updatedValues[d < 0]
                    # print(f'Enforced updates for {varName}')