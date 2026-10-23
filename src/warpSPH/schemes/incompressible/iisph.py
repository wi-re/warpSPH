"""Native IISPH stage ordering; all numerical work lives in modules.iisph.

Entering BCs -> adjacency -> density/coefficients -> nonpressure predictor
-> one pressure projection -> kick/drift -> update and diagnostics.
The package ProjectedStep driver consumes this complete state map once.
"""

from warpSPHCore.profiling import record_function
from ...configurations import SimulationConfig, IncompressibleSPHConfig
from ...systems.incompressible import ProjectedSystem
from ...modules.incompressible.operators import rebuildAdjacency
from ...modules.iisph.stages import (
    prepareIISPHStep,
    buildIISPHOperator,
    predictIISPHVelocity,
    solveIISPHPressure,
    advanceIISPHState,
    finishIISPHStep,
)

__all__ = ['iisph_step']


def iisph_step(
    system: ProjectedSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: IncompressibleSPHConfig,
    verbose=False,
):
    with record_function('[warpSPH] - [IISPH - 01] entering boundary conditions'):
        prepareIISPHStep(system, dt, config, schemeConfig)

    currentState = system.state
    with record_function('[warpSPH] - [IISPH - 02] adjacency'):
        adjacency = rebuildAdjacency(currentState, system, config)

    with record_function('[warpSPH] - [IISPH - 03] density and pressure coefficients'):
        operator = buildIISPHOperator(currentState, config, schemeConfig, adjacency)

    with record_function('[warpSPH] - [IISPH - 04] nonpressure predictor'):
        predicted = predictIISPHVelocity(system, dt, config, schemeConfig, adjacency, operator)

    with record_function('[warpSPH] - [IISPH - 05] pressure projection'):
        result = solveIISPHPressure(currentState, dt, operator, predicted, schemeConfig.iisph)

    with record_function('[warpSPH] - [IISPH - 06] pressure kick and position drift'):
        advanceIISPHState(currentState, dt, predicted, operator.fluid, result)

    with record_function('[warpSPH] - [IISPH - 07] update and boundary conditions'):
        return finishIISPHStep(system, dt, config, schemeConfig, adjacency, result, verbose)
