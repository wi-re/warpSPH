"""Native IISPH: boundary terms, prediction, one pressure solve and drift."""

from warpSPHCore import SupportScheme, buildVerletList
from warpSPHCore.profiling import record_function
from warpSPHIntegrators.drivers.mechanics.projection import kick, drift
from ...configurations import SimulationConfig, IncompressibleSPHConfig
from ...configurations.compute import neighborSearchOptions
from ...systems.incompressible import ProjectedSystem
from ...modules.boundaryConditions import enforceDirichlet, computeForcing
from ...modules.density import computeDensities
from ...modules.gravity import computeGravity, computeBodyForce
from ...modules.incompressible.stages import projectedUpdate
from ...modules.iisph import (computeIISPHBoundaryIntegrals, computeIISPHContactPoints,
                             computeIISPHBoundaryPressureAcceleration, computeIISPHCoefficients,
                             computeIISPHViscosity, solveIISPH)

__all__ = ['iisph_step']


def iisph_step(
    system: ProjectedSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: IncompressibleSPHConfig,
    verbose=False,
):
    currentState = system.state

    with record_function('[warpSPH] - [IISPH - 01] adjacency'):
        adjacency = buildVerletList(currentState, config.domain,
            verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=system.adjacency, verbose=False, **neighborSearchOptions(config))
        system.adjacency = adjacency

    with record_function('[warpSPH] - [IISPH - 02] density'):
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency,
            boundaryRepresentation=schemeConfig.iisph.boundaryRepresentation,
            wallMassDensity=schemeConfig.fluid.restDensity)

    with record_function('[warpSPH] - [IISPH - 03] Dirichlet boundary conditions'):
        enforceDirichlet(system, system.t, dt, config, schemeConfig)

    with record_function('[warpSPH] - [IISPH - 04] gravity and body force'):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        acceleration = gravity + computeBodyForce(currentState, schemeConfig)

    with record_function('[warpSPH] - [IISPH - 05] pressure boundary geometry'):
        boundaryIntegrals = computeIISPHBoundaryIntegrals(currentState, config, schemeConfig)
        contactPoints = computeIISPHContactPoints(currentState, schemeConfig)

    with record_function('[warpSPH] - [IISPH - 06] pressure coefficients'):
        coefficients = computeIISPHCoefficients(currentState, config, schemeConfig, adjacency,
                                                boundaryIntegrals=boundaryIntegrals, contactPoints=contactPoints)

    with record_function('[warpSPH] - [IISPH - 07] wall pressure extrapolation'):
        boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(currentState, config,
            schemeConfig, coefficients, boundaryIntegrals, contactPoints, gravity)

    with record_function('[warpSPH] - [IISPH - 08] viscosity'):
        acceleration = acceleration + computeIISPHViscosity(currentState, config, schemeConfig, adjacency)

    with record_function('[warpSPH] - [IISPH - 09] forcing and velocity prediction'):
        forcing = computeForcing(system, dt, system.t, config, schemeConfig)
        acceleration = acceleration + forcing / currentState.masses.view(-1, 1)
        fluid = (currentState.kinds == 0)[:, None]
        predicted = kick(currentState.velocities, dt, acceleration, mask=fluid)

    with record_function('[warpSPH] - [IISPH - 10] pressure projection'):
        result = solveIISPH(currentState, config, schemeConfig, predicted, dt,
                            coefficients, previousPressure=currentState.pressures,
                            boundaryAcceleration=boundaryAcceleration)

    with record_function('[warpSPH] - [IISPH - 11] pressure kick and position drift'):
        currentState.velocities = kick(predicted, dt, result.acceleration, mask=fluid)
        currentState.positions = drift(currentState.positions, dt, currentState.velocities, mask=fluid)
        currentState.pressures = result.pressure

    with record_function('[warpSPH] - [IISPH - 12] update and boundary conditions'):
        update = projectedUpdate(system, dt, config, schemeConfig)
        update.iisphResult = result
        if verbose:
            print(f'[iisph] t={system.t+dt:.6g} iterations={result.iterations} '
                  f'density error={result.error:.6g} termination={result.termination}')
        return update, adjacency, currentState, ([], [result.error, 0.0])
