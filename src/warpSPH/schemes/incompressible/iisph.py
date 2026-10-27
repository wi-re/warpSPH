"""Native IISPH right-hand side: density, forces and one pressure solve."""

from warpSPHCore import SupportScheme, buildVerletList
from warpSPHCore.profiling import record_function
from ...configurations import SimulationConfig, IncompressibleSPHConfig
from ...configurations.compute import neighborSearchOptions
from ...systems.incompressible import IISPHSystem
from ...modules.boundaryConditions import enforceDirichlet, computeForcing, enforceUpdates
from ...modules.density import computeDensities
from ...modules.gravity import computeGravity, computeBodyForce
from ...modules.deltaSPH import computeVelocityDiffusion
from ...modules.iisph import (computeIISPHContactPoints,
                             solveIISPH, assembleIISPHUpdate, maskIISPHUpdate)

__all__ = ['iisph_step']


def iisph_step(
    system: IISPHSystem,
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
            wallMassDensity=schemeConfig.fluid.restDensity, wallSupport=system.boundarySupport)

    with record_function('[warpSPH] - [IISPH - 03] Dirichlet boundary conditions'):
        enforceDirichlet(system, system.t, dt, config, schemeConfig)

    with record_function('[warpSPH] - [IISPH - 04] gravity and body force'):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        acceleration = gravity + computeBodyForce(currentState, schemeConfig)

    with record_function('[warpSPH] - [IISPH - 05] boundary contact points'):
        contactPoints, contactCoverageFailure = computeIISPHContactPoints(currentState, system.contactBoundaryProvider)

    with record_function('[warpSPH] - [IISPH - 06] viscosity'):
        acceleration = acceleration + computeVelocityDiffusion(currentState, config, schemeConfig, adjacency,
            boundaryRepresentation=schemeConfig.iisph.boundaryRepresentation, wallSupport=system.boundarySupport,
            boundaryVelocitiesPrepared=False)

    with record_function('[warpSPH] - [IISPH - 07] forcing'):
        forcing = computeForcing(system, dt, system.t, config, schemeConfig)
        acceleration = acceleration + forcing / currentState.masses.view(-1, 1)

    with record_function('[warpSPH] - [IISPH - 08] pressure projection'):
        result = solveIISPH(currentState, config, schemeConfig, acceleration, dt,
                            adjacency, previousPressure=currentState.pressures,
                            analyticProvider=system.analyticBoundaryProvider, boundarySupport=system.boundarySupport,
                            contactPoints=contactPoints, contactCoverageFailure=contactCoverageFailure,
                            contactGravity=schemeConfig.iisph.contactGravity, gravity=gravity)

    with record_function('[warpSPH] - [IISPH - 09] update'):
        currentState.pressures = result.pressure
        update = assembleIISPHUpdate(currentState, acceleration, result.acceleration)
        update.iisphResult = result

    with record_function('[warpSPH] - [IISPH - 10] update boundary conditions'):
        enforceUpdates(update, system, dt, system.t, config, schemeConfig)
        maskIISPHUpdate(update, currentState.kinds)
        if verbose:
            print(f'[iisph] t={system.t+dt:.6g} iterations={result.iterations} '
                  f'density error={result.error:.6g} termination={result.termination}')
        return update, adjacency, currentState, ([], [result.error, 0.0])
