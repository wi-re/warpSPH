"""Native DFSPH: divergence, nonpressure forces, density, then one kick/drift.

Both projections use native IISPH's C/F/exact diagonal at the same geometry.
The package ProjectedStep driver owns one evaluation of this staged map;
temporal kick/drift primitives come from warpSPHIntegrators. The explicit
band2018 option substitutes volume-centric fluid/wall pressure unknowns for
both constraints. No VD+PS shift or sound-speed pressure carry in this order.
"""

import torch
from warpSPHCore import SupportScheme, buildVerletList
from warpSPHCore.profiling import record_function
from warpSPHIntegrators.drivers.mechanics.projection import kick, symplecticProjection
from ...configurations.compute import neighborSearchOptions
from ...modules.boundaryConditions import enforceDirichlet, computeForcing, enforceUpdates
from ...modules.density import computeDensities
from ...modules.gravity import computeGravity, computeBodyForce
from ...modules.deltaSPH import computeVelocityDiffusion
from ...modules.iisph import (computeIISPHContactPoints, computeIISPHProjectionGeometry,
    computeIISPHBoundaryPressureAcceleration, assembleIISPHUpdate, maskIISPHUpdate)
from ...modules.dfsph import solveDFSPHProjection
from ...modules.incompressible.stages import requireProjectedStep
from ...modules.util import countNeighbors

__all__ = ['dfsph_step']


def dfsph_step(system, dt, config, schemeConfig, verbose=False):
    from warpSPH.modules.incompressible.chunked import checkProjectionChunks
    requireProjectedStep(system, config, 'dfsph')
    checkProjectionChunks(system, config, schemeConfig)
    if schemeConfig.dfsph.ordering != 'df_np_cd':
        from ...modules.dfsph.composition import advanceDFSPH
        return advanceDFSPH(system, dt, config, schemeConfig, schemeConfig.dfsph.ordering)
    state = system.state

    with record_function('[warpSPH] - [DFSPH - 01] adjacency'):
        adjacency = buildVerletList(state, config.domain,
            verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=system.adjacency, verbose=False, **neighborSearchOptions(config))
        system.adjacency = adjacency

    with record_function('[warpSPH] - [DFSPH - 02] density'):
        state.densities = computeDensities(state, config, schemeConfig, adjacency,
            boundaryRepresentation=schemeConfig.iisph.boundaryRepresentation,
            wallMassDensity=schemeConfig.fluid.restDensity, wallSupport=system.boundarySupport)

    with record_function('[warpSPH] - [DFSPH - 03] Dirichlet boundary conditions'):
        enforceDirichlet(system, system.t, dt, config, schemeConfig)

    with record_function('[warpSPH] - [DFSPH - 04] pressure geometry'):
        contactPoints, coverage = computeIISPHContactPoints(state, system.contactBoundaryProvider)
        pressureBoundaryGeometry = None
        if schemeConfig.dfsph.pressureBoundary == 'band2018':
            from ...modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
            pressureBoundaryGeometry = buildPressureBoundaryGeometry(state, config, schemeConfig,
                adjacency, analyticProvider=system.analyticBoundaryProvider)
            volumes, ratios = pressureBoundaryGeometry.restVolumes, pressureBoundaryGeometry.ratios
            wallGradient = gradientSum = torch.zeros_like(state.positions)
            diagonal = pressureBoundaryGeometry.diagonal
            contactPoints = contactAdjacency = contactDenominator = integrals = None
        else:
            (volumes, ratios, wallGradient, gradientSum, diagonal,
             contactAdjacency, contactDenominator, integrals) = computeIISPHProjectionGeometry(
                state, config, schemeConfig, adjacency,
                analyticProvider=system.analyticBoundaryProvider, boundarySupport=system.boundarySupport,
                contactPoints=contactPoints, contactCoverageFailure=coverage)

    with record_function('[warpSPH] - [DFSPH - 05] divergence projection'):
        deficient = None
        if schemeConfig.dfsph.deficiencyGuard and schemeConfig.dfsph.divergence.clampPressure:
            threshold = 7 if config.domain.dim == 2 else 20
            deficient = countNeighbors(state, config, schemeConfig, adjacency) < threshold
        divergence = solveDFSPHProjection(state, config, schemeConfig, dt, adjacency,
            volumes, ratios, wallGradient, gradientSum, diagonal,
            mode='divergence', velocity=state.velocities,
            pressureImpulse=state.divergencePressureImpulse,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator, deficient=deficient,
            pressureBoundaryGeometry=pressureBoundaryGeometry)
        state.velocities = kick(state.velocities, dt, divergence.acceleration)

    with record_function('[warpSPH] - [DFSPH - 06] gravity and body force'):
        gravity = computeGravity(state, config, schemeConfig, adjacency)
        acceleration = gravity + computeBodyForce(state, schemeConfig)

    with record_function('[warpSPH] - [DFSPH - 07] viscosity'):
        acceleration = acceleration + computeVelocityDiffusion(state, config, schemeConfig, adjacency,
            boundaryRepresentation=schemeConfig.iisph.boundaryRepresentation, wallSupport=system.boundarySupport,
            boundaryVelocitiesPrepared=False)

    with record_function('[warpSPH] - [DFSPH - 08] forcing'):
        acceleration = acceleration + computeForcing(system, dt, system.t, config, schemeConfig)/state.masses[:, None]

    with record_function('[warpSPH] - [DFSPH - 09] density projection'):
        knownWall = computeIISPHBoundaryPressureAcceleration(state, config, adjacency,
            volumes, wallGradient, integrals, contactPoints, gravity,
            contactGravity=schemeConfig.iisph.contactGravity,
            massFactor=float(getattr(schemeConfig, 'analyticWallMass', 1.0)))
        predictor = state.velocities + dt*torch.where((state.kinds == 0)[:, None], acceleration, 0)
        density = solveDFSPHProjection(state, config, schemeConfig, dt, adjacency,
            volumes, ratios, wallGradient, gradientSum, diagonal,
            mode='density', velocity=predictor, previousPressure=state.pressures,
            pressureImpulse=state.densityPressureImpulse, boundaryAcceleration=knownWall,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator, pressureBoundaryGeometry=pressureBoundaryGeometry)

    with record_function('[warpSPH] - [DFSPH - 10] update boundary conditions'):
        update = assembleIISPHUpdate(state, acceleration, density.acceleration)
        enforceUpdates(update, system, dt, system.t, config, schemeConfig)
        maskIISPHUpdate(update, state.kinds)
        update.divergenceResult, update.densityResult = divergence, density

    with record_function('[warpSPH] - [DFSPH - 11] kick and drift'):
        symplecticProjection(state, dt, update.dvdt, mask=(state.kinds == 0)[:, None])
        state.pressures = density.pressure
        state.divergencePressures = divergence.pressure
        state.densityPressureImpulse = density.pressure*(dt*dt)
        state.divergencePressureImpulse = divergence.pressure*dt
        if verbose:
            print(f'[dfsph] t={system.t+dt:.6g} '
                  f'DF {divergence.iterations} error={divergence.error:.6g} {divergence.termination}; '
                  f'CD {density.iterations} error={density.error:.6g} {density.termination}')
    return update, adjacency, state, ([], [density.error, divergence.error])
