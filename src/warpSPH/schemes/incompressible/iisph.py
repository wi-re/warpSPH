"""IISPH: summation density, predictor, one pressure solve, kick/drift.

Spatial operators and wall closure live in modules.incompressible. Core owns
neighbors, kernels and stationary iteration; Integrators owns time updates.
The pressure field is physical pressure, including its warm-start carry.
"""

import torch
from warpSPHIntegrators.drivers.mechanics.projection import kick, drift
from ...modules.incompressible.iisph import IISPHOperator, iisphViscosity
from ...modules.incompressible.iisphBoundary import buildIISPHBoundary
from ...modules.incompressible.operators import rebuildAdjacency
from ...modules.incompressible.stages import nonPressureAcceleration, projectedUpdate, requireProjectedStep
from ...modules.boundaryConditions import enforceDirichlet
from ...modules.gravity import computeGravity
from ...enumTypes import AdaptiveSupportScheme


def iisph_step(system, dt, config, schemeConfig, verbose=False):
    requireProjectedStep(system, config, 'iisph')
    if schemeConfig.adaptiveSupportScheme != AdaptiveSupportScheme.NoScheme:
        raise NotImplementedError('IISPH requires fixed common support')
    state = system.state
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    adjacency = rebuildAdjacency(state, system, config)
    gravity = computeGravity(state, config, schemeConfig, adjacency)
    boundary = buildIISPHBoundary(state, config, schemeConfig, gravity)
    operator = IISPHOperator(state, config, adjacency, schemeConfig.fluid.restDensity, boundary=boundary)
    state.densities = operator.densityRatio * schemeConfig.fluid.restDensity

    nonpressure = nonPressureAcceleration(system, dt, config, schemeConfig, adjacency,
        includeBodyForce=True, viscosity=False,
        extraAcceleration=lambda: iisphViscosity(state, config, schemeConfig, adjacency))
    fluid = operator.fluid[:, None]
    predicted = kick(state.velocities, dt, nonpressure, mask=fluid)
    previous = torch.zeros_like(state.masses) if state.pressures is None else state.pressures
    result = operator.solve(predicted, previous, dt, schemeConfig.iisph)
    if result.termination == 'nonfinite':
        raise FloatingPointError('Nonfinite IISPH pressure iteration')
    state.velocities = kick(predicted, dt, result.acceleration, mask=fluid)
    state.positions = drift(state.positions, dt, state.velocities, mask=fluid)
    state.pressures = result.pressure
    if verbose:
        print(f'[iisph] t={system.t+dt:.6g} iterations={result.iterations} '
              f'density error={result.error:.6g} termination={result.termination}')
    update = projectedUpdate(system, dt, config, schemeConfig)
    # The runner exposes the last stage update to diagnostics/postStep hooks.
    update.iisphResult = result
    return update, adjacency, state, ([], [result.error, 0.0])
