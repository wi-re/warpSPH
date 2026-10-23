"""Single-purpose stages for the native IISPH projected state map.

These stages own the predictor, warm-start carry, temporal recurrences and
result bookkeeping. The scheme only orders their calls. Solver policy stays
in schemeConfig.iisph; no stage mutates configuration or caches run state.
"""

import torch
from warpSPHIntegrators.drivers.mechanics.projection import kick, drift
from ...enumTypes import AdaptiveSupportScheme
from ..boundaryConditions import enforceDirichlet
from ..gravity import computeGravity
from ..incompressible.operators import rebuildAdjacency
from ..incompressible.stages import nonPressureAcceleration, projectedUpdate, requireProjectedStep
from .solver import IISPHOperator
from .boundary import buildIISPHBoundary
from .viscosity import iisphViscosity


def prepareIISPHStep(system, dt, config, schemeConfig):
    """Validate driver/support compatibility and enforce entering BCs."""
    requireProjectedStep(system, config, 'iisph')
    if schemeConfig.adaptiveSupportScheme != AdaptiveSupportScheme.NoScheme:
        raise NotImplementedError('IISPH requires fixed common support')
    enforceDirichlet(system, system.t, dt, config, schemeConfig)


def buildIISPHOperator(state, config, schemeConfig, adjacency):
    """Build fixed-state wall/pressure coefficients and install solve density."""
    gravity = computeGravity(state, config, schemeConfig, adjacency)
    boundary = buildIISPHBoundary(state, config, schemeConfig, gravity)
    operator = IISPHOperator(state, config, adjacency, schemeConfig.fluid.restDensity, boundary=boundary)
    state.densities = operator.densityRatio * schemeConfig.fluid.restDensity
    return operator


def initializeIISPHDensity(system, config, schemeConfig):
    """Use the same physical summation density for initialization and stepping."""
    adjacency = rebuildAdjacency(system.state, system, config)
    buildIISPHOperator(system.state, config, schemeConfig, adjacency)
    return system.state.densities


def predictIISPHVelocity(system, dt, config, schemeConfig, adjacency, operator):
    """Apply the configured nonpressure acceleration to fluid velocities."""
    state = system.state
    nonpressure = nonPressureAcceleration(system, dt, config, schemeConfig, adjacency,
        includeBodyForce=True, viscosity=False,
        extraAcceleration=lambda: iisphViscosity(state, config, schemeConfig, adjacency))
    return kick(state.velocities, dt, nonpressure, mask=operator.fluid[:, None])


def solveIISPHPressure(state, dt, operator, predictedVelocity, policy):
    """Warm-start the pressure projection and reject a nonfinite solve."""
    previous = torch.zeros_like(state.masses) if state.pressures is None else state.pressures
    result = operator.solve(predictedVelocity, previous, dt, policy)
    if result.termination == 'nonfinite':
        raise FloatingPointError('Nonfinite IISPH pressure iteration')
    return result


def advanceIISPHState(state, dt, predictedVelocity, fluidMask, result):
    """Pressure kick, masked drift and physical-pressure carry, in that order."""
    fluid = fluidMask[:, None]
    state.velocities = kick(predictedVelocity, dt, result.acceleration, mask=fluid)
    state.positions = drift(state.positions, dt, state.velocities, mask=fluid)
    state.pressures = result.pressure


def finishIISPHStep(system, dt, config, schemeConfig, adjacency, result, verbose=False):
    """Return the standard projected update and expose solve diagnostics."""
    if verbose:
        print(f'[iisph] t={system.t+dt:.6g} iterations={result.iterations} '
              f'density error={result.error:.6g} termination={result.termination}')
    update = projectedUpdate(system, dt, config, schemeConfig)
    update.iisphResult = result
    return update, adjacency, system.state, ([], [result.error, 0.0])
