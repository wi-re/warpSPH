"""Stateless native IISPH coefficients and positive Jacobi pressure solve."""

from dataclasses import dataclass
from types import SimpleNamespace
import torch
from warpSPHCore.linalg import CheckSchedule, runStationary
from .wp_densityRate import computeIISPHDensityRate
from .wp_pressureAcceleration import computeIISPHPressureAcceleration, computeIISPHPressureOperator
from .contactWeights import computeIISPHContactWeights
from .boundaryPressureAcceleration import computeIISPHBoundaryPressureAcceleration
from .boundaryIntegrals import computeIISPHBoundaryIntegrals
from .wp_wallGradient import computeIISPHWallGradient
from .wp_gradientSum import computeIISPHGradientSum
from .wp_pressureDiagonal import computeIISPHPressureDiagonal
from .wp_contactPressureDiagonal import computeIISPHContactPressureDiagonal


@dataclass
class IISPHResult:
    pressure: torch.Tensor
    acceleration: torch.Tensor
    iterations: int
    error: float
    termination: str
    source: torch.Tensor


def computeIISPHCoefficients(state, config, schemeConfig, adjacency, *, boundaryIntegrals=None, contactPoints=None):
    """Return rest volumes, density ratios, wall/total gradients, pressure
    diagonal, contact adjacency and contact normalization as separate outputs.

    Static kind-1 samples use rest volumes, as do the kind-0 fluid unknowns.
    Density is read from state after Dirichlet conditions are enforced.
    The diagonal includes the nonlocal contact-pressure interpolation.
    Raw analytic geometry supplies the wall gradient; no gravity term is
    evaluated or retained by these coefficients.
    """
    # region Normalized particle fields
    rho0 = schemeConfig.fluid.restDensity
    restVolumes = (state.masses / rho0).contiguous()
    densityRatio = (state.densities / rho0).contiguous()
    # endregion

    # region Pressure-force gradient terms
    wallGradient = computeIISPHWallGradient(state, config, adjacency, restVolumes,
        boundaryIntegrals=boundaryIntegrals,
        massFactor=float(getattr(schemeConfig, 'analyticWallMass', 1.0)))
    gradientSum = computeIISPHGradientSum(state, config, adjacency, restVolumes, wallGradient)
    # endregion

    # region Base pressure diagonal
    diagonal = computeIISPHPressureDiagonal(state, config, adjacency, restVolumes, densityRatio, gradientSum)
    # endregion

    # region Contact-pressure interpolation and diagonal contribution
    contactAdjacency = None
    contactDenominator = None
    if contactPoints is not None:
        contactAdjacency, contactDenominator = computeIISPHContactWeights(
            state, config, contactPoints, wallGradient, restVolumes)
        contactDiagonal = computeIISPHContactPressureDiagonal(state, config, adjacency,
            restVolumes, wallGradient, gradientSum, contactPoints, contactDenominator)
        diagonal = diagonal + contactDiagonal
    # endregion

    return restVolumes, densityRatio, wallGradient, gradientSum, diagonal, contactAdjacency, contactDenominator


def solveIISPH(state, config, schemeConfig, acceleration, dt, adjacency, previousPressure=None, *,
               contactPoints=None, gravity=None):
    """Assemble and solve IISPH at fixed geometry with physical-pressure carry.

    The signed statistic uses the new diagonal and old off-diagonal iterate,
    averaged over all fluid rows. Tolerance is a percentage of rest density.
    The nonpressure acceleration supplies a temporary predicted velocity;
    simulation positions and velocities remain untouched.
    Boundary integrals, pressure coefficients, contact interpolation and gravity
    extrapolation are local to this solve. The known wall acceleration contributes
    to both the source and the final pressure acceleration.
    No input field is modified; the caller applies the returned acceleration.
    """
    policy = schemeConfig.iisph
    rho0 = schemeConfig.fluid.restDensity
    fluid = state.kinds == 0
    policy.validate()

    # region Compute contributing pressure terms
    boundaryIntegrals = computeIISPHBoundaryIntegrals(state, config, schemeConfig)
    (restVolumes, densityRatio, wallGradient, gradientSum, pressureDiagonal,
     contactAdjacency, contactDenominator) = computeIISPHCoefficients(
        state, config, schemeConfig, adjacency, boundaryIntegrals=boundaryIntegrals, contactPoints=contactPoints)
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(state, config,
        schemeConfig, adjacency, restVolumes, wallGradient, boundaryIntegrals, contactPoints, gravity)
    # endregion

    # region Set up solver state
    velocity = state.velocities + dt * torch.where(
        fluid[:, None], acceleration, torch.zeros_like(acceleration))
    dt2 = dt * dt
    b = (1 - densityRatio - dt * computeIISPHDensityRate(state, config, adjacency, velocity, restVolumes, wallGradient)
         - dt2 * computeIISPHDensityRate(state, config, adjacency, boundaryAcceleration, restVolumes, wallGradient))
    b = torch.where(fluid, b, 0)
    if previousPressure is None:
        previousPressure = torch.zeros_like(state.masses)
    q = torch.where(fluid, 0.5 * previousPressure / rho0, 0)
    diagonal = dt2 * pressureDiagonal
    valid = fluid & (diagonal.abs() > 1.e-9)
    denominator = torch.where(valid, diagonal, 1)
    omega = policy.relaxationFactor
    # endregion

    # region Run the pressure solver
    def iteration(old, _):
        off = dt2 * computeIISPHPressureOperator(state, config, adjacency, old,
            restVolumes, densityRatio, gradientSum, wallGradient,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator) - diagonal * old
        new = torch.where(valid, ((1-omega)*old + omega*(b-off)/denominator).clamp_min(0), 0)
        errors = torch.where(new != 0, rho0*(diagonal*new + off-b), 0)
        return new, errors[fluid].mean(), None

    schedule = CheckSchedule(SimpleNamespace(), 'iisph', 'every',
                             policy.minIterations-1, policy.maxIterations)
    result = runStationary(iteration, q, maxIters=policy.maxIterations,
        tolerance=0.01*policy.maxErrorPercent*rho0, inclusive=True, schedule=schedule)
    if result.termination == 'nonfinite':
        raise FloatingPointError('Nonfinite IISPH pressure iteration')
    # endregion

    # region Compute final pressure acceleration and return
    return IISPHResult(result.x*rho0,
        computeIISPHPressureAcceleration(state, config, adjacency, result.x,
            restVolumes, densityRatio, gradientSum, wallGradient,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator)+boundaryAcceleration,
        result.iterations, result.stats[-1], result.termination, b)
    # endregion
