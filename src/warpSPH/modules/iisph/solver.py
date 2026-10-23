"""Stateless native IISPH coefficients and relaxed Jacobi pressure solve."""

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


def computeIISPHSource(state, config, adjacency, velocity, dt, restVolumes,
                      densityRatio, wallGradient, boundaryAcceleration):
    """Return the rest-density defect after predicted motion and known wall acceleration."""
    source = (1 - densityRatio
        - dt * computeIISPHDensityRate(state, config, adjacency, velocity, restVolumes, wallGradient)
        - dt * dt * computeIISPHDensityRate(state, config, adjacency, boundaryAcceleration, restVolumes, wallGradient))
    return torch.where(state.kinds == 0, source, 0)


def computeIISPHCoefficients(state, config, schemeConfig, adjacency, *, boundaryIntegrals=None, contactPoints=None):
    """Return rest volumes, density ratios, wall/total gradients, pressure
    diagonal, contact adjacency, contact normalization and a device-side
    missing-support flag as separate outputs.

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
    missingContactSupport = torch.zeros((), dtype=torch.bool, device=state.positions.device)
    if contactPoints is not None:
        contactAdjacency, contactDenominator, missingContactSupport = computeIISPHContactWeights(
            state, config, contactPoints, wallGradient, restVolumes)
        contactDiagonal = computeIISPHContactPressureDiagonal(state, config, adjacency,
            restVolumes, wallGradient, gradientSum, contactPoints, contactDenominator)
        diagonal = diagonal + contactDiagonal
    # endregion

    return (restVolumes, densityRatio, wallGradient, gradientSum, diagonal,
            contactAdjacency, contactDenominator, missingContactSupport)


def solveIISPH(state, config, schemeConfig, acceleration, dt, adjacency, previousPressure=None, *,
               analyticProvider=None, boundarySupport=None,
               contactPoints=None, contactCoverageFailure=None, contactGravity=True, gravity=None):
    """Assemble and solve IISPH at fixed geometry with physical-pressure carry.

    Nonnegative pressure uses the reference signed statistic with the new
    diagonal and old off-diagonal iterate. Signed pressure uses the mean
    absolute residual of the input iterate.
    Tolerance is a percentage of rest density.
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

    # region Compute contributing pressure terms
    boundaryIntegrals = None
    if analyticProvider is not None:
        boundaryIntegrals = computeIISPHBoundaryIntegrals(state, analyticProvider, boundarySupport, config.kernel)
    (restVolumes, densityRatio, wallGradient, gradientSum, pressureDiagonal,
     contactAdjacency, contactDenominator, missingContactSupport) = computeIISPHCoefficients(
        state, config, schemeConfig, adjacency, boundaryIntegrals=boundaryIntegrals, contactPoints=contactPoints)
    # One geometry diagnostic checkpoint, before entering the pressure iteration.
    if contactPoints is not None:
        coverage = torch.zeros_like(missingContactSupport) if contactCoverageFailure is None else contactCoverageFailure
        uncovered, unsupported = torch.stack((coverage, missingContactSupport)).cpu().tolist()
        if uncovered:
            raise ValueError('IISPH contact geometry did not cover every fluid query')
        if unsupported:
            raise ValueError('IISPH contact pressure has no fluid support at a solid contact')
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(state, config,
        adjacency, restVolumes, wallGradient, boundaryIntegrals, contactPoints, gravity,
        contactGravity=contactGravity, massFactor=float(getattr(schemeConfig, 'analyticWallMass', 1.0)))
    # endregion

    # region Set up solver state
    velocity = state.velocities + dt * torch.where(
        fluid[:, None], acceleration, torch.zeros_like(acceleration))
    dt2 = dt * dt
    b = computeIISPHSource(state, config, adjacency, velocity, dt, restVolumes,
                          densityRatio, wallGradient, boundaryAcceleration)
    if previousPressure is None:
        previousPressure = torch.zeros_like(state.masses)
    q = torch.where(fluid, 0.5 * previousPressure / rho0, 0)
    diagonal = dt2 * pressureDiagonal
    valid = fluid & (diagonal.abs() > 1.e-9)
    denominator = torch.where(valid, diagonal, 1)
    omega = policy.relaxationFactor
    fluidCount = fluid.sum()
    clampPressure = policy.clampPressure
    # endregion

    # region Run the pressure solver
    def iteration(old, _):
        off = dt2 * computeIISPHPressureOperator(state, config, adjacency, old,
            restVolumes, densityRatio, gradientSum, wallGradient,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator) - diagonal * old
        new = (1-omega)*old + omega*(b-off)/denominator
        if clampPressure:
            new = torch.where(valid, new.clamp_min(0), 0)
            errors = torch.where(new != 0, rho0*(diagonal*new + off-b), 0)
        else:
            new = torch.where(valid, new, 0)
            errors = torch.where(fluid, rho0*(diagonal*old + off-b).abs(), 0)
        return new, errors.sum() / fluidCount, None

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
