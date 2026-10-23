"""Stateless native IISPH coefficients and positive Jacobi pressure solve."""

from dataclasses import dataclass
from typing import NamedTuple
from types import SimpleNamespace
import torch
from warpSPHCore.linalg import CheckSchedule, runStationary
from .operators import (IISPHKernelData, computeIISPHKernelGeometry,
    computeIISPHDensityRate, computeIISPHPressureOperator,
    computeIISPHPressureAcceleration)
from .contactWeights import IISPHContactWeights, computeIISPHContactWeights
from .boundaryPressureAcceleration import computeIISPHBoundaryPressureAcceleration
from .boundaryIntegrals import computeIISPHBoundaryIntegrals
from .wallGradient import computeIISPHWallGradient
from .gradientSum import computeIISPHGradientSum
from .pressureDiagonal import computeIISPHPressureDiagonal
from .contactPressureDiagonal import computeIISPHContactPressureDiagonal


@dataclass
class IISPHResult:
    pressure: torch.Tensor
    acceleration: torch.Tensor
    iterations: int
    error: float
    termination: str
    source: torch.Tensor


class IISPHCoefficients(NamedTuple):
    pairs: IISPHKernelData
    restVolumes: torch.Tensor
    densityRatio: torch.Tensor
    wallGradient: torch.Tensor
    gradientSum: torch.Tensor
    diagonal: torch.Tensor
    contactWeights: IISPHContactWeights = None


def computeIISPHCoefficients(state, config, schemeConfig, adjacency, *, boundaryIntegrals=None, contactPoints=None):
    """Connect the fixed-position pressure terms into a plain tensor record.

    Static kind-1 samples use rest volumes, as do the kind-0 fluid unknowns.
    Density is read from state after Dirichlet conditions are enforced.
    The diagonal includes the nonlocal contact-pressure interpolation.
    Raw analytic geometry supplies the wall gradient; no gravity term is
    evaluated or retained by these coefficients.
    """
    rho0 = schemeConfig.fluid.restDensity
    pairs = computeIISPHKernelGeometry(state, config, adjacency)
    restVolumes = (state.masses / rho0).contiguous()
    densityRatio = (state.densities / rho0).contiguous()
    wallGradient = computeIISPHWallGradient(state, pairs, restVolumes,
        boundaryIntegrals=boundaryIntegrals,
        massFactor=float(getattr(schemeConfig, 'analyticWallMass', 1.0)))
    gradientSum = computeIISPHGradientSum(state, pairs, restVolumes, wallGradient)
    diagonal = computeIISPHPressureDiagonal(state, pairs, restVolumes, densityRatio, gradientSum)
    contact = None
    if contactPoints is not None:
        contact = computeIISPHContactWeights(state, config, contactPoints, wallGradient, restVolumes)
        contactDiagonal = computeIISPHContactPressureDiagonal(state, config, pairs,
            restVolumes, wallGradient, gradientSum, contactPoints, contact)
        diagonal = diagonal + contactDiagonal
    return IISPHCoefficients(pairs, restVolumes, densityRatio, wallGradient, gradientSum, diagonal, contact)


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
    if previousPressure is None:
        previousPressure = torch.zeros_like(state.masses)
    boundaryIntegrals = computeIISPHBoundaryIntegrals(state, config, schemeConfig)
    coefficients = computeIISPHCoefficients(state, config, schemeConfig, adjacency,
        boundaryIntegrals=boundaryIntegrals, contactPoints=contactPoints)
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(state, config,
        schemeConfig, coefficients, boundaryIntegrals, contactPoints, gravity)
    velocity = state.velocities + dt * torch.where(
        fluid[:, None], acceleration, torch.zeros_like(acceleration))
    dt2 = dt * dt
    b = (1 - coefficients.densityRatio - dt * computeIISPHDensityRate(state, velocity, coefficients)
         - dt2 * computeIISPHDensityRate(state, boundaryAcceleration, coefficients))
    b = torch.where(fluid, b, 0)
    q = torch.where(fluid, 0.5 * previousPressure / rho0, 0)
    diagonal = dt2 * coefficients.diagonal
    valid = fluid & (diagonal.abs() > 1.e-9)
    denominator = torch.where(valid, diagonal, 1)
    omega = policy.relaxationFactor

    def iteration(old, _):
        off = dt2 * computeIISPHPressureOperator(state, old, coefficients) - diagonal * old
        new = torch.where(valid, ((1-omega)*old + omega*(b-off)/denominator).clamp_min(0), 0)
        errors = torch.where(new != 0, rho0*(diagonal*new + off-b), 0)
        return new, errors[fluid].mean(), None

    schedule = CheckSchedule(SimpleNamespace(), 'iisph', 'every',
                             policy.minIterations-1, policy.maxIterations)
    result = runStationary(iteration, q, maxIters=policy.maxIterations,
        tolerance=0.01*policy.maxErrorPercent*rho0, inclusive=True, schedule=schedule)
    if result.termination == 'nonfinite':
        raise FloatingPointError('Nonfinite IISPH pressure iteration')
    return IISPHResult(result.x*rho0,
        computeIISPHPressureAcceleration(state, result.x, coefficients)+boundaryAcceleration,
        result.iterations, result.stats[-1], result.termination, b)
