"""Stateless native IISPH coefficients and positive Jacobi pressure solve."""

from dataclasses import dataclass
from typing import NamedTuple
from types import SimpleNamespace
import math
import torch
from warpSPHCore.linalg import CheckSchedule, runStationary
from warpSPHCore.type_config import get_torch_precision
from .operators import (IISPHKernelData, computeIISPHKernelGeometry, launchIISPHKernel,
    pairArguments, geometryParameters, computeIISPHDensityRate, computeIISPHPressureOperator,
    computeIISPHPressureAcceleration)
from .contactWeights import IISPHContactWeights, computeIISPHContactWeights
from .boundaryPressureAcceleration import computeIISPHBoundaryPressureAcceleration


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
    """Return fixed-position tensor coefficients; no state or policy is retained.

    Static kind-1 samples use rest volumes, as do the kind-0 fluid unknowns.
    Density is read from state after Dirichlet conditions are enforced.
    The diagonal includes the nonlocal contact-pressure interpolation.
    Raw analytic geometry supplies the wall gradient; no gravity term is
    evaluated or retained by these coefficients.
    """
    rho0 = schemeConfig.fluid.restDensity
    if not math.isfinite(rho0) or rho0 <= 0 or len(state.positions) == 0:
        raise ValueError("IISPH requires a nonempty state and positive rest density")
    if state.positions.shape[1] not in (2, 3):
        raise NotImplementedError("IISPH supports 2D and 3D")
    fields = [state.positions, state.supports, state.masses, state.velocities]
    if any(v.requires_grad for v in fields):
        raise NotImplementedError("IISPH has no autograd contract")
    if any(v.dtype != get_torch_precision() or v.device != state.positions.device for v in fields):
        raise ValueError("IISPH fields must share the bootstrapped precision and device")
    if not bool((torch.isfinite(state.masses) & (state.masses > 0)).all()):
        raise ValueError("IISPH requires finite positive particle masses")
    if not bool((state.supports == state.supports[0]).all()) or not bool((state.supports > 0).all()):
        raise NotImplementedError("IISPH requires fixed common support")
    if not bool(((state.kinds == 0) | (state.kinds == 1)).all()):
        raise NotImplementedError("IISPH supports fluid and static boundary samples; no ghost rows")
    fluid = state.kinds == 0
    if not bool(fluid.any()):
        raise ValueError("IISPH requires fluid particles")
    if bool((state.velocities[~fluid] != 0).any()):
        raise NotImplementedError("IISPH supports static bodies only")
    pairs = computeIISPHKernelGeometry(state, config, adjacency)
    volume = (state.masses / rho0).contiguous()
    kinds = state.kinds.to(torch.int32).contiguous()
    densityRatio = (state.densities / rho0).contiguous()
    G = torch.empty_like(state.positions)
    c = torch.empty_like(state.positions)
    diagonal = torch.empty_like(state.masses)
    wallGradient = torch.zeros_like(state.positions)
    if boundaryIntegrals is not None:
        massFactor = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
        wallGradient = massFactor * boundaryIntegrals.out['G'].sum(0).to(state.positions)
    sampled = boundaryIntegrals is None
    launchIISPHKernel('coefficients', state, pairArguments(pairs) +
        [volume, kinds, sampled, wallGradient, densityRatio, G, c, diagonal])
    contact = None
    if contactPoints is not None:
        points = contactPoints.contiguous()
        contact = computeIISPHContactWeights(state, config, points, G, volume)
        launchIISPHKernel('contact_diagonal', state, pairArguments(pairs) +
            [volume, kinds, state.positions, points, state.supports,
             G, c, contact.denominator, *geometryParameters(state, config), diagonal])
    return IISPHCoefficients(pairs, volume, densityRatio, G, c, diagonal, contact)


def solveIISPH(state, config, schemeConfig, velocity, dt, adjacency, previousPressure=None, *,
               boundaryIntegrals=None, contactPoints=None, gravity=None):
    """Assemble and solve IISPH at fixed geometry with physical-pressure carry.

    The signed statistic uses the new diagonal and old off-diagonal iterate,
    averaged over all fluid rows. Tolerance is a percentage of rest density.
    Pressure coefficients, contact interpolation and gravity extrapolation
    are local to this solve. The known wall acceleration contributes to
    both the source and the final pressure acceleration.
    No input field is modified; the caller applies the returned acceleration.
    """
    policy = schemeConfig.iisph
    rho0 = schemeConfig.fluid.restDensity
    fluid = state.kinds == 0
    policy.validate()
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("IISPH requires a finite positive timestep")
    if previousPressure is None:
        previousPressure = torch.zeros_like(state.masses)
    if velocity.requires_grad or previousPressure.requires_grad:
        raise NotImplementedError("IISPH has no autograd contract")
    coefficients = computeIISPHCoefficients(state, config, schemeConfig, adjacency,
        boundaryIntegrals=boundaryIntegrals, contactPoints=contactPoints)
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(state, config,
        schemeConfig, coefficients, boundaryIntegrals, contactPoints, gravity)
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
