"""Fixed-geometry density and divergence constraints on native IISPH C and F.

Pressure is physical p, and the iterative unknown is q=p/rho0. Warm histories
are physical p*dt**2 (density) and p_v*dt (divergence), never sound speeds.
The linear operator excludes the known hydrostatic wall acceleration. Only
the density stage receives that acceleration, in both its source and result.
"""

from types import SimpleNamespace
import torch
from warpSPHCore.linalg import CheckSchedule, runStationary
from ..iisph.solver import IISPHResult, computeIISPHSource, applyDensitySourcePolicy
from ..iisph import computeIISPHDensityRate, computeIISPHPressureOperator, computeIISPHPressureAcceleration
from ..iisph.wp_pressureDiagonal import computeIISPHPressureDiagonal


def solveDFSPHProjection(state, config, schemeConfig, dt, adjacency,
                         restVolumes, densityRatio, wallGradient, gradientSum, pressureDiagonal, *,
                         mode, velocity, previousPressure=None, pressureImpulse=None,
                         boundaryAcceleration=None, contactPoints=None,
                         contactAdjacency=None, contactDenominator=None, deficient=None,
                         pressureBoundaryGeometry=None):
    """Solve one constraint without mutating input geometry or velocity.

    Density solves dt² CF(q)=1-s-dt C(v*)-dt² C(a_wall), divergence solves
    dt CF(q_v)=-C(v). Nonnegative solves use a compression-only input residual;
    signed solves use its absolute value. Tolerances are percentages of rho0,
    divided by dt for divergence. The deficient-row reference policy suppresses
    initialization and the error statistic, but retains the raw source/update.
    Weak rest-volume factors are disabled before timestep/density scaling,
    as in the pinned reference; the remaining exact diagonal is unchanged.
    The minimum density ratio pins clamped divergence pressure below
    that ratio, allowing sampling recovery without masking neighbour forces.
    """
    if mode not in ('density', 'divergence'):
        raise ValueError(f'Unknown DFSPH projection mode: {mode!r}')
    if pressureBoundaryGeometry is not None:
        from .pressureBoundaries import solvePressureBoundaryProjection
        return solvePressureBoundaryProjection(state, config, schemeConfig, dt, adjacency,
            pressureBoundaryGeometry, mode=mode, velocity=velocity,
            pressureImpulse=pressureImpulse, previousPressure=previousPressure)
    if schemeConfig.dfsph.pressureBoundary != 'native':
        raise ValueError('Band DFSPH projection requires its pressure-boundary geometry')
    density = mode == 'density'
    policy = schemeConfig.dfsph.density if density else schemeConfig.dfsph.divergence
    rho0 = schemeConfig.fluid.restDensity
    fluid = state.kinds == 0
    active = fluid
    if not density and policy.clampPressure:
        active = fluid & (densityRatio >= schemeConfig.dfsph.minDivergenceDensityRatio)
    scale = dt*dt if density else dt
    if boundaryAcceleration is None:
        boundaryAcceleration = torch.zeros_like(state.velocities)
    if density:
        if getattr(schemeConfig.iisph, 'densitySourcePolicy', 'absolute') == 'periodicMeanFree' and policy.clampPressure:
            raise ValueError('periodicMeanFree requires a signed DFSPH density solve')
        source = computeIISPHSource(state, config, adjacency, velocity, dt,
            restVolumes, densityRatio, wallGradient, boundaryAcceleration)
        source = applyDensitySourcePolicy(source, state, config,
            getattr(schemeConfig.iisph, 'densitySourcePolicy', 'absolute'))
    else:
        source = torch.where(fluid, -computeIISPHDensityRate(
            state, config, adjacency, velocity, restVolumes, wallGradient), 0)
    # Divergence is a homogeneous pressure correction. A known physical wall
    # acceleration is applied once by density projection, never a second time.
    known = boundaryAcceleration if density else torch.zeros_like(state.velocities)
    if deficient is None or density or not policy.clampPressure:
        deficient = torch.zeros_like(fluid)

    if pressureImpulse is not None:
        previousPressure = pressureImpulse / scale
    if previousPressure is None or not schemeConfig.dfsph.warmStart:
        previousPressure = torch.zeros_like(state.densities)
    q = torch.where(fluid, .5*previousPressure/rho0, 0)
    if policy.clampPressure:
        q = torch.where((source < 0) & ~deficient, q.clamp_min(0), 0)

    diagonal = scale*pressureDiagonal
    # The reference factor denominator is |S|² + Vi sum_j Vj |grad W_ij|².
    # Our physical-pressure diagonal divides it by s_i² and may additionally
    # include contact interpolation. Remove both differences for the cutoff;
    # do not compare dt*diag (DF) or dt²*diag (CD) against a fixed epsilon.
    threshold = schemeConfig.dfsph.weakFactorThreshold
    baseDiagonal = (computeIISPHPressureDiagonal(state, config, adjacency,
        restVolumes, densityRatio, gradientSum) if contactPoints is not None
        else pressureDiagonal)
    factorDenominator = -baseDiagonal * densityRatio.square()
    valid = active & (factorDenominator > threshold) & (pressureDiagonal != 0)
    denominator = torch.where(valid, diagonal, 1)
    q = torch.where(valid, q, 0)
    fluidCount = fluid.sum().clamp_min(1)

    def iteration(old, _):
        applied = scale*computeIISPHPressureOperator(state, config, adjacency, old,
            restVolumes, densityRatio, gradientSum, wallGradient,
            contactPoints=contactPoints, contactAdjacency=contactAdjacency,
            contactDenominator=contactDenominator)
        residual = source-applied
        new = old + policy.relaxationFactor*residual/denominator
        if policy.clampPressure:
            new = new.clamp_min(0)
            error = (-residual).clamp_min(0)
        else:
            error = residual.abs()
        new = torch.where(valid, new, 0)
        # Keep the original all-fluid normalization, matching the trial rule.
        # Disabled constraints cannot recreate pressure or delay convergence.
        error = torch.where(active & ~deficient, error, 0)
        return new, rho0*error.sum()/fluidCount, None

    schedule = CheckSchedule(SimpleNamespace(), f'dfsph.{mode}', 'every',
                             policy.minIterations-1, policy.maxIterations)
    tolerance = .01*policy.maxErrorPercent*rho0 / (1 if density else dt)
    result = runStationary(iteration, q, maxIters=policy.maxIterations,
                           tolerance=tolerance, inclusive=True, schedule=schedule)
    if result.termination == 'nonfinite':
        raise FloatingPointError(f'Nonfinite DFSPH {mode} pressure iteration')
    acceleration = computeIISPHPressureAcceleration(state, config, adjacency, result.x,
        restVolumes, densityRatio, gradientSum, wallGradient,
        contactPoints=contactPoints, contactAdjacency=contactAdjacency,
        contactDenominator=contactDenominator) + known
    return IISPHResult(result.x*rho0, acceleration, result.iterations,
                       result.stats[-1], result.termination, source)
