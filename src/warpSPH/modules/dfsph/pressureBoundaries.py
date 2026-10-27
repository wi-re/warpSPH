"""Opt-in Band volume-centric pressure unknowns for native DFSPH stages.

Band et al. 2018, Eqs. 8–10, 16–20; divergence uses the same CF with
dt CF(q)=-C(v). This two-constraint composition is a local extension.
The sampled *volume band* uses nominal V0 and full kernel sums, rather than
the paper's one-layer cubic-wall gamma/beta calibration. No legacy diagonal
shift, median-row cutoff, automatic gauge, source floor or volume mutation.
Static one-way sampled walls only. Analytic/contact wall closures stay native.
"""
from copy import copy
from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace
import torch
from warpSPHCore import (OperationProperties, WarpOperation, GradientScheme,
                        SupportScheme, OperationDirection, warpOperation)
from warpSPHCore.linalg import CheckSchedule, runStationary
from ..iisph.solver import IISPHResult, applyDensitySourcePolicy
from ..incompressible.wp_alpha import computeAlphaWarp


@dataclass
class PressureBoundaryGeometry:
    proxy: object
    restVolumes: torch.Tensor
    volumes: torch.Tensor
    ratios: torch.Tensor
    fluid: torch.Tensor
    rows: torch.Tensor
    diagonal: torch.Tensor

    def acceleration(self, config, adjacency, q):
        gradient = warpOperation(self.proxy, OperationProperties(kernel=config.kernel,
            operation=WarpOperation.Gradient, gradientMode=GradientScheme.Summation,
            supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
            queryValues=q, domain=config.domain, adjacency=adjacency)
        return torch.where(self.fluid[:, None], -gradient/self.ratios[:, None], 0)

    def rate(self, config, adjacency, field):
        return -warpOperation(self.proxy, OperationProperties(kernel=config.kernel,
            operation=WarpOperation.Divergence, gradientMode=GradientScheme.Difference,
            supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
            queryValues=field, domain=config.domain, adjacency=adjacency,
            consistentDivergence=False)

    def apply(self, config, adjacency, q):
        return self.rate(config, adjacency, self.acceleration(config, adjacency, q))


def buildPressureBoundaryGeometry(state, config, schemeConfig, adjacency, *, analyticProvider=None):
    if analyticProvider is not None or schemeConfig.iisph.boundaryRepresentation == 'analytic':
        raise ValueError('Band DFSPH pressure unknowns require sampled walls, not analytic integrals')
    # Kind-2 mDBC interpolation queries are present in normal particle tanks.
    # AllToAll excludes them as neighbours; they never become unknowns.
    if bool(((state.kinds < 0) | (state.kinds > 2)).any()) or bool((state.velocities[state.kinds == 1] != 0).any()):
        raise ValueError('Band DFSPH supports static kind-1 sampled walls only')
    if not bool((torch.isfinite(state.supports) & (state.supports > 0)
                 & (state.supports == state.supports[0])).all()):
        raise ValueError('Band DFSPH currently requires finite positive uniform support')
    rho0 = schemeConfig.fluid.restDensity
    V0 = state.masses/rho0
    V = bandActualVolumes(state, config, adjacency, V0, rho0)
    proxy = copy(state)
    proxy.densities = state.masses/V
    fluid = state.kinds == 0
    rows = fluid | bandBoundaryUnknownMask(state, config, adjacency, rho0)
    # Reuse only the bare pair primitive: the legacy wrapper automatically
    # adds analytic-provider terms even when a sampled band is present.
    # Eq. 19 includes self motion; Eq. 20 removes it on fixed solid rows.
    properties = OperationProperties(kernel=config.kernel, supportMode=SupportScheme.Scatter,
                                     operationMode=OperationDirection.AllToAll)
    diagonal = -rho0*computeAlphaWarp(proxy, properties, config.domain,
                                    queryApparentAreas=V, adjacency=adjacency)
    G = warpOperation(proxy, OperationProperties(kernel=config.kernel,
        operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
        supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
        queryValues=torch.ones_like(V), domain=config.domain, adjacency=adjacency)
    diagonal = diagonal+torch.where(fluid, 0, (V/V0)*G.square().sum(-1))
    return PressureBoundaryGeometry(proxy, V0, V, V0/V, fluid, rows, diagonal)


def solvePressureBoundaryProjection(state, config, schemeConfig, dt, adjacency, geometry, *,
                                    mode, velocity, pressureImpulse=None, previousPressure=None):
    if mode not in ('density', 'divergence'):
        raise ValueError(f'Unknown Band DFSPH projection mode: {mode!r}')
    density = mode == 'density'
    policy = schemeConfig.dfsph.density if density else schemeConfig.dfsph.divergence
    rho0 = schemeConfig.fluid.restDensity
    scale = dt*dt if density else dt
    source = -geometry.rate(config, adjacency, velocity)
    if density:
        source = 1-geometry.ratios+dt*source
        if getattr(schemeConfig.iisph, 'densitySourcePolicy', 'absolute') == 'periodicMeanFree' and policy.clampPressure:
            raise ValueError('periodicMeanFree requires signed pressure')
        source = applyDensitySourcePolicy(source, state, config, schemeConfig.iisph.densitySourcePolicy)
    active = geometry.rows
    if not density and policy.clampPressure:
        active = active & (~geometry.fluid | (geometry.ratios >= schemeConfig.dfsph.minDivergenceDensityRatio))
    valid = active & (-geometry.diagonal > schemeConfig.dfsph.weakFactorThreshold)
    source = torch.where(geometry.rows, source, 0)
    denominator = torch.where(valid, scale*geometry.diagonal, 1)
    if pressureImpulse is not None:
        previousPressure = pressureImpulse/scale
    q = torch.zeros_like(source)
    if previousPressure is not None and schemeConfig.dfsph.warmStart:
        q = .5*previousPressure/rho0
    if policy.clampPressure:
        q = torch.where(source < 0, q.clamp_min(0), 0)
    q = torch.where(valid, q, 0)
    count = geometry.rows.sum().clamp_min(1)

    def iteration(old, _):
        residual = source-scale*geometry.apply(config, adjacency, old)
        new = old+policy.relaxationFactor*residual/denominator
        error = residual.abs()
        if policy.clampPressure:
            new = new.clamp_min(0)
            error = (-residual).clamp_min(0)
        new = torch.where(valid, new, 0)
        return new, rho0*torch.where(active, error, 0).sum()/count, None

    schedule = CheckSchedule(SimpleNamespace(), f'dfsph.band2018.{mode}', 'every',
                             policy.minIterations-1, policy.maxIterations)
    result = runStationary(iteration, q, maxIters=policy.maxIterations,
        tolerance=.01*policy.maxErrorPercent*rho0/(1 if density else dt),
        inclusive=True, schedule=schedule)
    if result.termination == 'nonfinite':
        raise FloatingPointError(f'Nonfinite Band DFSPH {mode} pressure iteration')
    return IISPHResult(result.x*rho0, geometry.acceleration(config, adjacency, result.x),
                       result.iterations, result.stats[-1], result.termination, source)


MIN_FLUID_CONTACT = 0.02

def _bareKernelSum(state: Any, config: Any, adjacency: Any,
                   direction: OperationDirection,
                   weights: torch.Tensor | None = None) -> torch.Tensor:
    """`sum_{j in `direction`} w_j W_ij` via an `Interpolate` whose `m_j/rho_j`
    weight is divided back out. `weights=None` -> `w_j = 1` (a plain kernel
    sum); otherwise `w_j = weights[j]`. Independent of the actual density value
    (the `rho_j` cancels), so it is valid before the volume injection."""
    inv = state.densities / state.masses
    ref = inv if weights is None else weights * inv
    return warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.SuperSymmetric,
            operationMode=direction),
        referenceValues=ref, domain=config.domain, adjacency=adjacency)




def bandActualVolumes(state: Any, config: Any, adjacency: Any,
                      V0: torch.Tensor, rho0: float) -> torch.Tensor:
    """`V_i = V0_i / sum_{all j} V0_j W_ij` -- Eq. 13, applied to fluid and
    boundary rows alike. Near-complete support (a fluid row in the bulk, or a
    boundary interface row with a full band behind it) gives `V_i ~ V0_i`."""
    denom = _bareKernelSum(state, config, adjacency,
                           OperationDirection.AllToAll, weights=V0)
    return V0 / denom.clamp_min(1e-12)


def bandBoundaryUnknownMask(state: Any, config: Any, adjacency: Any,
                            rho0: float) -> torch.Tensor:
    """`kind == 1` rows with enough fluid contact to be pressure unknowns --
    `sum_bf V0_f W_bf > MIN_FLUID_CONTACT`. The 3-4 band layers behind the
    interface get no fluid neighbours (zero Eq. 20 diagonal) and are excluded;
    their pressure stays 0 and is never read by a fluid row anyway."""
    boundary = state.kinds == 1
    V0f = state.masses / rho0
    kf = _bareKernelSum(state, config, adjacency,
                        OperationDirection.FluidToBoundary, weights=V0f)
    return boundary & (kf > MIN_FLUID_CONTACT)
