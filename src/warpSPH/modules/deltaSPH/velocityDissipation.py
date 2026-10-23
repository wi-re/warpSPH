"""Public entry point for the delta-SPH momentum-dissipation term `dvdt_diss`:
a thin wrapper that runs `computeVelocityDiffusionDeltaSPH` as a Laplacian
operation over `currentState.velocities` and forwards the viscosity
selection straight from `schemeConfig.diffusionParams` (`inviscid`, its
artificial-viscosity `inviscidAlpha`, `fixedSoundSpeed`, and the physical
kinematic viscosity `viscidNu`) — see `wp_viscosityDelta.py` for which of the
two forms is actually applied.
"""

from typing import Any, Optional, Union
import torch
from warpSPHCore import (AdjacencyList, CompactHashMap, OperationDirection,
    OperationProperties, SupportScheme, WarpOperation)
from warpSPHCore.profiling import record_function
from ...configurations.simulationConfig import SimulationConfig
from ...enumTypes import ViscosityTerm
from .wp_viscosityDelta import computeVelocityDiffusionDeltaSPH

__all__ = ['computeVelocityDiffusion']

def computeVelocityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], approachOnly: Optional[bool] = None, wall: Optional[Any] = None, operationMode: OperationDirection = OperationDirection.AllToAll, *, boundaryRepresentation: str = 'auto', wallSupport: Optional[float] = None) -> torch.Tensor:
    """Sampled walls use reflected fluid samples unless mDBC ghost rows already
    supply their velocities. Analytic walls use the solid Laplacian and local
    mirrored velocity without evaluating wall pressure. boundaryRepresentation
    selects the geometric path; a supplied WallState reuses its geometry.
    wallSupport supplies the host support already resolved during initialization.

    `approachOnly=False` lifts the approaching-neighbours clamp, turning the
    `inviscid=False` branch into the Monaghan & Gingold (1983) velocity
    Laplacian proper -- see `wp_viscosityDelta.py`'s docstring.
    An omitted approachOnly uses diffusionParams.approachOnly. Excluding
    boundary interactions removes their viscosity contribution entirely;
    it does not impose a reflected free-slip velocity field."""
    with record_function("[warpSPH] - (deltaSPH) - computeVelocityDiffusion"):
        if approachOnly is None:
            approachOnly = schemeConfig.diffusionParams.approachOnly
        dp = schemeConfig.diffusionParams
        morris = (getattr(dp, 'viscousTerm', ViscosityTerm.monaghanGingold) == ViscosityTerm.morris1997)
        rho0 = float(getattr(schemeConfig.fluid, 'restDensity', 1.0))
        # Projected viscosity divides by mean density. Convert the configured
        # kinematic coefficient to its reference dynamic coefficient mu0=rho0*nu.
        # Morris already forms dynamic viscosity from each particle's density.
        nu = dp.viscidNu if morris else rho0 * dp.viscidNu
        provider = getattr(schemeConfig, 'boundaryProvider', None)
        analytic = boundaryRepresentation == 'analytic' or (
            boundaryRepresentation == 'auto' and provider is not None)
        referenceVelocities = currentState.velocities
        if not analytic and wall is not False:
            from ..mdbc.viscosityVelocity import computeViscosityBoundaryVelocities
            referenceVelocities = computeViscosityBoundaryVelocities(currentState, config, schemeConfig)
        dvdt_diss = computeVelocityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Laplacian,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = operationMode,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryVelocities = currentState.velocities,
            referenceVelocities = referenceVelocities,
            inviscid = schemeConfig.diffusionParams.inviscid,
            c_s = schemeConfig.fluid.fixedSoundSpeed,
            alpha = rho0 * dp.inviscidAlpha,
            nu = nu,
            approachOnly = approachOnly,
            morris = morris,
        )
        if wall is False or not analytic:
            return dvdt_diss
        if wall is None and getattr(schemeConfig, 'wallViscosityClosure', 'mirror') == 'noslipMoment':
            from ..analyticBoundary import resolveWall
            wall = resolveWall(currentState, config, schemeConfig, adjacency)
        if wall is not None and getattr(schemeConfig, 'wallViscosityClosure', 'mirror') == 'noslipMoment':
            # the no-slip wall closure of the Morris viscosity (modules/analyticBoundary/wallViscosity.py): the wall moments balance the particle's own fluid sum
            dp = schemeConfig.diffusionParams
            if dp.inviscid or getattr(dp, 'viscousTerm', ViscosityTerm.monaghanGingold) != ViscosityTerm.morris1997:
                raise NotImplementedError("wallViscosityClosure='noslipMoment' is written for the Morris viscosity (inviscid=False, viscousTerm=morris1997)")
            from ..analyticBoundary.wallViscosity import wallNoSlipAcceleration
            return dvdt_diss + wallNoSlipAcceleration(currentState, config, schemeConfig, adjacency, wall, dvdt_diss)
        from ..analyticBoundary.viscosity import computeWallViscosity
        return dvdt_diss + computeWallViscosity(currentState, config, schemeConfig, wall,
            approachOnly=approachOnly, support=wallSupport)
