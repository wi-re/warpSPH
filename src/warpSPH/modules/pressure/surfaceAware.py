"""Case/scheme-facing wrapper around `wp_surfaceAware.computePressureSurfaceAwareWarp`:
assembles the operation properties (`SupportScheme.SuperSymmetric`) and
forwards the current pressures, the configured pressure-force formulation
(`schemeConfig.pressureForceTerm`) and the surface-indicator mask consumed
by the Antuono surface-aware term.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from .wp_surfaceAware import computePressureSurfaceAwareWarp

__all__ = ['computePressureForceSurfaceAware', 'antuonoSwitch']


def antuonoSwitch(pressures: torch.Tensor, surfaceIndicators: torch.Tensor) -> torch.Tensor:
    """The Antuono switch s of the pressure force (`p_i + s p_j` form): +1 where the pressure is non-negative or the particle is a free-surface particle, -1 elsewhere."""
    return torch.where((pressures >= 0) | (surfaceIndicators == 1), torch.ones_like(pressures), -torch.ones_like(pressures))

def computePressureForceSurfaceAware(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], renormalizationState: Optional[Any] = None, wall: Optional[Any] = None) -> torch.Tensor:
    """`wall`: the analytic boundary's `WallState` (`modules/analyticBoundary`) adds the wall's pressure force with the same Antuono switch as the fluid pairs;
    `None`: resolved from `schemeConfig.boundaryProvider` (`resolveWall`), no wall term for boundary particles."""
    with record_function("[warpSPH] - computePressureForceSurfaceAware"):
        dvdt = computePressureSurfaceAwareWarp(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                supportMode = SupportScheme.SuperSymmetric,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryPressures = currentState.pressures,
            pressureTerm = schemeConfig.pressureForceTerm,
            querySurfaceMask = currentState.surfaceIndicators,
            # `renormalizationState` (`None` unless the caller opts in via
            # `schemeConfig.pressureForceRenormalized`) makes
            # `wp_surfaceAware.py`'s existing but previously-unused
            # `useGradientRenormalization`/`Li` path apply the same L matrix
            # `deltaSPH.py` already computes for `gradRhoL` to this kernel
            # gradient too. See `configurations/weaklyCompressible.py`'s
            # `pressureForceRenormalized` docstring.
            renormalizationState = renormalizationState,
        )
        from ..analyticBoundary import resolveWall
        wall = resolveWall(currentState, config, schemeConfig, adjacency, wall)
        if wall is not None:
            from ..analyticBoundary import wallPressureAcceleration
            sw = antuonoSwitch(currentState.pressures, currentState.surfaceIndicators)
            dvdt = dvdt + wallPressureAcceleration(wall, currentState.pressures, sw, currentState.densities, wallMass=wall.wm, h=wall.support)
            if getattr(schemeConfig, 'pressureConsistent', False):
                # the boundaries repo's `pressureConsistent`: the force a UNIFORM pressure would exert (the fluid pairs with the Antuono switch s, the wall with its clamp) is the static wall-consistency residual
                # S_i = sum_j V_j grad W_ij + G_i times (1 + s) P_i: spurious where the layout does not fill to the wall (a cut lattice on a curved wall). Removing it leaves the difference form,
                # exact for a uniform pressure of any sign and level; the wall then acts through its hydrostatic part only. Not at a free surface (the truncation there is physical).
                from ..analyticBoundary.wallTerms import F64
                apparent = currentState.masses / currentState.densities
                Sf = warpOperation(currentState, OperationProperties(kernel=config.kernel, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, supportMode=SupportScheme.Gather),
                                   queryValues=apparent * currentState.densities / currentState.masses, domain=config.domain, adjacency=adjacency)
                interior = (currentState.surfaceIndicators == 0).to(F64)[:, None]
                P = currentState.pressures.to(F64)
                G = wall.G.sum(0).to(F64)
                rho = currentState.densities.to(F64)[:, None]
                dvdt = dvdt + (interior * (((1.0 + sw.to(F64)) * P)[:, None] * Sf.to(F64) + (P.clamp(min=0) + sw.to(F64) * P)[:, None] * G) / rho).to(dvdt.dtype)
        return dvdt
