"""Local Band pressure-boundary projection composition.

The extended fluid/boundary PPE lives in modules.incompressible.band and
pressureBoundaries; time advancement belongs to warpSPHIntegrators.
Full-band volume estimation, active-row thresholds, Tikhonov and auto gauge
are local variants. The residual is a floor-clipped signed mean, not an
absolute mean. See docs/incompressible/INVENTORY_2026-10-21.md."""

from __future__ import annotations
from warpSPHIntegrators.drivers.mechanics.projection import symplecticProjection
from typing import Any
import torch
from warpSPHCore import SupportScheme
from ...modules.boundaryConditions import enforceDirichlet
from ...modules.density import computeDensities
from ...modules.incompressible.pressureBoundaries import bandBoundaryUnknownMask
from ...modules.incompressible.operators import rebuildAdjacency as _rebuildAdjacency
from ...modules.incompressible.band import solvePressureBoundaries
from ...modules.incompressible.policies import BandPressurePolicy
from ...modules.incompressible.stages import (
    nonPressureAcceleration,
    projectedUpdate,
    requireProjectedStep,
)

__all__ = ["band2018pb_step"]
OMEGA_FLUID = 0.1
DENSITY_MIN_ITERATIONS = 8
DENSITY_MAX_ITERATIONS = 256
ALPHA_FLOOR = 1e-25
RESIDUAL_FLOOR = -0.001
DAMPING = 0.0
DIAG_TIKHONOV = 0.05
CLOSED_DOMAIN_GAUGE = "auto"
NULL_MODE_THRESHOLD = 0.25
XSPH_FLUID = 0.0
XSPH_BOUNDARY = 0.0


def pressurePolicy():
    """Snapshot legacy experiment switches for this solve."""
    return BandPressurePolicy(
        OMEGA_FLUID=OMEGA_FLUID,
        DENSITY_MIN_ITERATIONS=DENSITY_MIN_ITERATIONS,
        DENSITY_MAX_ITERATIONS=DENSITY_MAX_ITERATIONS,
        ALPHA_FLOOR=ALPHA_FLOOR,
        RESIDUAL_FLOOR=RESIDUAL_FLOOR,
        DIAG_TIKHONOV=DIAG_TIKHONOV,
        CLOSED_DOMAIN_GAUGE=CLOSED_DOMAIN_GAUGE,
        NULL_MODE_THRESHOLD=NULL_MODE_THRESHOLD,
    )


def _solve(*args, **kwargs):
    return solvePressureBoundaries(*args, policy=pressurePolicy(), **kwargs)


def _tol(schemeConfig: Any) -> float:
    return float(schemeConfig.solverConfig.pressureSolver.tolerance)


def _xsphFilter(state, config, adjacency, fluidMask):
    """Compatibility adapter for explicit legacy filter coefficients."""
    from ...modules.xsph.interpolated import interpolatedVelocityFilter

    return interpolatedVelocityFilter(
        state,
        config,
        adjacency,
        fluidMask,
        fluidCoefficient=XSPH_FLUID,
        boundaryCoefficient=XSPH_BOUNDARY,
    )


def band2018pb_step(system: Any, dt: float, config: Any, schemeConfig: Any, verbose: bool = False):
    requireProjectedStep(system, config, "band2018pb")
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)
    boundaryUnknown = bandBoundaryUnknownMask(st, config, adjacency, rho0)
    boundary = boundaryUnknown
    solveRows = fluid | boundaryUnknown
    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    pPrior = st.pressures.clone()
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    accel = nonPressureAcceleration(
        system, dt, config, schemeConfig, adjacency, includeBodyForce=False,
        viscosity=bool(getattr(schemeConfig, 'bandViscosity', False)), maskViscosity=True
    )
    accel = torch.where(fcol, accel, torch.zeros_like(accel))
    vStar = st.velocities + dt * accel
    a_p, pSolve, nIt, err = _solve(
        st,
        config,
        schemeConfig,
        adjacency,
        solveRows=solveRows,
        fluid=fluid,
        rho0=rho0,
        vStar=vStar,
        warmStart=0.5 * pPrior,
        dt=dt,
    )
    accel = accel + a_p
    if XSPH_FLUID != 0.0 or XSPH_BOUNDARY != 0.0:
        st.velocities = st.velocities + _xsphFilter(st, config, adjacency, fluid)
    symplecticProjection(st, dt, accel, mask=fcol, damping=DAMPING)
    st.pressures = pSolve
    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        pf = pSolve[fluid]
        pb = pSolve[boundary]
        print(
            f"[band2018pb] t={system.t + dt:.4g}  RHO {nIt:3d} it err {err:.3g}   |v|max {vmax:.4g}   p_f[{float(pf.min()):+.3g}, {float(pf.max()):+.3g}]  p_b[{(float(pb.min()) if pb.numel() else 0):+.3g}, {(float(pb.max()) if pb.numel() else 0):+.3g}]"
        )
    update = projectedUpdate(system, dt, config, schemeConfig)
    return (update, adjacency, st, ([], [err]))
