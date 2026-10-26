"""Local omni-derived projection composition at fixed geometry.

Pressure/source/operator/load implementations live in reusable modules;
ProjectedStep and the temporal recurrences live in warpSPHIntegrators.
Local relaxation, budgets, wall representation and closed-domain policies
differ from the pinned C++ oracle. See the incompressible source inventory.
Legacy experiment switches below are preserved by explicit policy snapshots."""

from __future__ import annotations
from warpSPHIntegrators.drivers.mechanics.projection import symplecticProjection
import os
from typing import Any
import torch
from ...modules.boundaryConditions.pinned import pinnedKeepWeight
from ...configurations.compute import resolveCheckSchedule
from ...modules.boundaryConditions import enforceDirichlet
from ...modules.density import computeDensities
from ...modules.analyticBoundary import resolveWall
from ...modules.incompressible.operators import (
    rebuildAdjacency as _rebuildAdjacency,
    scatterDivergence as _divergence,
    pressureAcceleration as _pressureAccel,
)
from ...modules.incompressible.loads import (
    bookProjectionWallLoads as _bookWallLoads,
    wallViscousPerBody as _wallViscousPerBody,
)
from ...modules.incompressible.omni import solveOmni
from ...modules.incompressible.policies import OmniPressurePolicy
from ...modules.incompressible.stages import (
    nonPressureAcceleration,
    projectedUpdate,
    requireProjectedStep,
)

__all__ = ["omniIncompressible_step"]
OMEGA = 0.3
DIVERGENCE_ITERATIONS = 3
DENSITY_MIN_ITERATIONS = 3
DENSITY_MAX_ITERATIONS = 256
RESIDUAL_FLOOR = -0.001
ALPHA_FLOOR = 1e-25
DAMPING = 0.0
VERBOSE_SOLVE = False
XSPH_FLUID = 0.05
XSPH_BOUNDARY = 0.0
_WALL_PRESSURE_ENV = os.environ.get("WARPSPH_WALL_PRESSURE_MODE")
WALL_PRESSURE_MODE = (
    (None if _WALL_PRESSURE_ENV.lower() == "none" else _WALL_PRESSURE_ENV)
    if _WALL_PRESSURE_ENV is not None
    else "shepard"
)
CD_SOURCE_PROJECT = "auto"
CD_PROJECT_THRESHOLD = 0.7
CD_SOLVER = "jacobi"
CD_KRYLOV_MAXITER = 256
CD_KRYLOV_RTOL = 0.001
CD_KRYLOV_RESTART = 50
CD_TIKHONOV = 0.0


def velocityFilter(
    state: Any, config: Any, adjacency: Any, fluidMask: torch.Tensor
) -> torch.Tensor:
    """omniSPH's XSPH velocity filter at the module constants `XSPH_FLUID` / `XSPH_BOUNDARY` (`divergenceFree`'s `xsphFilterScale` scales it); the filter itself is `modules/xsph`."""
    from ...modules.xsph import computeXSPH

    return computeXSPH(
        state,
        config,
        None,
        adjacency,
        fluidCoefficient=XSPH_FLUID,
        boundaryCoefficient=XSPH_BOUNDARY,
    )


def pressurePolicy():
    """Snapshot legacy experiment switches for this solve."""
    return OmniPressurePolicy(
        OMEGA=OMEGA,
        RESIDUAL_FLOOR=RESIDUAL_FLOOR,
        ALPHA_FLOOR=ALPHA_FLOOR,
        VERBOSE_SOLVE=VERBOSE_SOLVE,
        WALL_PRESSURE_MODE=WALL_PRESSURE_MODE,
        CD_SOURCE_PROJECT=CD_SOURCE_PROJECT,
        CD_PROJECT_THRESHOLD=CD_PROJECT_THRESHOLD,
        CD_SOLVER=CD_SOLVER,
        CD_KRYLOV_MAXITER=CD_KRYLOV_MAXITER,
        CD_KRYLOV_RTOL=CD_KRYLOV_RTOL,
        CD_KRYLOV_RESTART=CD_KRYLOV_RESTART,
        CD_TIKHONOV=CD_TIKHONOV,
    )


def _solve(*args, **kwargs):
    return solveOmni(*args, policy=pressurePolicy(), **kwargs)


def omniIncompressible_step(
    system: Any, dt: float, config: Any, schemeConfig: Any, verbose: bool = False
):
    requireProjectedStep(system, config, "omniIncompressible")
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity
    from ...modules.incompressible.compactProjection import resolveClosedPreset

    resolveClosedPreset(schemeConfig, config)
    provider = getattr(schemeConfig, "boundaryProvider", None)
    if provider is not None and any((getattr(rb, "dynamic", False) for rb in provider.rigidBodies)):
        raise NotImplementedError(
            "free (dynamic) analytic bodies are driven by the delta+ scheme only: the incompressible loops book the load but do not integrate the body from it"
        )
    solver = schemeConfig.solverConfig
    divCfg = solver.divergenceFreeSolver
    denCfg = solver.pressureSolver
    solver.akinciBoundaryVolume = True
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)
    fricPerBody = None
    if getattr(schemeConfig, "boundaryFriction", 0.0):
        from ...modules.xsph import computeBoundaryFriction

        if getattr(schemeConfig, "analyticWallLoads", False):
            fricPerBody = computeBoundaryFriction(st, config, schemeConfig, adjacency, perBody=True)
            st.velocities = st.velocities + fricPerBody.sum(0)
        else:
            st.velocities = st.velocities + computeBoundaryFriction(
                st, config, schemeConfig, adjacency
            )
    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    pPrior = st.pressures.clone()
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    viscousAcc = getattr(schemeConfig, "omniViscosity", False)
    accel = nonPressureAcceleration(
        system,
        dt,
        config,
        schemeConfig,
        adjacency,
        includeBodyForce=True,
        viscosity=viscousAcc,
        boundaryVelocitiesPrepared=False,
        maskViscosity=True,
    )
    accel = torch.where(fcol, accel, torch.zeros_like(accel))
    vEnter = st.velocities + dt * accel
    a_p_div, pDiv, nDiv, errDiv = _solve(
        st,
        config,
        schemeConfig,
        adjacency,
        fluid=fluid,
        rho0=rho0,
        vEnter=vEnter,
        warmStart=torch.zeros_like(st.densities),
        dt=dt,
        mode="divergence",
        minIters=DIVERGENCE_ITERATIONS,
        maxIters=DIVERGENCE_ITERATIONS,
        tol=divCfg.tolerance,
        checkSchedule=resolveCheckSchedule(divCfg, config),
    )
    accel = accel + a_p_div
    vEnter = st.velocities + dt * accel
    if getattr(schemeConfig, "densitySolve", True):
        a_p_rho, pRho, nRho, errRho = _solve(
            st,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            vEnter=vEnter,
            warmStart=0.5 * pPrior,
            dt=dt,
            mode="density",
            minIters=DENSITY_MIN_ITERATIONS,
            maxIters=DENSITY_MAX_ITERATIONS,
            tol=denCfg.tolerance,
            checkSchedule=resolveCheckSchedule(denCfg, config),
        )
    else:
        a_p_rho, pRho, nRho, errRho = (torch.zeros_like(accel), pDiv, 0, 0.0)
    accel = accel + a_p_rho
    _bookWallLoads(
        st,
        config,
        schemeConfig,
        adjacency,
        fluid=fluid,
        rho0=rho0,
        pDiv=pDiv,
        pRho=pRho,
        frictionPerBody=fricPerBody,
        dt=dt,
        viscousPerBody=(
            _wallViscousPerBody(
                st,
                config,
                schemeConfig,
                adjacency,
                resolveWall(st, config, schemeConfig, adjacency),
            )
            if viscousAcc and getattr(schemeConfig, "analyticWallLoads", False)
            else None
        ),
    )
    if getattr(schemeConfig, "xsphCoefficient", 0.0) or getattr(
        schemeConfig, "xsphBoundaryCoefficient", 0.0
    ):
        from ...modules.xsph import computeXSPH

        st.velocities = st.velocities + computeXSPH(st, config, schemeConfig, adjacency)

    def beforeDrift(st):
        if getattr(schemeConfig, "shifting", "none") == "none":
            return None
        from ...modules.shifting.fickian import computeFickianShift

        return computeFickianShift(
            st,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            dt=dt,
            velocities=st.velocities,
            wall=resolveWall(st, config, schemeConfig, adjacency),
        )

    symplecticProjection(st, dt, accel, mask=fcol, beforeDrift=beforeDrift)
    st.pressures = pRho
    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        pf = pRho[fluid]
        print(
            f"[omniIncompressible] t={system.t + dt:.4g}  DIV {nDiv:2d} it err {errDiv:.3g}   RHO {nRho:3d} it err {errRho:.3g}   |v|max {vmax:.4g}   p[{float(pf.min()):+.3g}, {float(pf.max()):+.3g}]"
        )
    update = projectedUpdate(system, dt, config, schemeConfig)
    keepBand = pinnedKeepWeight(st, config, schemeConfig)
    if keepBand is not None and getattr(update, "dvdt", None) is not None:
        update.dvdt = update.dvdt * keepBand
    return (update, adjacency, st, ([], [errDiv, errRho]))


_xsphFilter = velocityFilter  # compatibility for existing direct callers
