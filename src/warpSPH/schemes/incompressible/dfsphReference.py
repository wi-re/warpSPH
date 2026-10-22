"""Local composed DFSPH reference variant, with explicit CD/drift/DF stages.

The pressure equation lives in modules.incompressible.reference, temporal
recurrences in warpSPHIntegrators. This is not literal SPlisHSPlasH ordering
or default policy: see docs/incompressible/INVENTORY_2026-10-21.md and the
historic DFSPH improvement plan for measured numerical limitations.
Legacy module switches remain compatibility adapters for existing probes."""

from __future__ import annotations
from warpSPHIntegrators.drivers.mechanics.projection import kick, drift
from typing import Any
import torch
from warpSPHCore import OperationProperties, SupportScheme, WarpOperation, warpOperation
from ...configurations import BoundaryPressureMode
from ...modules.boundaryConditions import enforceDirichlet
from ...modules.density import computeDensities
from ...modules.incompressible.consistent import applyConsistentCoupling
from ...modules.mdbc import computeBoundaryVelocities
from ...modules.surfaceDetection import detectFreeSurface
from ...modules.incompressible.operators import rebuildAdjacency as _rebuildAdjacency
from ...modules.incompressible.reference import (
    solveReference as _jacobiSolve,
    computeReferenceFactor as _factor,
    referenceDensityRate as _drhodt,
    _pressureAccel,
)
from ...modules.incompressible.loads import (
    bookProjectionWallLoads as _bookWallLoads,
    wallViscousPerBody as _wallViscousPerBody,
)
from ...modules.incompressible.stages import (
    nonPressureAcceleration,
    projectedUpdate,
    requireProjectedStep,
)

__all__ = ["dfsphReference_step", "iisph_step"]
FREE_SURFACE_GAUGE = False
DAMPED_WARM_START = False
XSPH_FLUID_EPSILON = 0.0
XSPH_BOUNDARY_EPSILON = 0.0
SKIP_DIVERGENCE_SOLVE = False
WALL_PRESSURE_MODE = None
WALL_PRESSURE_ON_DIVERGENCE = False


def _xspH(state, config, schemeConfig, adjacency, fluidMask):
    """Compatibility adapter for explicit legacy filter coefficients."""
    from ...modules.xsph.interpolated import referenceFilterAcceleration

    return referenceFilterAcceleration(
        state,
        config,
        adjacency,
        fluidMask,
        fluidCoefficient=XSPH_FLUID_EPSILON,
        boundaryCoefficient=XSPH_BOUNDARY_EPSILON,
    )


def dfsphReference_step(
    system: Any,
    dt: float,
    config: Any,
    schemeConfig: Any,
    verbose: bool = False,
    skipDivergence: bool = None,
):
    requireProjectedStep(system, config, "dfsphReference")
    if skipDivergence is None:
        skipDivergence = SKIP_DIVERGENCE_SOLVE
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity
    solver = schemeConfig.solverConfig
    psCfg = solver.pressureSolver
    dfCfg = solver.divergenceFreeSolver
    solver.akinciBoundaryVolume = True
    if solver.akinciBoundaryVolumeScale == 1.0:
        solver.akinciBoundaryVolumeScale = 2.0
    if psCfg.maxIterations < 100:
        psCfg.maxIterations = 100
    if dfCfg.maxIterations < 100:
        dfCfg.maxIterations = 100
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)
    provider = getattr(schemeConfig, "boundaryProvider", None)
    if provider is not None:
        if any(
            (
                bool((rb.linearVelocity != 0).any())
                or bool(rb.angularVelocity != 0)
                or getattr(rb, "dynamic", False)
                for rb in provider.rigidBodies
            )
        ):
            raise NotImplementedError(
                "dfsphReference / iisph on analytic walls are written for static bodies (moving and free bodies: omniIncompressible, divergenceFree)"
            )
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
    if getattr(schemeConfig, "xsphCoefficient", 0.0) or getattr(
        schemeConfig, "xsphBoundaryCoefficient", 0.0
    ):
        from ...modules.xsph import computeXSPH

        st.velocities = st.velocities + computeXSPH(st, config, schemeConfig, adjacency)
    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    kappa = st.pressures.clone()
    if st.soundspeeds is None or st.soundspeeds.shape != st.densities.shape:
        st.soundspeeds = torch.zeros_like(st.densities)
    kappaV = st.soundspeeds.clone()
    if DAMPED_WARM_START and system.t == 0.0:
        kappa = torch.zeros_like(kappa)
        kappaV = torch.zeros_like(kappaV)
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    st.velocities = computeBoundaryVelocities(st, config, schemeConfig, adjacency)
    extra = (
        (lambda: _xspH(st, config, schemeConfig, adjacency, fluid))
        if XSPH_FLUID_EPSILON != 0.0 or XSPH_BOUNDARY_EPSILON != 0.0
        else None
    )
    a_nonp = nonPressureAcceleration(
        system,
        dt,
        config,
        schemeConfig,
        adjacency,
        includeBodyForce=True,
        viscosity=True,
        extraAcceleration=extra,
    )
    st.velocities = kick(st.velocities, dt, a_nonp, mask=fcol)
    vEnter = st.velocities.clone()
    with applyConsistentCoupling(
        st, config, schemeConfig, adjacency, BoundaryPressureMode.consistent
    ):
        a_p_cd, kappa, nCd, errCd = _jacobiSolve(
            st,
            config,
            schemeConfig,
            adjacency,
            vEnter=vEnter,
            rho0=rho0,
            warmStart=kappa,
            opDt=dt * dt,
            solverCfg=psCfg,
            minIters=max(psCfg.minIterations, 2),
            fluidMask=fluid,
            mode="density",
            warmStartDamped=DAMPED_WARM_START,
            wallPressure=WALL_PRESSURE_MODE,
        )
    st.velocities = kick(vEnter, dt, a_p_cd)
    st.positions = drift(st.positions, dt, st.velocities, mask=fcol)
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)
    surfaceMask = None
    if skipDivergence:
        a_p_df = torch.zeros_like(st.velocities)
        nDf, errDf = (0, 0.0)
    else:
        if FREE_SURFACE_GAUGE:
            _, fs, _, _ = detectFreeSurface(
                st,
                config,
                schemeConfig,
                schemeConfig.surfaceDetectionConfig,
                adjacency,
                returnNormals=False,
            )
            surfaceMask = fluid & (fs > 0.5)
        vEnterDf = st.velocities.clone()
        with applyConsistentCoupling(
            st, config, schemeConfig, adjacency, BoundaryPressureMode.consistent
        ):
            a_p_df, kappaV, nDf, errDf = _jacobiSolve(
                st,
                config,
                schemeConfig,
                adjacency,
                vEnter=vEnterDf,
                rho0=rho0,
                warmStart=kappaV,
                opDt=dt,
                solverCfg=dfCfg,
                minIters=max(dfCfg.minIterations, 1),
                fluidMask=fluid,
                mode="divergence",
                surfaceMask=surfaceMask,
                warmStartDamped=DAMPED_WARM_START,
                wallPressure=WALL_PRESSURE_MODE if WALL_PRESSURE_ON_DIVERGENCE else None,
            )
        st.velocities = kick(vEnterDf, dt, a_p_df)
    if provider is not None:
        from ...modules.analyticBoundary import resolveWall

        _bookWallLoads(
            st,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            pDiv=kappaV,
            pRho=kappa,
            frictionPerBody=fricPerBody,
            dt=dt,
            densityApplied=True,
            divergenceHasWall=not skipDivergence,
            viscousPerBody=(
                _wallViscousPerBody(
                    st,
                    config,
                    schemeConfig,
                    adjacency,
                    resolveWall(st, config, schemeConfig, adjacency),
                )
                if getattr(schemeConfig, "analyticWallLoads", False)
                and (not schemeConfig.diffusionParams.inviscid)
                else None
            ),
        )
    if DAMPED_WARM_START:
        st.pressures = kappa * dt * dt
        st.soundspeeds = kappaV * dt
    else:
        st.pressures = kappa
        st.soundspeeds = kappaV
    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        kf = kappa[fluid]
        gaugeTag = f"   fsGauge={int(surfaceMask.sum())}" if surfaceMask is not None else ""
        print(
            f"[dfsphReference] t={system.t + dt:.4g}  CD {nCd:2d} it err {errCd:.3g}   DF {nDf:2d} it err {errDf:.3g}   |v|max {vmax:.4g}   kappa[{float(kf.min()):+.3g}, {float(kf.max()):+.3g}]{gaugeTag}"
        )
    update = projectedUpdate(system, dt, config, schemeConfig)
    return (update, adjacency, st, ([], [errCd, errDf]))


def iisph_step(*args, **kwargs):
    """Compatibility import; IISPH has its own scheme module."""
    from .iisph import iisph_step as step

    return step(*args, **kwargs)
