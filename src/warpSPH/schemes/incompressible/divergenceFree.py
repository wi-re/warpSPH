"""Local VD+PS/hybrid incompressible RHS composition.

Unlike the full-map reference/omni/Band compositions, this returns an ordinary
RHS for the selected integrator. The density projection may become a velocity
kick (gravity auto policy) or a post-integrator particle shift/resample.
See docs/incompressible/INVENTORY_2026-10-21.md for source and wall contracts."""

from typing import Any
from warpSPH.configurations.compute import neighborSearchOptions
from warpSPH.configurations import SimulationConfig
from warpSPH.modules.incompressible.omni import solveOmni
from warpSPH.schemes.incompressible.omniIncompressible import (
    pressurePolicy,
    velocityFilter as _xsphFilter,
)


def _solve(*args, **kwargs):
    return solveOmni(*args, policy=pressurePolicy(), **kwargs)


from warpSPH.systems import WeaklyCompressibleSystemUpdate
from warpSPH.modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from warpSPH.modules.deltaSPH import computeVelocityDiffusion
from warpSPH.modules.density import computeDensities
from warpSPH.modules.gravity import computeGravity, computeBodyForce
from warpSPH.modules.incompressible import solveDivergenceFree
from warpSPH.modules.mdbc import computeBoundaryVelocities, computeMdbcNoPenShift
from warpSPH.modules.mdbc._util import stateHasBoundaryParticles
from warpSPH.modules.momentum import computeMomentum
from warpSPH.modules.surfaceDetection import detectFreeSurface
from warpSPHCore import SupportScheme, buildVerletList
import torch
from ...modules.boundaryConditions.pinned import pinnedKeepWeight
from warpSPHCore.profiling import record_function
import numpy as np

__all__ = ["divergenceFree_step"]
SURFACE_SOURCE = "full"
DIVERGENCE_SOLVER = "omni"
INSTEP_CD = "auto"
SOLVE_ORDER = "div_then_cd"
XSPH_SCALE = 0.0
GRAVITY_OSC = None
NOPEN_SHIFT = "config"


def divergenceFree_step(
    system: Any, dt: float, config: SimulationConfig, schemeConfig: Any, verbose=False
):
    currentSystem = system
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency
    from warpSPH.modules.incompressible.compactProjection import resolveClosedPreset

    resolveClosedPreset(schemeConfig, config)
    with record_function("[warpSPH] - [deltaSPH - 01] - compute adjacency"):
        verletScale = config.verletScale
        adjacency = buildVerletList(
            currentState,
            config.domain,
            verletScale=verletScale,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=currentSystem.adjacency,
            verbose=False,
            **neighborSearchOptions(config),
        )
        currentSystem.adjacency = adjacency
    with record_function("[warpSPH] - [deltaSPH - 02] - compute density"):
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)
        currentState.densities[currentState.kinds == 2] = 1.0
    if getattr(schemeConfig, "boundaryFriction", 0.0):
        from warpSPH.modules.xsph import computeBoundaryFriction

        currentState.velocities = currentState.velocities + computeBoundaryFriction(
            currentState, config, schemeConfig, adjacency
        )
    if verbose:
        print(
            f"step {currentSystem.t:.6g}: density stats: mean={currentState.densities.mean().item():.6g}, min={currentState.densities.min().item():.6g}, max={currentState.densities.max().item():.6g}"
        )
    boundaryPressureMode = schemeConfig.solverConfig.boundaryPressureMode
    with record_function("[warpSPH] - [deltaSPH - 06] - compute boundary velocities"):
        currentVelocities = currentState.velocities.clone()
        currentState.velocities = computeBoundaryVelocities(
            currentState, config, schemeConfig, adjacency
        )
    with record_function("[warpSPH] - [deltaSPH - 04] - enforce BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)
    with record_function("[warpSPH] - [deltaSPH - 08] - compute surface detection"):
        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(
            currentState,
            config,
            schemeConfig,
            schemeConfig.surfaceDetectionConfig,
            adjacency,
            returnNormals=True,
        )
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin
    with record_function("[warpSPH] - [deltaSPH - 11] - compute dvdt_diss"):
        dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency)
    with record_function("[warpSPH] - [deltaSPH - 12] - compute drhodt"):
        drhodt = computeMomentum(currentState, config, schemeConfig, adjacency)
    with record_function("[warpSPH] - [deltaSPH - 14] - compute forcing"):
        forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        dvdt = forcing / currentState.masses.view(-1, 1)
    with record_function("[warpSPH] - [deltaSPH - 15] - compute gravity"):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        if GRAVITY_OSC is not None:
            _amp, _per = GRAVITY_OSC
            gravity = gravity * (_amp * np.sin(2.0 * np.pi * currentSystem.t / _per))
        gravity[currentState.kinds != 0] = 0.0
        dvdt += gravity
        dvdt += computeBodyForce(currentState, schemeConfig)
    with record_function("[warpSPH] - [deltaSPH - 16] - compute mDBC no-pen shift"):
        _noPenOn = (
            schemeConfig.solverConfig.mdbcNoPenetrationShift
            if NOPEN_SHIFT == "config"
            else NOPEN_SHIFT
        )
        if _noPenOn:
            _provider = getattr(schemeConfig, "boundaryProvider", None)
            if _provider is None or stateHasBoundaryParticles(currentState, config):
                nopenshift = computeMdbcNoPenShift(currentState, config, schemeConfig, adjacency)
                nopenshift[currentState.kinds != 0] = 0.0
            else:
                nopenshift = torch.zeros_like(currentState.velocities)
            if _provider is not None:
                from warpSPH.modules.analyticBoundary import analyticNoPenShift

                _analytic = analyticNoPenShift(
                    _provider,
                    currentState,
                    config,
                    schemeConfig,
                    getattr(schemeConfig, "_analyticDx", None) or float(config.dx),
                )
                nopenshift = nopenshift + torch.where(
                    (currentState.kinds != 0).unsqueeze(-1), torch.zeros_like(_analytic), _analytic
                )
            dvdt = dvdt + nopenshift / dt
    currentState.pressures = (
        torch.zeros_like(currentState.densities)
        if currentState.pressures is None
        else currentState.pressures
    )
    fluid = currentState.kinds == 0
    rho0 = schemeConfig.fluid.restDensity
    _zeros = torch.zeros_like(currentState.densities)
    _instepCD = schemeConfig.gravityConfig.active if INSTEP_CD == "auto" else INSTEP_CD

    def _omniPass(mode, priorAccel, warmStart, minIters, maxIters, tol, surfaceSource="full"):
        """One `omniIncompressible._solve` pass: build `vEnter` from
        `dvdt + dvdt_diss + priorAccel`, refresh the Dirichlet wall velocity on
        it, solve, restore the velocity."""
        accel = dvdt + dvdt_diss + priorAccel
        accel[currentState.kinds != 0] = 0.0
        vEnter = currentState.velocities + dt * accel
        vSaved = currentState.velocities.clone()
        currentState.velocities = vEnter
        vEnter = computeBoundaryVelocities(currentState, config, schemeConfig, adjacency)
        currentState.velocities = vSaved
        return _solve(
            currentState,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            vEnter=vEnter,
            warmStart=warmStart,
            dt=dt,
            mode=mode,
            minIters=minIters,
            maxIters=maxIters,
            tol=tol,
            surfaceSource=surfaceSource,
        )

    if DIVERGENCE_SOLVER == "vdps":
        dvdt_pressure, pDiv, errDiv, _ = solveDivergenceFree(
            particles=currentState,
            config=config,
            schemeConfig=schemeConfig,
            adjacency=adjacency,
            dvdt=dvdt + dvdt_diss,
            dt=dt,
        )
        a_p_rho, pRho, nRho, errRho = _omniPass(
            "density", dvdt_pressure, 0.5 * _zeros, 3, 256, 0.001, SURFACE_SOURCE
        )
        dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
    elif DIVERGENCE_SOLVER == "omni":
        if SOLVE_ORDER == "cd_then_div":
            if not getattr(schemeConfig, "densitySolve", True):
                raise NotImplementedError(
                    "densitySolve=False is written for SOLVE_ORDER 'div_then_cd'"
                )
            a_p_rho, pRho, nRho, errRho = _omniPass(
                "density", 0.0, 0.5 * _zeros, 3, 256, 0.001, SURFACE_SOURCE
            )
            dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
            a_p_div, pDiv, nDiv, errDiv = _omniPass("divergence", dvdt_inStep, _zeros, 2, 32, 0.01)
            dvdt_pressure = a_p_div
        else:
            a_p_div, pDiv, nDiv, errDiv = _omniPass("divergence", 0.0, _zeros, 2, 32, 0.01)
            dvdt_pressure = a_p_div
            if getattr(schemeConfig, "densitySolve", True):
                a_p_rho, pRho, nRho, errRho = _omniPass(
                    "density", dvdt_pressure, 0.5 * _zeros, 3, 256, 0.001, SURFACE_SOURCE
                )
            else:
                a_p_rho, pRho, nRho, errRho = (torch.zeros_like(dvdt_pressure), pDiv, 0, 0.0)
            dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
    else:
        raise ValueError(f"Unknown DIVERGENCE_SOLVER: {DIVERGENCE_SOLVER!r}")
    _pRho = pRho if pRho is not None else _zeros
    _pDiv = pDiv if pDiv is not None else _zeros
    currentState.pressures = torch.where(fluid, _pRho, torch.zeros_like(_pRho))
    currentState.soundspeeds = torch.where(fluid, _pDiv, torch.zeros_like(_pDiv))
    xsphScale = getattr(schemeConfig, "xsphFilterScale", XSPH_SCALE)
    dvdt_xsph = torch.zeros_like(dvdt)
    if xsphScale:
        dvdt_xsph = (
            xsphScale * _xsphFilter(currentState, config, adjacency, currentState.kinds == 0) / dt
        )
        dvdt_xsph[currentState.kinds != 0] = 0.0
    if getattr(schemeConfig, "xsphCoefficient", 0.0) or getattr(
        schemeConfig, "xsphBoundaryCoefficient", 0.0
    ):
        from warpSPH.modules.xsph import computeXSPH

        dvdt_xsph = dvdt_xsph + computeXSPH(currentState, config, schemeConfig, adjacency) / dt
    from warpSPH.modules.incompressible.loads import (
        bookProjectionWallLoads as _bookWallLoads,
        wallViscousPerBody as _wallViscousPerBody,
    )
    from warpSPH.modules.analyticBoundary import resolveWall as _resolveWall

    _bookWallLoads(
        currentState,
        config,
        schemeConfig,
        adjacency,
        fluid=fluid,
        rho0=rho0,
        pDiv=_pDiv,
        pRho=_pRho,
        viscousPerBody=(
            _wallViscousPerBody(
                currentState,
                config,
                schemeConfig,
                adjacency,
                _resolveWall(currentState, config, schemeConfig, adjacency),
            )
            if getattr(schemeConfig, "analyticWallLoads", False)
            and (not schemeConfig.diffusionParams.inviscid)
            else None
        ),
        densityApplied=bool(_instepCD) and bool(getattr(schemeConfig, "densitySolve", True)),
    )
    dxShift = torch.zeros_like(currentState.velocities)
    if getattr(schemeConfig, "shifting", "none") != "none":
        from warpSPH.modules.shifting.fickian import computeFickianShift
        from warpSPH.modules.analyticBoundary import resolveWall

        dxShift = computeFickianShift(
            currentState,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            dt=dt,
            velocities=currentState.velocities
            + dt * (dvdt + dvdt_diss + dvdt_pressure + dvdt_inStep),
            wall=resolveWall(currentState, config, schemeConfig, adjacency),
        )
    with record_function("[warpSPH] - [deltaSPH - 16] - build update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt=currentState.velocities.clone() + dxShift / dt,
            dvdt=dvdt + dvdt_diss + dvdt_pressure + dvdt_inStep + dvdt_xsph,
            drhodt=torch.zeros_like(currentState.densities),
            passive=torch.zeros(
                currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool
            ),
        )
    with record_function("[warpSPH] - [deltaSPH - 17] - enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        keepBand = pinnedKeepWeight(currentSystem.state, config, schemeConfig)
        if keepBand is not None and getattr(update, "dvdt", None) is not None:
            update.dvdt = update.dvdt * keepBand
        nonFluidMask = (currentState.kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluidMask, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluidMask, torch.zeros_like(update.dvdt), update.dvdt)
        update.drhodt = torch.where(
            nonFluidMask.squeeze(-1), torch.zeros_like(update.drhodt), update.drhodt
        )
    return (update, adjacency, currentState, ([-1], [0]))
