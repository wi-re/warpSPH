"""Analytic-body load bookkeeping for projection schemes.

The caller states which pressure impulses were actually applied and where
they were evaluated. This preserves the historical accounting convention;
reference DFSPH's post-drift density-load evaluation remains a known caveat.
"""

from typing import Any
import torch
from ..analyticBoundary import resolveWall


def bookProjectionWallLoads(
    state: Any,
    config: Any,
    schemeConfig: Any,
    adjacency: Any,
    *,
    fluid: Any,
    rho0: float,
    pDiv: Any,
    pRho: Any,
    frictionPerBody: Any = None,
    dt: float = 1.0,
    densityApplied: Any = None,
    divergenceHasWall: Any = None,
    viscousPerBody: Any = None
) -> None:
    """The load of the fluid on every analytic body, booked on its `RigidBody.load` ([2, B, 3] = (pressure, friction) x body x (Fx, Fy, torque about the body centre), `wallLoads`), when
    `schemeConfig.analyticWallLoads` is set: the reaction to the wall accelerations the solves applied (the divergence stage's only where it carries the wall: the compact projection, or
    `analyticWallInDivergence`; the density stage's unless `densitySolve` is off, when its pressure is the divergence one) and to the boundary-friction impulse `dv / dt`. The no-penetration
    impulse and the XSPH filter are not booked (`DFSPH2D`'s `forcePressure` / `forceFriction` are the oracle).
    """
    if not getattr(schemeConfig, "analyticWallLoads", False):
        return
    provider = getattr(schemeConfig, "boundaryProvider", None)
    wall = resolveWall(state, config, schemeConfig, adjacency)
    if provider is None or wall is None:
        return
    from ...modules.analyticBoundary import (
        wallPressureAccelerationOmni,
        wallLoads,
        buildMLSPressureFit,
    )

    acc = torch.zeros(
        (wall.G.shape[0],) + tuple(state.velocities.shape),
        dtype=torch.float64,
        device=state.velocities.device,
    )
    compact = getattr(schemeConfig, "projection", "jacobi") == "compact"
    mls = getattr(schemeConfig, "analyticWallPressure", "hydrostatic") == "mls"
    fit = buildMLSPressureFit(state, config, adjacency, fluid, rho0) if mls else None

    def part(p, clamp=True):
        return wallPressureAccelerationOmni(
            wall,
            p,
            state.densities,
            rho0,
            wallMass=wall.wm,
            h=wall.support,
            perBody=True,
            gradient=None if fit is None else fit.gradient(p),
            clampPressure=clamp,
        ).to(torch.float64)

    if divergenceHasWall is None:
        divergenceHasWall = compact or getattr(schemeConfig, "analyticWallInDivergence", False)
    if divergenceHasWall:
        acc = acc + part(pDiv, clamp=not compact)
    if densityApplied is None:
        densityApplied = getattr(schemeConfig, "densitySolve", True)
    if densityApplied:
        acc = acc + part(pRho)
    centers = torch.stack(
        [rb.centerOfMass.to(torch.float64).reshape(2) for rb in provider.rigidBodies]
    )
    fric = (
        torch.zeros_like(acc) if frictionPerBody is None else frictionPerBody.to(torch.float64) / dt
    )
    if viscousPerBody is not None:
        fric = fric + viscousPerBody.to(
            torch.float64
        )  # the wall's viscous acceleration (no-slip closure or exact wall Laplacian), booked with the friction
    loads = wallLoads(acc, fric, state.positions, state.masses, centers)
    for bi, rb in enumerate(provider.rigidBodies):
        rb.load = loads[:, bi, :].clone()


def wallViscousPerBody(state: Any, config: Any, schemeConfig: Any, adjacency: Any, wall: Any):
    """The wall's viscous acceleration per body [B, N, 2] of the physical viscosity (`schemeConfig.omniViscosity`): the no-slip moment closure (needs the fluid-only Morris acceleration) or the
    exact wall Laplacian of the delta+ forms; None without a wall."""
    if wall is None:
        return None
    from ...modules.deltaSPH import computeVelocityDiffusion

    if getattr(schemeConfig, "wallViscosityClosure", "mirror") == "noslipMoment":
        from ...modules.analyticBoundary.wallViscosity import wallNoSlipAcceleration

        viscf = computeVelocityDiffusion(state, config, schemeConfig, adjacency, wall=False)
        return wallNoSlipAcceleration(
            state, config, schemeConfig, adjacency, wall, viscf, perBody=True
        )
    from ..analyticBoundary.viscosity import computeWallViscosity
    return computeWallViscosity(state, config, schemeConfig, wall, perBody=True)
