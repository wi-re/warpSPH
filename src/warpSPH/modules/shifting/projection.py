"""Density-projection particle shift and Cornelis Eq. 17 resampling.

This post-integrator correction is distinct from a pressure velocity kick.
Legacy gravity/case gates are supplied explicitly by the system adapter.
"""

import torch
from warpSPHCore import (
    buildVerletList,
    SupportScheme,
    warpOperation,
    OperationProperties,
    WarpOperation,
    GradientScheme,
    OperationDirection,
)
from warpSPHCore.profiling import record_function
from ...configurations.compute import neighborSearchOptions
from ..density import computeDensities
from ..incompressible import solveIncompressible
from ..boundaryConditions.pinned import pinnedKeepWeight
from .wrapper import solveShifting


def applyProjectionShift(
    self,
    dt,
    config,
    schemeConfig,
    *,
    restore="auto",
    mode="cd",
    positionShift=True,
    velocityResample=True,
    shiftAsVelocity=False,
    surfaceMask=False,
    pressurePolicy=None
):
    _psShift = restore
    if _psShift == "auto":
        _psShift = schemeConfig is not None and (not schemeConfig.gravityConfig.active)
    if (
        _psShift
        and schemeConfig is not None
        and (getattr(schemeConfig, "shifting", "none") != "none")
    ):
        _psShift = False
    _analyticWalls = (
        schemeConfig is not None and getattr(schemeConfig, "boundaryProvider", None) is not None
    )
    if _psShift and config is not None and (schemeConfig is not None):
        with record_function("[warpSPH] - [dfsph] - VD+PS particle shift"):
            self.adjacency = buildVerletList(
                self.state,
                config.domain,
                verletScale=config.verletScale,
                supportMode=SupportScheme.SuperSymmetric,
                priorNeighborhood=self.adjacency,
                verbose=False,
                **neighborSearchOptions(config)
            )
            self.state.densities = computeDensities(
                self.state, config, schemeConfig, self.adjacency
            )
            if mode == "delta":
                dx_ps = solveShifting(
                    systemState=self.state,
                    config=config,
                    schemeConfig=schemeConfig,
                    adjacency=self.adjacency,
                    dt=dt,
                )
                dvdt_incomp = dx_ps / (dt * dt)
            elif _analyticWalls:
                from ..incompressible.omni import solveOmni as _solve

                dvdt_incomp, _p_ps, _e_ps, _ = _solve(
                    self.state,
                    config,
                    schemeConfig,
                    self.adjacency,
                    fluid=self.state.kinds == 0,
                    rho0=schemeConfig.fluid.restDensity,
                    vEnter=self.state.velocities,
                    warmStart=torch.zeros_like(self.state.densities),
                    dt=dt,
                    mode="density",
                    minIters=3,
                    maxIters=256,
                    tol=0.001,
                    policy=pressurePolicy,
                )
            else:
                dvdt_incomp, _p_ps, _e_ps, _ps_ps = solveIncompressible(
                    particles=self.state,
                    config=config,
                    schemeConfig=schemeConfig,
                    adjacency=self.adjacency,
                    dvdt=torch.zeros_like(self.state.velocities),
                    dt=dt,
                    verbose=False,
                )
            gradVel = warpOperation(
                self.state,
                operationProperties=OperationProperties(
                    operation=WarpOperation.Gradient,
                    kernel=config.kernel,
                    supportMode=SupportScheme.Gather,
                    operationMode=OperationDirection.AllToAll,
                    gradientMode=GradientScheme.Difference,
                ),
                queryValues=self.state.velocities,
                domain=config.domain,
                adjacency=self.adjacency,
            )
            if _analyticWalls:
                from ..analyticBoundary import resolveWall

                wall = resolveWall(self.state, config, schemeConfig, self.adjacency)
                if wall is not None:
                    v64 = self.state.velocities.to(torch.float64)
                    for bi in range(wall.G.shape[0]):
                        ub = (
                            torch.zeros_like(v64)
                            if wall.pinned[bi]
                            else wall.kin.velocity[bi].to(torch.float64)
                        )
                        G = wall.G[bi].to(torch.float64)
                        jump = ub - v64
                        if not wall.mirror[bi]:
                            jump = (
                                (jump * G).sum(1, keepdim=True)
                                / (G * G).sum(1, keepdim=True).clamp(min=1e-300)
                                * G
                            )
                        gradVel = gradVel + torch.einsum("ni,nj->nij", jump, G).to(gradVel.dtype)
            dx = dt * dt * dvdt_incomp
            proj_vel = torch.einsum("nij, nj -> ni", gradVel, dx)
            fluidCol = (self.state.kinds == 0).unsqueeze(-1)
            if surfaceMask:
                si = getattr(self.state, "surfaceIndicators", None)
                if si is not None:
                    if surfaceMask == "taper":
                        sl = getattr(self.state, "surfaceLambdas", None)
                        w = (
                            sl.clamp(0.0, 1.0).unsqueeze(-1)
                            if sl is not None
                            else (si < 0.5).to(dx.dtype).unsqueeze(-1)
                        )
                    else:
                        w = si.to(torch.bool).logical_not().to(dx.dtype).unsqueeze(-1)
                    dx = dx * w
                    proj_vel = proj_vel * w
            keep = pinnedKeepWeight(self.state, config, schemeConfig)
            if keep is not None:
                dx = dx * keep
                proj_vel = proj_vel * keep
            if positionShift:
                if shiftAsVelocity:
                    self.state.velocities = torch.where(
                        fluidCol, self.state.velocities + dx / dt, self.state.velocities
                    )
                else:
                    self.state.positions = torch.where(
                        fluidCol, self.state.positions + dx, self.state.positions
                    )
            if velocityResample:
                self.state.velocities = torch.where(
                    fluidCol, self.state.velocities + proj_vel, self.state.velocities
                )
