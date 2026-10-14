"""The analytic-boundary parts of the delta-SPH right-hand side (a boundary provider, no wall particles).

* `evaluateStageWall` -- the kernel integrals over the solid at the fluid particles, once per right-hand-side evaluation; `modules/analyticBoundary` turns
  them into the wall terms of the stages that follow (None without analytic bodies: the particle path is then untouched).
* `wallPressureViscousAcceleration` -- the wall pressure condition with the viscous term (the boundaries repo's `wallPressureViscous`).
* `bookWallLoads` -- the load of the fluid on every analytic body, booked on the `RigidBody` for diagnostics and for the free bodies.
"""

import torch

from ...modules.analyticBoundary import evaluateWall, viscousPrefactor, wallLoads, wallPressureAcceleration, wallViscousAcceleration
from ...modules.deltaSPH import computeVelocityDiffusion
from ...modules.gravity import computeGravity
from ...modules.pressure import antuonoSwitch
from warpSPHCore.profiling import record_function

__all__ = ['evaluateStageWall', 'wallPressureViscousAcceleration', 'bookWallLoads']


def evaluateStageWall(currentState, config, schemeConfig, adjacency):
    """The wall aggregates of this right-hand-side evaluation, or None without a boundary provider."""
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    if provider is None:
        return None
    if getattr(schemeConfig, 'analyticWallPressure', 'hydrostatic') == 'mls':
        raise NotImplementedError("analyticWallPressure='mls' is an omniIncompressible option (the delta+ wall pressure is hydrostatic or normal)")
    return evaluateWall(provider, currentState, config, schemeConfig, computeGravity(currentState, config, schemeConfig, adjacency))


def wallPressureViscousAcceleration(wall, schemeConfig, dvdt_diss, currentState):
    """The extra wall acceleration of ``wallPressureViscous`` (or None when it is off): at a no-slip wall the momentum balance gives grad p = rho (g + f - a_wall) +
    rho nu lap u, so the ghost pressure extends with the hydrostatic gradient (already in A) PLUS the particle's own viscous acceleration (fluid pairs + wall
    term): matters where the pressure is viscous-dominated (flow past a curved wall at low Re). Shape ``[B, N, dim]``."""
    if wall is None or not getattr(schemeConfig, 'wallPressureViscous', False):
        return None
    from warpSPHBoundaries.scene import WallOutput
    a1v = (schemeConfig.fluid.restDensity * dvdt_diss.double())[None].expand(wall.G.shape[0], -1, -1).contiguous()
    Av = wall.wm * wall.agg.evaluate((WallOutput('A', 0, 'a1g1'),), a1=a1v)['A']
    return -Av / currentState.densities.double()[None, :, None]


def bookWallLoads(wall, accVP, currentState, config, schemeConfig, adjacency, provider):
    """Book the load of the fluid on every analytic body (pressure, wall viscous): the reaction to the wall terms. Only when asked for
    (`analyticWallLoads`) or when a body is dynamic (the free bodies need it)."""
    if wall is None or not (getattr(schemeConfig, 'analyticWallLoads', False) or any(rb.dynamic for rb in provider.rigidBodies)):
        return
    with record_function("[warpSPH] - [deltaSPH - 16] analytic wall loads"):
        hWall, wm = wall.support, wall.wm
        switch = antuonoSwitch(currentState.pressures, currentState.surfaceIndicators)
        if getattr(schemeConfig, 'pressureConsistent', False):
            # the load must not depend on the pressure level: only the deviation from the mean pressure of the fluid exerts a net force (`pressureConsistent`: the wall's symmetric term 2 (P - mean P) G + A)
            fl = currentState.kinds == 0
            Pload = currentState.pressures - (currentState.pressures * fl).sum() / fl.sum()                     # the mean over the fluid rows, without a boolean-mask gather (a host sync)
            accP = (-(2.0 * Pload.double()[None, :, None] * wall.G.double() + wall.A.double()) / currentState.densities.double()[None, :, None]).to(wall.dtype)
        else:
            accP = wallPressureAcceleration(wall, currentState.pressures, switch, currentState.densities, wallMass=wm, h=hWall, perBody=True)
        if accVP is not None:
            accP = accP + accVP.to(accP.dtype)
        if getattr(schemeConfig, 'wallViscosityClosure', 'mirror') == 'noslipMoment':
            # the no-slip moment closure's per-body term (its balance needs the fluid-only Morris acceleration of the particles)
            from ...modules.analyticBoundary.wallViscosity import wallNoSlipAcceleration
            viscf = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency, wall=False)
            accV = wallNoSlipAcceleration(currentState, config, schemeConfig, adjacency, wall, viscf, perBody=True)
        else:
            accV = wallViscousAcceleration(wall, currentState.densities, currentState.velocities, viscousPrefactor(schemeConfig, config, hWall), hWall, wallMass=wm, kernel=config.kernel, perBody=True)
        loads = wallLoads(accP, accV, currentState.positions, currentState.masses, torch.stack([rb.centerOfMass.to(torch.float64).reshape(2) for rb in provider.rigidBodies]))
        for bi, rb in enumerate(provider.rigidBodies):
            cur = getattr(rb, 'load', None)
            if cur is not None and cur.shape == loads[:, bi, :].shape and cur.dtype == loads.dtype and cur.device == loads.device:
                cur.copy_(loads[:, bi, :])                       # in place: a CUDA-graph replay updates this tensor, a fresh one would be left at the state of the capture
            else:
                rb.load = loads[:, bi, :].clone()
