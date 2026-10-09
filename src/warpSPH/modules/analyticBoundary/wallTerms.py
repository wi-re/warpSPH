"""The wall terms of the analytic-boundary flavour of the delta+ scheme (formulas of `DeltaSPH2D.rhs`, validated
against the boundary-particle scheme; derivations in warpSPHBoundaries docs/derivation.md and
docs/dfsph-validation.md s.7).

With W the kernel, mu the wall mass density (`analyticWallMass`, 1 = the fluid's own packing), the provider's
aggregates at a fluid particle i, per body b (all arrays [B, N, ...], raw: this module applies the factors):

    lam    = int_solid W dA                    completeness of the wall
    G      = mu grad_i lam                     wall normal times a magnitude (the pair sum of wall particles V_b grad W)
    A(a1)  = mu int_solid (a1 . y) grad_i W    the hydrostatic extrapolation, a1 = rho0 (g - a_wall)
    lap    = int_solid Laplace-weight         the exact wall Laplacian (free-slip / mirror viscosity)

    continuity     d rho / dt   += sum_b 2 rho ((v - u_b) . n_b) |G_b|            (free-slip mirror: the wall continuum moves with u_b)
    pressure       a            += - sum_b [ (p^+ + s p) G_b + A_eff ] / rho        (Antuono switch s; p_b >= 0 clamp on the hydrostatic part)
    viscosity      a            += - 2 (fac / 8) mu / rho * dl * ...                 laplacian (free slip) or noslipMirror (antisymmetric mirror)

with n_b = G_b / |G_b|, fac = alpha c0 H / xi, dl = lap_factor(H) * lap_b.  Gravity is taken uniform (the wall
pressure condition needs a1 at the particle).
"""
from dataclasses import dataclass
from typing import Any, List

import torch

from warpSPHCore import KernelFunctions

from ...boundary.provider import bindBodies
from ...configurations.region import BCType

__all__ = ['WallState', 'evaluateWall', 'wallContinuity', 'wallPressureAcceleration', 'wallViscousAcceleration', 'viscousPrefactor']

F64 = torch.float64
FAMILY = {KernelFunctions.Wendland2: 'w2', KernelFunctions.Wendland4: 'w4'}
XI = {KernelFunctions.Wendland2: 2.8213846683502197, KernelFunctions.Wendland4: 3.56734561920166}      # warpSPHCore sphKernel_xi(kernel, 2D)


@dataclass
class WallState:
    agg: Any                    # the provider's aggregate (FusedWall): .out, .evaluate, .cone_area
    kin: Any                    # BodyKinematics of all bodies at the particle positions
    lam: torch.Tensor           # [B, N]
    G: torch.Tensor             # [B, N, 2]
    A: torch.Tensor             # [B, N, 2]
    near: torch.Tensor          # [N] 1 where any wall is within the support
    pinned: List[bool]          # per body: the wall velocity is pinned to zero (BCType.zeros)
    mirror: List[bool]          # per body: antisymmetric mirror (no-slip) instead of free slip
    dtype: Any


def _policies(provider):
    """(pinned zero velocity, no-slip mirror) per analytic body from the BCType of its RigidBody."""
    pinned, mirror = [], []
    for rb in provider.rigidBodies:
        if rb.kind == BCType.extended:
            raise NotImplementedError('BCType.extended has no analytic counterpart (open boundaries are out of scope)')
        pinned.append(rb.kind == BCType.zeros)
        mirror.append(rb.kind != BCType.freeSlip)            # noSlip, constant (the wall particles keep the body velocity: an effective no-slip), zeros
    return pinned, mirror


def evaluateWall(provider, state, config, schemeConfig, gravity):
    """The wall aggregates at the fluid particles of `state` (one provider call per stage). `gravity`: [N, 2] or [2], uniform."""
    from warpSPHBoundaries.scene import WallOutput
    bindBodies(provider, provider.rigidBodies)
    pinned, mirror = _policies(provider)
    wm = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
    needLap = True                                         # the Laplacian channel serves the free-slip and the mirror form
    support = float(getattr(schemeConfig, '_analyticSupport', None) or state.supports.max())
    agg = provider.aggregate(state, support, config.kernel, laplacian=needLap, fixedAdjacency=True)
    x = state.positions.to(F64)
    kin = provider.scene.kinematics(x)
    g = torch.as_tensor(gravity).to(F64)
    g = g.reshape(-1, 2)[0] if g.dim() > 1 else g
    a1 = schemeConfig.fluid.restDensity * (g[None, None, :] - kin.acceleration)
    lam, G = wm * agg.out['lam'], wm * agg.out['G']
    A = wm * agg.evaluate((WallOutput('A', 0, 'a1g1'),), a1=a1)['A']
    near = (lam.sum(0) > 1e-9).to(F64)
    return WallState(agg, kin, lam, G, A, near, pinned, mirror, state.positions.dtype)


def _wallVelocity(wall, bi):
    return torch.zeros_like(wall.kin.velocity[bi]) if wall.pinned[bi] else wall.kin.velocity[bi]


def wallContinuity(wall, rho, v):
    """d rho / dt of the wall (free-slip mirror), summed over the bodies in body order."""
    rho64, v64 = rho.to(F64), v.to(F64)
    out = torch.zeros_like(rho64)
    for bi in range(wall.G.shape[0]):
        gm = wall.G[bi].norm(dim=1)
        nb = wall.G[bi] / gm.clamp(min=1e-300)[:, None]
        out = out + 2.0 * rho64 * ((v64 - _wallVelocity(wall, bi)) * nb).sum(1) * gm
    return out.to(wall.dtype)


def wallPressureAcceleration(wall, P, switch, rho, wallMass=1.0, h=1.0, clamp=True):
    """The pressure force of the wall: a = - sum_b [(p^+ + s p) G_b + A_eff,b] / rho. `switch`: the Antuono switch s (+1 / -1) per particle."""
    P64, s64, rho64 = P.to(F64), switch.to(F64), rho.to(F64)
    pp = P64.clamp(min=0)
    A, G = wall.A, wall.G
    if clamp:                                              # p_b >= 0: remove (1 - theta) q G of the hydrostatic offset q (docs/dfsph-validation.md s.7)
        eps = 1e-5 * wallMass / h
        q = (A * G).sum(2) / (G * G).sum(2).clamp(min=eps * eps)
        theta = torch.where(q < 0, (pp[None] / (-q).clamp(min=1e-300)).clamp(0.0, 1.0), torch.ones_like(q))
        A = A - ((1.0 - theta) * q)[:, :, None] * G
    wallTerm = (pp + s64 * P64)[None, :, None] * G + A
    return (-wallTerm / rho64[None, :, None]).sum(0).to(wall.dtype)


def wallViscousAcceleration(wall, rho, v, fac, h, wallMass=1.0, kernel=KernelFunctions.Wendland2):
    """The wall term of the velocity diffusion of stage 11 through the exact wall Laplacian: free slip ('laplacian':
    the normal relative velocity along the wall normal) or the antisymmetric mirror ('noslipMirror'). `fac` is the
    prefactor of the pair term, alpha c_s h / xi for the artificial viscosity and 2 (dim + 2) nu = 8 nu for the
    physical one (the effective kinematic viscosity is fac / 8)."""
    from warpSPHBoundaries.scene.viscosity import lap_factor
    rho64, v64 = rho.to(F64), v.to(F64)
    dl_all = lap_factor(h, FAMILY[kernel]) * wall.agg.out['lap']
    out = torch.zeros_like(v64)
    for bi in range(wall.G.shape[0]):
        gm = wall.G[bi].norm(dim=1)
        nb = wall.G[bi] / gm.clamp(min=1e-300)[:, None]
        vrel = v64 - _wallVelocity(wall, bi)
        dl = dl_all[bi]
        if wall.mirror[bi]:
            out = out + (-2.0 * (fac / 8.0) * wallMass / rho64 * dl * wall.near)[:, None] * vrel
        else:
            un = (vrel * nb).sum(1)
            out = out + (-2.0 * (fac / 8.0) * wallMass * un / rho64 * dl * wall.near)[:, None] * nb
    return out.to(wall.dtype)


def viscousPrefactor(schemeConfig, config, h):
    """The prefactor `fac` of the pair viscosity of stage 11 (`modules/deltaSPH/wp_viscosityDelta.py`): alpha c_s h / xi
    (artificial viscosity, `inviscid`) or 2 (dim + 2) nu = 8 nu (physical, 2D). The shear-carrying Morris term has no
    exact wall Laplacian here."""
    from ...enumTypes import ViscosityTerm
    dp = schemeConfig.diffusionParams
    if getattr(dp, 'viscousTerm', ViscosityTerm.monaghanGingold) == ViscosityTerm.morris1997 and not dp.inviscid:
        raise NotImplementedError('the Morris viscosity has no analytic-wall term')
    if dp.inviscid:
        return float(dp.inviscidAlpha) * float(schemeConfig.fluid.fixedSoundSpeed) * float(h) / XI[config.kernel]
    return 8.0 * float(dp.viscidNu)
