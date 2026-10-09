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

__all__ = ['WallState', 'evaluateWall', 'resolveWall', 'wallDensity', 'wallDivergence', 'wallAlphaCorrection', 'wallPressureAccelerationOmni', 'wallContinuity', 'wallPressureAcceleration', 'wallViscousAcceleration', 'viscousPrefactor', 'wallLoads']

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
    support: float = 0.0        # the (constant) kernel support the aggregates were evaluated for
    wm: float = 1.0             # the wall mass density factor


def resolveWall(state, config, schemeConfig, adjacency, wall=None):
    """The wall of a module's wall term: `wall` when the caller passes it, else the aggregates of the scheme's boundary provider at these positions (shared through
    the `evaluateWall` cache), else `None` (boundary particles, or no wall). The one convention of every module that takes a `wall` argument: `wall=None` never means
    "skip the wall" on a scene that has analytic bodies."""
    if wall is not None:
        return wall
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    if provider is None:
        return None
    from ..gravity import computeGravity
    return evaluateWall(provider, state, config, schemeConfig, computeGravity(state, config, schemeConfig, adjacency))


def _policies(provider):
    """(pinned zero velocity, no-slip mirror) per analytic body from the BCType of its RigidBody."""
    pinned, mirror = [], []
    for rb in provider.rigidBodies:
        if rb.kind == BCType.extended:
            raise NotImplementedError('BCType.extended has no analytic counterpart (open boundaries are out of scope)')
        pinned.append(rb.kind == BCType.zeros)
        mirror.append(rb.kind != BCType.freeSlip)            # noSlip, constant (the wall particles keep the body velocity: an effective no-slip), zeros
    return pinned, mirror


def _token(v):
    if isinstance(v, torch.Tensor):
        return (id(v), v._version)
    return v


def _wallPinned(provider):
    """per analytic body: the wall velocity is pinned to zero (BCType.zeros)."""
    return _policies(provider)[0]


def evaluateWall(provider, state, config, schemeConfig, gravity):
    """The wall aggregates at the fluid particles of `state`. `gravity`: [N, 2] or [2], uniform. The last evaluation is kept
    while the positions (storage and version) and every body's integrated state are unchanged, so the right-hand side, the
    surface detector and the shifting share one provider call per position set."""
    from warpSPHBoundaries.scene import WallOutput
    x0 = state.positions
    key = (id(provider), x0._version, tuple(x0.shape),
           tuple((_token(rb.centerOfMass), _token(rb.orientation), _token(rb.linearVelocity), _token(rb.angularVelocity)) for rb in provider.rigidBodies))
    hit = getattr(provider, '_wallCache', None)
    if hit is not None and hit[0] == key and hit[1] is x0:          # the cached entry keeps x0 alive: an address cannot be reused by another tensor while it is cached
        return hit[2]
    wall = _evaluateWall(provider, state, config, schemeConfig, gravity)
    provider._wallCache = (key, x0, wall)
    return wall


def _evaluateWall(provider, state, config, schemeConfig, gravity):
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
    if getattr(schemeConfig, 'analyticWallPressure', 'hydrostatic') == 'normal':
        # the wall pressure condition on the NORMAL only (dp/dn = rho (g - a_w) . n, the boundary-particle flavour's ghost extrapolation): no tangential
        # pressure gradient is imposed on the fluid next to the wall, so a fluid that is not in hydrostatic balance (the free fall of a dam break
        # from a uniform density) feels no tangential force
        nb = G / G.norm(dim=2, keepdim=True).clamp(min=1e-300)
        a1 = (a1 * nb).sum(-1, keepdim=True) * nb
    A = wm * agg.evaluate((WallOutput('A', 0, 'a1g1'),), a1=a1)['A']
    near = (lam.sum(0) > 1e-9).to(F64)
    return WallState(agg, kin, lam, G, A, near, pinned, mirror, state.positions.dtype, support, wm)


def _wallVelocity(wall, bi):
    return torch.zeros_like(wall.kin.velocity[bi]) if wall.pinned[bi] else wall.kin.velocity[bi]


def wallDensity(wall):
    """The wall's share of the summation density, `wm * int W dA` summed over the bodies (the boundary particles' `sum_k V_k W_ik`)."""
    return wall.lam.sum(0).to(wall.dtype)


def wallDivergence(wall, field, bodyVelocity=False):
    """The wall's share of the difference-form SPH divergence `sum_j V_j (f_j - f_i) . grad W_ij`: `sum_b (f_b - f_i) . G_b`, `G_b = wm int grad W dA`. `f_b` is the body's velocity at
    the particle with `bodyVelocity` (a velocity field: the wall moves with its body, zero for a body pinned to zero velocity), else 0 (a field the wall rows do not carry, the
    pressure acceleration: a static wall takes no reaction)."""
    f = field.to(F64)
    out = -(f * wall.G.sum(0)).sum(1)
    if bodyVelocity:
        for bi in range(wall.G.shape[0]):
            out = out + (_wallVelocity(wall, bi) * wall.G[bi]).sum(1)
    return out.to(wall.dtype)


def wallAlphaCorrection(wall, gradSum, rho):
    """The wall in the IISPH diagonal `alpha_i = -[ |sum_j V_j grad W_ij|^2 / rho_i + V_i sum_j (V_j^2 / m_j) |grad W_ij|^2 ]` (`modules/incompressible/wp_alpha.py`): the wall adds `G = sum_b G_b`
    to the vector sum only (a static wall takes no reaction, so the second sum has no wall part), `-(2 gradSum . G + |G|^2) / rho_i`, with `gradSum` the fluid's `sum_j V_j grad W_ij`."""
    G = wall.G.sum(0)
    return (-(2.0 * (gradSum.to(F64) * G).sum(1) + (G * G).sum(1)) / rho.to(F64)).to(wall.dtype)


def wallContinuity(wall, rho, v):
    """d rho / dt of the wall (free-slip mirror), summed over the bodies in body order."""
    rho64, v64 = rho.to(F64), v.to(F64)
    out = torch.zeros_like(rho64)
    for bi in range(wall.G.shape[0]):
        gm = wall.G[bi].norm(dim=1)
        nb = wall.G[bi] / gm.clamp(min=1e-300)[:, None]
        out = out + 2.0 * rho64 * ((v64 - _wallVelocity(wall, bi)) * nb).sum(1) * gm
    return out.to(wall.dtype)


def wallPressureAcceleration(wall, P, switch, rho, wallMass=1.0, h=1.0, clamp=True, perBody=False):
    """The pressure force of the wall: a = - sum_b [(p^+ + s p) G_b + A_eff,b] / rho. `switch`: the Antuono switch s (+1 / -1) per particle. `perBody`: the [B, N, 2] terms of the bodies instead of their sum."""
    P64, s64, rho64 = P.to(F64), switch.to(F64), rho.to(F64)
    pp = P64.clamp(min=0)
    A, G = wall.A, wall.G
    if clamp:                                              # p_b >= 0: remove (1 - theta) q G of the hydrostatic offset q (docs/dfsph-validation.md s.7)
        eps = 1e-5 * wallMass / h
        q = (A * G).sum(2) / (G * G).sum(2).clamp(min=eps * eps)
        theta = torch.where(q < 0, (pp[None] / (-q).clamp(min=1e-300)).clamp(0.0, 1.0), torch.ones_like(q))
        A = A - ((1.0 - theta) * q)[:, :, None] * G
    wallTerm = (pp + s64 * P64)[None, :, None] * G + A
    acc = -wallTerm / rho64[None, :, None]
    return acc.to(wall.dtype) if perBody else acc.sum(0).to(wall.dtype)


def wallPressureAccelerationOmni(wall, P, rho, rho0, wallMass=1.0, h=1.0, perBody=False):
    """The pressure force of the wall in omniSPH's symmetric form (the fluid pairs `-sum_j V_j (p_i / rho_i^2 + p_j / rho_j^2) grad W_ij`, the wall a mirror: `p_b = p_i^+`, `rho_b = rho0`):
    `a = - sum_b [ (p^+ / rho_i^2 + p^+ / rho0^2) G_b + A_eff,b ]`, with the hydrostatic offset `A_b = int (a1 . y) grad W dA`, `a1 = rho_i (g - a_w)` (so `A_i = rho_i / rho0` times the
    `WallState`'s, evaluated at `rho0`) clamped so the wall pressure `p_i + q` stays >= 0 as in `wallPressureAcceleration`. Differs from the delta+ form (`-[(p^+ + s p) G + A] / rho_i`) by the
    density factors, which matter on under-dense wall rows. `DFSPH2D._boundary_accel` of the boundaries repo is the oracle (equal to ~1e-6)."""
    P64, rho64 = P.to(F64), rho.to(F64)
    pp = P64.clamp(min=0)
    G = wall.G
    A = wall.A * (rho64 / float(rho0))[None, :, None]
    eps = 1e-5 * wallMass / h
    q = (A * G).sum(2) / (G * G).sum(2).clamp(min=eps * eps)
    theta = torch.where(q < 0, (pp[None] / (-q).clamp(min=1e-300)).clamp(0.0, 1.0), torch.ones_like(q))
    A = A - ((1.0 - theta) * q)[:, :, None] * G
    acc = -((pp / rho64 ** 2 + pp / float(rho0) ** 2)[None, :, None] * G + A)
    return acc.to(wall.dtype) if perBody else acc.sum(0).to(wall.dtype)


def wallViscousAcceleration(wall, rho, v, fac, h, wallMass=1.0, kernel=KernelFunctions.Wendland2, perBody=False):
    """The wall term of the velocity diffusion of stage 11 through the exact wall Laplacian: free slip ('laplacian':
    the normal relative velocity along the wall normal) or the antisymmetric mirror ('noslipMirror'). `fac` is the
    prefactor of the pair term, alpha c_s h / xi for the artificial viscosity and 2 (dim + 2) nu = 8 nu for the
    physical one (the effective kinematic viscosity is fac / 8)."""
    from warpSPHBoundaries.scene.viscosity import lap_factor
    rho64, v64 = rho.to(F64), v.to(F64)
    dl_all = lap_factor(h, FAMILY[kernel]) * wall.agg.out['lap']
    out = torch.zeros_like(v64)
    parts = []
    for bi in range(wall.G.shape[0]):
        gm = wall.G[bi].norm(dim=1)
        nb = wall.G[bi] / gm.clamp(min=1e-300)[:, None]
        vrel = v64 - _wallVelocity(wall, bi)
        dl = dl_all[bi]
        if wall.mirror[bi]:
            part = (-2.0 * (fac / 8.0) * wallMass / rho64 * dl * wall.near)[:, None] * vrel
        else:
            un = (vrel * nb).sum(1)
            part = (-2.0 * (fac / 8.0) * wallMass * un / rho64 * dl * wall.near)[:, None] * nb
        out = out + part
        parts.append(part)
    return torch.stack(parts).to(wall.dtype) if perBody else out.to(wall.dtype)


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


def wallLoads(accP, accV, x, m, centers):
    """The load of the fluid on every analytic body from the per-body particle accelerations of the wall terms, accP / accV [B, N, 2] (pressure, viscous): the reaction is -m a, acting at the fluid particle
    (the reference solver's lever, `DeltaSPH2D._load`; the fluid angular momentum balance, so the torque is conserved between the walls). Returns float64 [2, B, 3] = (term) x (body) x (Fx, Fy, torque z
    about `centers` [B, 2]); 2D force per unit depth. The no-penetration impulse (a velocity correction inside the step) is not booked."""
    x64, m64, c64 = x.to(F64), m.to(F64), centers.to(F64)
    out = []
    for acc in (accP, accV):
        F = -m64[None, :, None] * acc.to(F64)
        r = x64[None] - c64[:, None, :]
        tau = (r[..., 0] * F[..., 1] - r[..., 1] * F[..., 0]).sum(1)
        out.append(torch.cat([F.sum(1), tau[:, None]], 1))
    return torch.stack(out)
