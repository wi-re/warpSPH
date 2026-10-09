"""The mDBC no-penetration correction ('impulse' placement) for analytic walls.

A fluid particle at signed distance d < dx/4 from a wall that is closing on it (v_rel . n < 0, n into the fluid) gets
    v += - f vn n ,   f = 3 - 4 clip(1/2 + d / dx, 1/4, 1) ,
f = 1 at the face (inelastic), 2 for a particle dx/4 inside (reflection): the boundary-particle geometry of a flat wall
dx/2 inside the solid, as `computeMdbcNoPenShift` evaluates it on the boundary particles and their ghost nodes. v_rel = v - u_w
with u_w the velocity of the nearest wall at the contact point x - d n (a rigid body's field; 0 for BCType.zeros), so the law is
Galilean: a wall moving with the fluid does not act, the correction brings the RELATIVE normal velocity to (1 - f) vn.
"""
import torch

from ...boundary.provider import bindBodies
from ..gravity import computeGravity
from .wallTerms import _wallPinned, evaluateWall

__all__ = ['analyticNoPenShift']

F64 = torch.float64


def analyticNoPenShift(provider, state, config, schemeConfig, dx):
    """The velocity correction [N, 2] (state dtype), zero where the law does not act."""
    wall = evaluateWall(provider, state, config, schemeConfig, computeGravity(state, config, schemeConfig, None))
    x, v = state.positions.to(F64), state.velocities.to(F64)
    d, n, hit, bidx = provider.scene.signed_distance(x, supportMax=wall.support, want_body=True)
    cp = x - d[:, None] * n
    pinned = _wallPinned(provider)
    W = torch.stack([torch.zeros_like(cp) if pinned[bi] else b.velocityAt(cp) for bi, b in enumerate(provider.scene.bodies)])      # [B, N, 2]
    uw = W[bidx.clamp(min=0), torch.arange(len(x), device=x.device)]
    uw = torch.where((bidx >= 0)[:, None], uw, torch.zeros_like(uw))
    vn = ((v - uw) * n).sum(1)
    f = 3.0 - 4.0 * (0.5 + d / dx).clamp(0.25, 1.0)
    act = (wall.lam.sum(0) > 1e-9) & hit & (d < 0.25 * dx) & (vn < 0)
    return torch.where(act[:, None], (-f * vn)[:, None] * n, torch.zeros_like(n)).to(state.velocities.dtype)
