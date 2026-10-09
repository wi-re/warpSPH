"""The wall's share of the delta+ shift sum with analytic walls.

warpSPH's `computeDeltaShiftWarp` returns the raw fluid sum

    S_i = sum_j  m_j / (2 (rho_i + rho_j)) [1 + R (W_ij / W0)^n] grad_i W_ij ,       W0 = W(r = dx / kernelScale)  (n = 4)

A wall particle b adds the same term (rho_b ~ rho_i, m_b = mu dx^2 rho0 / ... the continuum of wall particles of the
packing the fluid has); summed over the continuum it is

    S_i^wall = rho0 / (4 rho_i) * [ mu int grad_i W dA  +  mu R / W0^n  int W^n grad_i W dA ] = rho0 / (4 rho_i) * [ G_i + mu R / W0^4 tensileFactor(H) tens_i ]

with G = mu grad lam (exact) and `tens` the exact edge reduction of int W^4 grad W (the `wp5` kernel group of the
provider, times `tensile_factor`). The wall is only added within the support of a wall (`near`). W0 is warpSPH's own
(the kernel at dx / kernelScale, not at dx: a 3.7 % difference of the tensile term for Wendland C2 at h = 4 dx), so
the fluid and the wall part of the sum are consistent.
"""
import math

import torch

from warpSPHCore import KernelFunctions

from .wallTerms import FAMILY, evaluateWall

__all__ = ['wallShiftRaw', 'kernelAtSpacing', 'wallUChar', 'wallConcentrationGradient']

F64 = torch.float64
KERNEL_SCALE = {KernelFunctions.Wendland2: 1.897367, KernelFunctions.Wendland4: 2.171239}          # warpSPHCore sphKernelScale(kernel, 2D)


def kernelAtSpacing(kernel, support, mass, rho0):
    """warpSPH's tensile reference value W0 = W(r = (m / rho0)^(1/2) / kernelScale; support) of `computeDeltaShiftWarp` (2D)."""
    if kernel != KernelFunctions.Wendland2:
        raise NotImplementedError('analytic-wall shifting: Wendland C2 only')
    q = (mass / rho0) ** 0.5 / KERNEL_SCALE[kernel] / float(support)          # host double: warpSPHCore's wendland2_k is a Warp function at the process precision
    return 7.0 / (math.pi * float(support) ** 2) * (1.0 - q) ** 4 * (1.0 + 4.0 * q) if q < 1.0 else 0.0


def wallShiftRaw(wall, state, config, schemeConfig, R, volumeWeighted=False):
    """The wall part of the raw shift sum at the particles of `state` (float64 [N, 2]); `R` the tensile coefficient of the fluid sum.
    `volumeWeighted`: the weight of a wall particle is its apparent volume m_b / rho_b = mu dx^2 rho0 / rho_i (Michel 2022 Eq. 2-3, `computeDeltaShiftWarp(volumeWeighted=True)`)
    instead of the mean-density weight m_b / (2 (rho_i + rho_b)) = mu dx^2 rho0 / (4 rho_i) of Sun's law: a factor 4."""
    from warpSPHBoundaries.scene.tensile import tensile_factor
    rho0 = schemeConfig.fluid.restDensity
    H, wm = wall.support, wall.wm
    mass = getattr(schemeConfig, '_analyticMass', None) or float(state.masses.mean())          # the host constant fixed at initialisation: no device-to-host read inside a graph capture
    w0 = kernelAtSpacing(config.kernel, H, mass, rho0)
    tens = wm * R / w0 ** 4 * tensile_factor(H, FAMILY[config.kernel]) * wall.agg.out['tens'].sum(0) * wall.near[:, None]
    pref = (1.0 if volumeWeighted else 0.25) * rho0 / state.densities.to(F64)
    return pref[:, None] * (wall.G.sum(0) + tens)


def _ghostVelocity(pinned, noSlip, v, normal, bodyVelocity):
    """The velocity u_g of the wall continuum seen by a fluid particle of velocity `v` (unit wall `normal`, the body's own velocity `bodyVelocity`), the BC policy of the body as a closure of the
    fluid velocity (what `modules/mdbc/velocity.py` writes into the ghost particles): pinned (BCType.zeros) 0; no slip u_body - w_t; free slip u_body + w_t - w_n, with w = v - u_body."""
    if pinned:
        return torch.zeros_like(v + bodyVelocity)
    w = v - bodyVelocity
    wn = (w * normal).sum(-1, keepdim=True) * normal
    wt = w - wn
    return bodyVelocity - wt if noSlip else bodyVelocity + wt - wn


def wallUChar(wall, state):
    """The wall part of Michel's characteristic velocity U_char,i = max_j |(u_j - u_i) . x_hat_ij| (Eq. 20; warpSPH takes the wall particles into the maximum, `modules/shifting/michel.py`):
    the wall continuum of body b moves with the ghost velocity u_g (the body's boundary condition as a closure of the fluid velocity, `_ghostVelocity`: no slip, free slip, pinned),
    so the maximum over its points is |u_g - u_i| times the largest |cos| between u_g - u_i and a direction to a wall point within the support (`FusedWall.dir_extreme`: 1 where the line of the
    relative velocity meets the wall within the support). float64 [N]; 0 away from the walls."""
    v = state.velocities.to(F64)
    B = wall.G.shape[0]
    rel = []
    for bi in range(B):
        G = wall.G[bi]
        nb = G / G.norm(dim=1, keepdim=True).clamp(min=1e-300)
        rel.append(_ghostVelocity(wall.pinned[bi], wall.mirror[bi], v, nb, wall.kin.velocity[bi]) - v)
    rel = torch.stack(rel)
    return (rel.norm(dim=2) * wall.agg.dir_extreme(rel)).amax(0) * wall.near


def wallConcentrationGradient(wall, state, schemeConfig):
    """The wall's share of grad C_i = sum_j omega_j grad_i W_ij of the implicit shifting (float64 [N, 2]): the wall particles carry omega_b = m_b / rho_b (summation density: rho_b ~ rho_i, so
    rho0 mu dx^2 / rho_i per wall particle) or m_b / rho0 (mu dx^2), the continuum of which is  (rho0 / rho_i or 1) * G  with G = mu grad lam. The wall particles are fixed, so they add nothing to the
    solver's matrix except through `exactHessian`'s diagonal (the wall Hessian integral, not provided)."""
    if schemeConfig.shiftProperties.summationDensity:
        return (schemeConfig.fluid.restDensity / state.densities.to(F64))[:, None] * wall.G.sum(0)
    return wall.G.sum(0)
