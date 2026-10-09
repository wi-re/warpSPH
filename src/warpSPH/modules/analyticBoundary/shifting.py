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

__all__ = ['wallShiftRaw', 'kernelAtSpacing']

F64 = torch.float64
KERNEL_SCALE = {KernelFunctions.Wendland2: 1.897367, KernelFunctions.Wendland4: 2.171239}          # warpSPHCore sphKernelScale(kernel, 2D)


def kernelAtSpacing(kernel, support, mass, rho0):
    """warpSPH's tensile reference value W0 = W(r = (m / rho0)^(1/2) / kernelScale; support) of `computeDeltaShiftWarp` (2D)."""
    if kernel != KernelFunctions.Wendland2:
        raise NotImplementedError('analytic-wall shifting: Wendland C2 only')
    q = (mass / rho0) ** 0.5 / KERNEL_SCALE[kernel] / float(support)          # host double: warpSPHCore's wendland2_k is a Warp function at the process precision
    return 7.0 / (math.pi * float(support) ** 2) * (1.0 - q) ** 4 * (1.0 + 4.0 * q) if q < 1.0 else 0.0


def wallShiftRaw(wall, state, config, schemeConfig, R):
    """The wall part of the raw shift sum at the particles of `state` (float64 [N, 2]); `R` the tensile coefficient of the fluid sum."""
    from warpSPHBoundaries.scene.tensile import tensile_factor
    rho0 = schemeConfig.fluid.restDensity
    H, wm = wall.support, wall.wm
    w0 = kernelAtSpacing(config.kernel, H, float(state.masses.mean()), rho0)
    tens = wm * R / w0 ** 4 * tensile_factor(H, FAMILY[config.kernel]) * wall.agg.out['tens'].sum(0) * wall.near[:, None]
    return (rho0 / (4.0 * state.densities.to(F64)))[:, None] * (wall.G.sum(0) + tens)
