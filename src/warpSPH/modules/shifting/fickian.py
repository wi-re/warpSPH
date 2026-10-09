"""Fickian particle shift of the DFSPH2D / omniSPH incompressible loop (`shifting='fixed'` and `'fickian'` of the boundaries repo's `DFSPH2D`): a position move only (no momentum),

    dx_i = - D_i grad C_i ,   grad C_i = sum_j V_j grad W_ij + G_i        (the concentration gradient; the analytic wall completes it with `G = sum_b wm int grad W dA`),

`D = shiftA h_s^2` (`'fixed'`, per step) or `D = shiftA h_s |v_i| dt` (`'fickian'`, Lind et al. 2012), `h_s` half the support. Next to a free surface (the surface particles and every particle with a surface
neighbour, `rho < surfaceRho`) only the part tangential to `grad C` is kept (Lind et al.: no shift across the surface); the magnitude is capped at `shiftCap` spacings. This is the shift of the closed-domain
preset (`CLOSED_PRESET`: compact projection, no density solve, `fixed` shift `shiftA = 0.5`), where the shift keeps the distribution that the density solve would otherwise.
Written for Wendland C2 (the pair gradient is written out); units as `omniIncompressible` (rest density 1, mass = volume).
"""
import math
from typing import Any, Optional

import torch

from warpSPHCore import KernelFunctions

from ..incompressible.compactProjection import _dwendland2, minimumImage

__all__ = ['computeFickianShift']

F64 = torch.float64


def computeFickianShift(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *, fluid: torch.Tensor, rho0: float, dt: float,
                        velocities: Optional[torch.Tensor] = None, wall: Optional[Any] = None) -> torch.Tensor:
    """The shift [N, 2] (state dtype), zero on non-fluid rows. `schemeConfig.shifting` in {'none', 'fixed', 'fickian'} (any other raises), `shiftA`, `shiftCap`, `surfaceRho`, `freeSurface`;
    `velocities` default the state's (`'fickian'` uses the velocity of the end of the step)."""
    mode = getattr(schemeConfig, 'shifting', 'none')
    zeros = torch.zeros_like(state.velocities)
    if mode == 'none':
        return zeros
    if mode not in ('fixed', 'fickian'):
        raise ValueError(f"shifting must be 'none', 'fixed' or 'fickian', got {mode!r}")
    if config.kernel != KernelFunctions.Wendland2:
        raise NotImplementedError('the Fickian shift is written for the Wendland C2 kernel')
    if wall is None and bool((state.kinds != 0).any()):
        raise NotImplementedError('the Fickian shift is written for analytic walls and wall-free domains (no boundary-particle terms)')
    x = state.positions.to(F64)
    h = state.supports.to(F64)
    V = state.masses.to(F64) / float(rho0)
    i, j = adjacency.i.long(), adjacency.j.long()
    keep = (i != j) & fluid[i] & fluid[j]
    i, j = i[keep], j[keep]
    d = minimumImage(x[i] - x[j], config.domain)
    r = d.norm(dim=1).clamp(min=1e-300)
    inside = r < h[i]                                                                         # the Verlet list also carries pairs beyond the support (W = 0): they must not count as neighbours
    i, j, d, r = i[inside], j[inside], d[inside], r[inside]
    gW = (_dwendland2(r, h[i]) / r)[:, None] * d                                              # grad_i W_ij
    gC = torch.zeros_like(x).index_add_(0, i, V[j][:, None] * gW)
    if wall is not None:
        gC = gC + wall.G.sum(0).to(F64)
    hs = 0.5 * h
    A = float(getattr(schemeConfig, 'shiftA', 0.5))
    if mode == 'fickian':
        v = (velocities if velocities is not None else state.velocities).to(F64)
        D = A * hs * v.norm(dim=1) * dt
    else:
        D = A * hs * hs
    dxs = -D[:, None] * gC
    if getattr(schemeConfig, 'freeSurface', True):
        surf = (state.densities.to(F64) < float(getattr(schemeConfig, 'surfaceRho', 0.85))) & fluid
        layer = surf | (torch.zeros(len(x), dtype=F64, device=x.device).index_add_(0, i, surf[j].to(F64)) > 0.5)         # the surface particles and their neighbours
        n = gC / gC.norm(dim=1, keepdim=True).clamp(min=1e-300)
        dxs = torch.where(layer[:, None], dxs - (dxs * n).sum(1, keepdim=True) * n, dxs)
    cap = float(getattr(schemeConfig, 'shiftCap', 0.25)) * float(config.dx)
    nrm = dxs.norm(dim=1)
    dxs = dxs * torch.where(nrm > cap, cap / nrm.clamp(min=1e-300), torch.ones_like(nrm))[:, None]
    return torch.where(fluid[:, None], dxs, torch.zeros_like(dxs)).to(state.velocities.dtype)
