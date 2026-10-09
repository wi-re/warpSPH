"""The MLS wall pressure of the analytic walls: the pressure field the wall continuum carries is the linear extrapolation of the fluid's own pressure, `p_b(x') = p_i^+ + a1_i . (x' - x_i)`,
`a1_i` the weighted least-squares gradient of the neighbours' pressure (anchored at `p_i`), instead of the hydrostatic `a1 = rho (g - a_wall)`.

Port of `wallPressure='linear'` of the boundaries repo's `DFSPH2D` (omniSPH's MLS wall pressure, evaluated per Jacobi iterate from the current pressure iterate, so the near-wall rows see the pressure
they are solving for). With the weights `w_ij = V_j W_ij` and `y_ij = x_j - x_i`:

    M_i  = sum_j w_ij y_ij (x) y_ij                       (a pseudo-inverse: eigenvalues below 1e-3 of the largest are dropped)
    a1_i = M_i^+ sum_j w_ij y_ij (p_j - p_i)

and the wall's hydrostatic-type term `A_b = wm int (a1 . y) grad W dA = wm a1_d C_dj` with the provider's first-moment tensor `C_dj = int y_d d_j W` (`wallPressureAccelerationOmni(gradient=a1)`).
"""
from dataclasses import dataclass
from typing import Any

import torch

from warpSPHCore import KernelFunctions

__all__ = ['MLSPressureFit', 'buildMLSPressureFit']

F64 = torch.float64
EIGENVALUE_RATIO = 1e-3          # DFSPH2D: the pseudo-inverse of the moment matrix keeps eigenvalues above this fraction of the largest


@dataclass
class MLSPressureFit:
    """The pressure-independent part of the fit (pairs, weights, `M^+`) of one set of positions; `gradient(p)` is then one scatter per Jacobi iterate."""
    i: torch.Tensor              # [P] fluid-fluid pairs (distinct, both directions)
    j: torch.Tensor
    wy: torch.Tensor             # [P, 2] w_ij y_ij
    Minv: torch.Tensor           # [N, 2, 2]

    def gradient(self, p: torch.Tensor) -> torch.Tensor:
        """`a1_i`, [N, 2], of the pressure `p` (clamped at 0 as the wall pressure is: `p_b >= 0` starts from `p_i^+`)."""
        pp = p.to(F64).clamp(min=0)
        b = torch.zeros((len(pp), 2), dtype=F64, device=pp.device).index_add_(0, self.i, self.wy * (pp[self.j] - pp[self.i])[:, None])
        return torch.einsum('nij,nj->ni', self.Minv, b)


def _wendland2(r, h):
    q = (r / h).clamp(max=1.0)
    return torch.where(r < h, 7.0 / (torch.pi * h * h) * (1.0 - q) ** 4 * (1.0 + 4.0 * q), torch.zeros_like(r))


def buildMLSPressureFit(state: Any, config: Any, adjacency: Any, fluid: torch.Tensor, rho0: float) -> MLSPressureFit:
    """The fit geometry at the positions of `state` over the fluid-fluid neighbours of `adjacency`; weights `V_j W_ij` with the particle volume `m_j / rho0` (DFSPH2D's constant `V`). Wendland C2 only (the wall integrals
    of the analytic walls are for the Wendland family; the pair kernel is written out here rather than launched through Warp, once per solve)."""
    if config.kernel != KernelFunctions.Wendland2:
        raise NotImplementedError('the MLS wall pressure is written for the Wendland C2 kernel')
    i, j = adjacency.i.long(), adjacency.j.long()
    keep = (i != j) & fluid[i] & fluid[j]
    i, j = i[keep], j[keep]
    x = state.positions.to(F64)
    h = state.supports.to(F64)
    V = (state.masses / float(rho0)).to(F64)
    y = x[j] - x[i]
    w = V[j] * _wendland2(y.norm(dim=1), h[i])
    wy = w[:, None] * y
    M = torch.zeros((len(x), 2, 2), dtype=F64, device=x.device).index_add_(0, i, wy[:, :, None] * y[:, None, :])
    ev, U = torch.linalg.eigh(M)
    inv = torch.where(ev > EIGENVALUE_RATIO * ev.amax(1, keepdim=True).clamp(min=1e-300), 1.0 / ev.clamp(min=1e-300), torch.zeros_like(ev))
    Minv = U @ torch.diag_embed(inv) @ U.transpose(1, 2)
    return MLSPressureFit(i, j, wy, Minv)
