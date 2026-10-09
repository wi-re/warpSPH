"""The compact approximate projection of the incompressible schemes: the divergence-free correction `v = v* - dt grad P / rho` with `P` from the compact (Morris / Brookshaw) Laplacian instead of the
composed `div(grad)` of the scheme's own operators (Cummins & Rudman 1999). Port of `warpSPHBoundaries/sim/projection.py` and `DFSPH2D._solve_compact` (`projection='compact'`).

The composed operator, converged, is an exact discrete projection that also removes the divergence noise of moving particles and damps the resolved flow (TGV decay 1.38x analytic at 100 iterations); the
compact one does not (1.00-1.01x converged), and it is symmetric, so a Krylov method applies where the relaxed Jacobi of `omniIncompressible._solve` is limited by its relaxation factor. The solve is

    sum_j w_ij (P_i - P_j) + D_i P_i = Vt_i s_i ,   w_ij = dt^2 2 Vt_i Vt_j (x_ij . grad W_ij) / (r_ij^2 + eta^2 h_i^2) / cal  <= 0   (symmetric)

`Vt = m / (rho0 rho)` the apparent volume, `s` the divergence source `dt div(v*)` of the scheme (`omniIncompressible._divergence`, the wall flux included), `cal = nu_eff / nu` of the Morris operator on the
lattice (`morrisCalibration`), `D_i <= 0` the wall row (a Robin condition: the response of the wall's own divergence term to the wall pressure force on the particle's own pressure; without it the wall is
Neumann and a wall correction that moves the near-wall flux more than the fluid Laplacian predicts overshoots every step). Free-surface rows (`rho < surfaceRho`) are Dirichlet, `P = 0`. Solved by
Jacobi-preconditioned CG to the relative residual `tol`, warm started from the previous solution of the same state.

Units as `omniIncompressible`: rest density 1, mass = volume. Wendland C2 only (the pair kernel is written out here).
"""
import math
from typing import Any, Optional, Tuple

import numpy as np
import torch

from warpSPHCore import KernelFunctions

__all__ = ['MORRIS_ETA2', 'CLOSED_PRESET', 'morrisCalibration', 'compactWeights', 'CompactCG', 'solveCompactProjection', 'resolveClosedPreset']

F64 = torch.float64
MORRIS_ETA2 = 0.0025               # eta^2 / h^2 of the Morris pair weight (`modules/deltaSPH/wp_viscosityDelta.py`)


#: DFSPH2D `CLOSED_PRESET`: the converged-projection setup for closed / periodic flows without a free surface (Stokes arrays, TGV, channel, Couette ~ 1.00 against 1.2-1.45 for the density-solve path).
#: Fields still at their library default take these values (`resolveClosedPreset`); explicit values win.
CLOSED_PRESET = dict(projection='compact', densitySolve=False, shifting='fixed', shiftA=0.5, divergenceGauge='min')
_LIBRARY_DEFAULTS = dict(projection='jacobi', densitySolve=True, shifting='none', shiftA=0.5, divergenceGauge='none')


def resolveClosedPreset(schemeConfig: Any, config: Any) -> None:
    """`schemeConfig.closedPreset` (default False: opt-in here, unlike DFSPH2D's `None` = on iff the domain is fully periodic, so the shipped incompressible cases keep their behaviour): True applies `CLOSED_PRESET`
    to every field still at its library default, once per scheme config; `'auto'` is DFSPH2D's rule (on iff every dimension of the domain is periodic). A walled closed domain needs `True`."""
    flag = getattr(schemeConfig, 'closedPreset', False)
    if getattr(schemeConfig, '_closedPresetResolved', False) or flag is False or flag is None:
        return
    if flag == 'auto':
        per = getattr(config.domain, 'periodic', None)
        flag = per is not None and bool(torch.as_tensor(per).all())
    schemeConfig._closedPresetResolved = True
    if not flag:
        return
    for k, v in CLOSED_PRESET.items():
        if getattr(schemeConfig, k, _LIBRARY_DEFAULTS[k]) == _LIBRARY_DEFAULTS[k]:
            setattr(schemeConfig, k, v)


def _dwendland2(r, h):
    """dW/dr of Wendland C2 in 2D, support h (<= 0)."""
    q = (r / h).clamp(max=1.0)
    return torch.where(r < h, -20.0 * q * (1.0 - q) ** 3 * (7.0 / math.pi) / h ** 3, torch.zeros_like(r))


def morrisCalibration(h: float, dx: float, V: float, eta2: float = MORRIS_ETA2) -> float:
    """`nu_eff / nu` of the Morris operator `a_i = sum_j V_j nu (rho_i + rho_j) / rho_i K_ij (v_i - v_j)`, `K = (x_ij . grad W) / (r^2 + eta2 h^2)`, on a square lattice of spacing `dx` with particle volume `V`
    at rest density 1: for a shear field the long-wave rate is `-sum_j V K_ij y_j^2` (the continuum value with eta -> 0 is 1)."""
    m = int(math.ceil(h / dx)) + 1
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing='ij')
    r = np.hypot(i * dx, j * dx).ravel()
    y = (j * dx).ravel()
    keep = (r > 1e-14) & (r < h)
    r, y = r[keep], y[keep]
    dw = _dwendland2(torch.as_tensor(r), h).numpy()
    return float(-np.sum(V * dw * r / (r * r + eta2 * h * h) * y * y))


def minimumImage(d: torch.Tensor, domain: Any) -> torch.Tensor:
    """The pair separations `d` through the periodic dimensions of `domain` (nearest image), unchanged without periodicity."""
    per = getattr(domain, 'periodic', None)
    if per is None or not bool(torch.as_tensor(per).any()):
        return d
    length = (domain.max - domain.min).to(d.dtype)
    flags = torch.as_tensor(per, device=d.device).to(d.dtype)
    return d - flags * length * torch.round(d / length)


def compactWeights(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor, Vt: torch.Tensor, h: torch.Tensor, cal: float, dt: float, domain: Any = None) -> torch.Tensor:
    """`w_ij` of the compact Laplacian times `dt^2` for the pairs (i, j) (`x_ij . grad W = r dW/dr`), nearest periodic images."""
    d = minimumImage(x[i] - x[j], domain)
    r = d.norm(dim=1)
    hi = h[i]
    return dt * dt * 2.0 * Vt[i] * Vt[j] * _dwendland2(r, hi) * r / (r * r + MORRIS_ETA2 * hi ** 2) / cal


class CompactCG:
    """Jacobi-preconditioned CG for `sum_j w_ij (P_i - P_j) + D_i P_i = rhs_i` over the pair list (i, j, w) (both directions), `dirichlet` rows fixed at 0 (eliminated symmetrically). Keeps the last
    solution as the warm start; the true residual decides convergence."""

    def __init__(self, warmStart: bool = True):
        self.warmStart = warmStart
        self.prevP = None
        self.stats = {'solves': 0, 'iterations': 0}

    def solve(self, i, j, w, rhs, tol=1e-8, maxIterations=2000, diag=None, dirichlet=None) -> Tuple[torch.Tensor, int, float]:
        D = diag.clone() if diag is not None else torch.zeros_like(rhs)
        if dirichlet is not None:
            fixed = dirichlet.to(torch.bool)
            toFixed = fixed[j] & ~fixed[i]
            D = D.index_add(0, i, torch.where(toFixed, w, torch.zeros_like(w)))               # w_ij (P_i - 0): the coupling to a fixed neighbour becomes diagonal
            w = torch.where(fixed[i] | fixed[j], torch.zeros_like(w), w)
            ref = D.abs().mean() + torch.zeros_like(rhs).index_add_(0, i, w.abs()).mean()
            D = torch.where(fixed, -ref.expand_as(D), D)                                        # P_i = 0 (any scale: the row decouples)
            rhs = torch.where(fixed, torch.zeros_like(rhs), rhs)

        def Aop(p):
            return torch.zeros_like(rhs).index_add_(0, i, w * (p[i] - p[j])) + D * p

        self.stats['solves'] += 1
        scale = float(rhs.norm())
        if scale <= 1e-30:
            self.prevP = torch.zeros_like(rhs)
            return self.prevP.clone(), 0, 0.0
        b = -rhs / scale                                                                        # (-A) P = -rhs, normalised (a near-solenoidal field underflows the inner products otherwise)
        x0 = self.prevP / scale if (self.warmStart and self.prevP is not None and self.prevP.shape == rhs.shape) else None
        dg = torch.zeros_like(rhs).index_add_(0, i, w) + D
        Minv = torch.where(dg.abs() > 0, -1.0 / dg, torch.zeros_like(dg))
        p = x0.clone() if x0 is not None else torch.zeros_like(b)
        r = b + Aop(p) if x0 is not None else b.clone()
        z = Minv * r
        q = z.clone()
        rz = (r * z).sum()
        rel = float(r.norm())
        it = 0
        if rel >= tol:
            for it in range(1, maxIterations + 1):
                Aq = -Aop(q)
                alpha = rz / (q * Aq).sum()
                p = p + alpha * q
                r = r - alpha * Aq
                if it % 10 == 0:
                    rel = float(r.norm())
                    if not math.isfinite(rel) or rel < tol:
                        break
                z = Minv * r
                rz2 = (r * z).sum()
                q = z + (rz2 / rz) * q
                rz = rz2
            rel = float(r.norm())
        P = p * scale
        self.prevP = P.clone()
        self.stats['iterations'] += it
        return P, it, rel


def solveCompactProjection(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *, fluid: torch.Tensor, rho0: float, vEnter: torch.Tensor, dt: float,
                           wall: Optional[Any], solver: CompactCG, accelBase: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """The divergence stage of `omniIncompressible` / `divergenceFree` as a compact projection. `vEnter`: the velocity entering the solve (`v + dt accel`). Returns
    `(a_p, P, iterations, relative residual)`: the pressure acceleration (fluid pairs + wall, as `omniIncompressible._pressureAccel`), the pressure and the CG statistics.

    The wall (Robin, `projectionWall='robin'`): `v*` carries the pressure-independent wall part (the hydrostatic extrapolation, the wall force at `P = 0`), the source the wall flux (times
    `schemeConfig.projectionWallFlux`), the operator the wall row `D`, the correction is the scheme's own pressure force."""
    from ...schemes.omniIncompressible import _divergence, _pressureAccel
    from ..analyticBoundary import wallPressureAccelerationOmni, wallDivergence
    if config.kernel != KernelFunctions.Wendland2:
        raise NotImplementedError('the compact projection is written for the Wendland C2 kernel')
    cfg = lambda name, default: getattr(schemeConfig, name, default)
    dev = vEnter.device
    zeros = torch.zeros_like(state.densities)
    rho = state.densities.to(F64)
    Vt = (state.masses.to(F64) / float(rho0)) / rho

    ab0 = None
    vStar = vEnter
    if wall is not None:
        ab0 = wallPressureAccelerationOmni(wall, zeros, state.densities, rho0, wallMass=wall.wm, h=wall.support, clampPressure=False)
        vStar = vEnter + dt * torch.where(fluid.unsqueeze(-1), ab0, torch.zeros_like(ab0))
    fluxFactor = float(cfg('projectionWallFlux', 1.0))
    src = dt * _divergence(state, config, adjacency, vStar, None, bodyVelocity=True).to(F64)
    if wall is not None:
        src = src + dt * fluxFactor * wallDivergence(wall, vStar, bodyVelocity=True).to(F64)
    rhs = torch.where(fluid, Vt * src, torch.zeros_like(src))

    D = None
    if wall is not None:
        G = wall.G.sum(0).to(F64)
        D = torch.where(fluid, -Vt * dt * dt * (1.0 / rho ** 2 + 1.0 / float(rho0) ** 2) * (G * G).sum(1), torch.zeros_like(rho))
    surface = None
    if cfg('freeSurface', True):
        surface = (rho < float(cfg('surfaceRho', 0.85))) & fluid
    if D is None and surface is None:                                   # closed domain (no wall, no free surface): the constant is the null space, the mean of the right-hand side is removed
        rhs = torch.where(fluid, rhs - rhs[fluid].mean(), torch.zeros_like(rhs))
    i, j = adjacency.i.long(), adjacency.j.long()
    keep = (i != j) & fluid[i] & fluid[j]
    i, j = i[keep], j[keep]
    cal = cfg('morrisCalibration', None)
    if cal is None:
        h = float(state.supports.double().median())
        cal = morrisCalibration(h, float(config.dx), float(state.masses.double().median()) / float(rho0))
    w = compactWeights(state.positions.to(F64), i, j, Vt, state.supports.to(F64), float(cal), dt, config.domain)
    P, it, rel = solver.solve(i, j, w, rhs, float(cfg('projectionTol', 1e-8)), int(cfg('projectionMaxIterations', 2000)), diag=D, dirichlet=surface)
    gauge = cfg('divergenceGauge', 'none')
    if surface is None or not bool(surface.any()):
        if gauge == 'min':
            P = P - P[fluid].min()
        elif gauge == 'mean':
            V = state.masses.to(F64)
            P = P - (V * P)[fluid].sum() / V[fluid].sum()
    P = torch.where(fluid, P, torch.zeros_like(P))
    a_p = _pressureAccel(state, config, adjacency, P.to(state.densities.dtype), fluid, wall, rho0, clampPressure=False)
    return a_p, P.to(state.densities.dtype), it, rel
