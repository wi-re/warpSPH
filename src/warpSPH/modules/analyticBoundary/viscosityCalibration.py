"""Wendland-C2 Morris calibration and periodic pair distances (no pressure solver)."""
import math
from typing import Any
import numpy as np
import torch
MORRIS_ETA2 = 0.0025

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


_ANY_PERIODIC: dict = {}


def _anyPeriodic(domain: Any) -> bool:
    """Whether any dimension of `domain` is periodic: a host bool, read from the device once per value of the flags (a sync per call otherwise: 4 per step). `DomainDescription` is slotted: the cache lives here, keyed on the flags tensor."""
    per = domain.periodic
    hit = _ANY_PERIODIC.get(id(per))
    version = getattr(per, '_version', 0)
    if hit is None or hit[0] is not per or hit[1] != version:
        hit = (per, version, bool(torch.as_tensor(per).any()))
        _ANY_PERIODIC[id(per)] = hit
    return hit[2]


def minimumImage(d: torch.Tensor, domain: Any) -> torch.Tensor:
    """The pair separations `d` through the periodic dimensions of `domain` (nearest image), unchanged without periodicity."""
    per = getattr(domain, 'periodic', None)
    if per is None or not _anyPeriodic(domain):
        return d
    length = (domain.max - domain.min).to(d.dtype)
    flags = torch.as_tensor(per, device=d.device).to(d.dtype)
    return d - flags * length * torch.round(d / length)
