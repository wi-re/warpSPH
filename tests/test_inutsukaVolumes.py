"""GODUNOV_SPH_PLAN layer 3: the algebra of Inutsuka (2002) §3.1 (`modules/godunov/wp_inutsuka.py`).

`inutsukaVolumes` returns `V_ij^2(h) = E[V(t)^2]` and `s*(h) = E[t V(t)^2] / E[V(t)^2]` for the specific volume interpolated along the pair axis,
`t ~ N(0, h^2/4)` (the normalised product of two Gaussians of width h centred at +-s/2). Checked against numerical quadrature of the cubic Hermite
interpolant and of the linear one, plus its defining properties: the Hermite data are reproduced; a uniform volume gives `V^2 = V_0^2`, `s* = 0`; the
linear fallback when the end derivatives disagree in sign; the closed forms of Eqs. (52), (57), (64), (65).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from warpSPHCore import scalar_t

from warpSPH.modules.godunov.wp_inutsuka import inutsukaVolumes

FLOAT64 = scalar_t is wp.float64
NP = np.float64 if FLOAT64 else np.float32
TOL = 1e-10 if FLOAT64 else 5e-5


@wp.kernel
def _volKernel(cubic: wp.bool, a: wp.array2d(dtype=scalar_t), out: wp.array2d(dtype=scalar_t)):
    k = wp.tid()
    v2, sStar = inutsukaVolumes(a[k, 0], a[k, 1], a[k, 2], a[k, 3], a[k, 4], a[k, 5], cubic)
    out[k, 0] = v2
    out[k, 1] = sStar


def volumes(rows, cubic=True):
    a = wp.array(np.asarray(rows, dtype=NP), dtype=scalar_t)
    out = wp.zeros((a.shape[0], 2), dtype=scalar_t)
    wp.launch(_volKernel, dim=a.shape[0], inputs=[cubic, a], outputs=[out])
    return out.numpy().astype(np.float64)


def hermite(Vi, Vj, dVi, dVj, s):
    """The cubic of Inutsuka's Eq. (60)-(61) as a function of t in [-s/2, s/2]: V(s/2) = V_i, V(-s/2) = V_j, V'(s/2) = V'_i, V'(-s/2) = V'_j."""
    dV = Vi - Vj
    A = -2 * dV / s ** 3 + (dVi + dVj) / s ** 2
    B = 0.5 * (dVi - dVj) / s
    C = 1.5 * dV / s - 0.25 * (dVi + dVj)
    D = 0.5 * (Vi + Vj) - 0.125 * (dVi - dVj) * s
    return lambda t: A * t ** 3 + B * t ** 2 + C * t + D


def quadrature(V, h, n=400001):
    sigma = h / 2
    t = np.linspace(-9 * sigma, 9 * sigma, n)
    w = np.exp(-t ** 2 / (2 * sigma ** 2))
    w /= w.sum()
    v2 = (V(t) ** 2 * w).sum()
    return v2, (t * V(t) ** 2 * w).sum() / v2


CASES = [  # V_i, V_j, V'_i, V'_j, s, h
    (1.0, 1.4, 0.8, 1.1, 0.12, 0.09),
    (0.7, 0.5, -0.4, -0.9, 0.05, 0.07),
    (1.0, 1.0, 0.3, 0.3, 0.1, 0.1),
    (2.0, 0.5, 5.0, 3.0, 0.2, 0.1),
]


def test_hermiteDataAreReproduced():
    for Vi, Vj, dVi, dVj, s, h in CASES:
        V = hermite(Vi, Vj, dVi, dVj, s)
        dt = 1e-6
        assert V(s / 2) == pytest.approx(Vi, abs=1e-12) and V(-s / 2) == pytest.approx(Vj, abs=1e-12)
        assert (V(s / 2 + dt) - V(s / 2 - dt)) / (2 * dt) == pytest.approx(dVi, rel=1e-6)
        assert (V(-s / 2 + dt) - V(-s / 2 - dt)) / (2 * dt) == pytest.approx(dVj, rel=1e-6)


def test_cubicMomentsMatchQuadrature():
    got = volumes(CASES, cubic=True)
    for k, (Vi, Vj, dVi, dVj, s, h) in enumerate(CASES):
        v2, sStar = quadrature(hermite(Vi, Vj, dVi, dVj, s), h)
        assert got[k, 0] == pytest.approx(v2, rel=TOL * 10)
        assert got[k, 1] == pytest.approx(sStar, abs=TOL * 10 * h)


def test_linearMomentsMatchQuadrature():
    got = volumes(CASES, cubic=False)
    for k, (Vi, Vj, dVi, dVj, s, h) in enumerate(CASES):
        V = lambda t, Vi=Vi, Vj=Vj, s=s: (Vi - Vj) / s * t + 0.5 * (Vi + Vj)
        v2, sStar = quadrature(V, h)
        assert got[k, 0] == pytest.approx(v2, rel=TOL * 10)
        assert got[k, 1] == pytest.approx(sStar, abs=TOL * 10 * h)
        C, D = (Vi - Vj) / s, 0.5 * (Vi + Vj)
        assert got[k, 0] == pytest.approx(0.25 * h * h * C * C + D * D, rel=TOL * 10)         # Eq. (52)
        assert got[k, 1] == pytest.approx(h * h * C * D / (2 * got[k, 0]), rel=TOL * 10, abs=TOL)   # Eq. (57)


def test_uniformVolumeAndTheSignFallback():
    got = volumes([(0.8, 0.8, 0.0, 0.0, 0.1, 0.1)], cubic=True)
    assert got[0, 0] == pytest.approx(0.64, rel=TOL) and got[0, 1] == pytest.approx(0.0, abs=TOL)
    # end derivatives of opposite sign: the cubic would over- and undershoot, the linear interpolant is used (Inutsuka 2002 §3.1.2)
    row = (1.0, 1.4, 0.8, -1.1, 0.12, 0.09)
    np.testing.assert_allclose(volumes([row], cubic=True), volumes([row], cubic=False), rtol=TOL)
