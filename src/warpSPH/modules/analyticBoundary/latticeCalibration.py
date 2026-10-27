"""The rest lattice at an analytic wall: particle mass, wall mass and wall distance that make a regular lattice an exact rest state of the SPH density sum.

Port of `lattice_calibration` of the boundaries repo (`warpSPHBoundaries/sim/dfsph2d.py`, the "calibrated lattice" of its omniSPH comparison), extended to Wendland C4. A lattice of spacing `dx, dy`
and support `h`, against a flat wall whose continuum has the mass density `mu` of the wall integral `mu * int_solid W dA`:

    S   = sum over the infinite lattice of W * dx * dy        the discrete kernel sum, 1 + O(1e-2) at h / dx ~ 2.5
    V'  = dx dy / S                                          particle mass (rho0 = 1) that makes the bulk density exactly 1
    mu  = 1 / S                                              the wall continuum is "a wall filled with the same lattice"
    d_x, d_y                                                 the distance of the first lattice row from the wall plane for which that row also has density exactly 1

omniSPH uses V = pi r^2 (0.9715 of the lattice cell) and a wall one spacing out: a rest lattice of ~0.98 where the `p >= 0` density solve is inactive. With these numbers the rest lattice
measures 1 to ~1e-3 and the wall rows too (OPEN_PROBLEMS 31 of warpSPH: with the plain `rho0 dx^2` the density solve is active and its relaxed Jacobi iteration is unstable at free surfaces).

`wallIntegral(d, h)` is the half-plane integral of W beyond the distance d from the particle, `int_d^h W(r) 2 arccos(d / r) r dr` (equal to the boundaries repo's `Tier3.lam`).
"""
import math

import numpy as np

from warpSPHCore import KernelFunctions

__all__ = ['latticeCalibration', 'wallIntegral', 'wendland2', 'wendland4']


def wendland2(r, h):
    """Wendland C2, 2D, support h."""
    q = np.minimum(np.asarray(r, dtype=np.float64) / h, 1.0)
    return np.where(np.asarray(r) < h, 7.0 / (math.pi * h * h) * (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)


def wendland4(r, h):
    """Wendland C4, 2D, with the same compact support as the runtime kernel."""
    q = np.minimum(np.asarray(r, dtype=np.float64) / h, 1.0)
    return np.where(np.asarray(r) < h, 9.0/(math.pi*h*h)*(1-q)**6*(1+6*q+35/3*q*q), 0.0)


def _kernelFunction(kernel):
    if kernel == KernelFunctions.Wendland2:
        return wendland2
    if kernel == KernelFunctions.Wendland4:
        return wendland4
    raise NotImplementedError('the calibrated lattice supports Wendland C2 and C4')


_GL = np.polynomial.legendre.leggauss(96)


def wallIntegral(d, h, kernel=KernelFunctions.Wendland2):
    """`int_{half plane beyond distance d} W dA` for the selected Wendland kernel with support h (the integral of the kernel over a flat solid whose surface is at distance d from the particle)."""
    if d >= h:
        return 0.0
    s = 0.5 * (_GL[0] + 1.0)                          # r = d + (h - d) s^2: the arccos(d / r) square-root singularity at r = d is smooth in s
    w = 0.5 * _GL[1]
    r = d + (h - d) * s ** 2
    drds = 2.0 * (h - d) * s
    return float(np.sum(w * _kernelFunction(kernel)(r, h) * 2.0 * np.arccos(np.clip(d / r, -1.0, 1.0)) * r * drds))


def latticeCalibration(dx, dy, h, kernel=KernelFunctions.Wendland2, *, calibrateNormalization=False):
    """Return lattice mass and wall geometry for the chosen density kernel.

    The wall geometry preserves the oracle's first-row fit. Normalized
    density keeps physical cell mass, avoiding a second application of the
    lattice normalization. Wall density continues to use continuum integrals.
    """
    kernelFunction = _kernelFunction(kernel)
    N = int(math.ceil(h / min(dx, dy))) + 2
    n, m = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing='ij')
    S = float((kernelFunction(np.hypot(n * dx, m * dy), h) * dx * dy).sum())
    Vp, mu = dx * dy / S, 1.0 / S

    def rowDensity(d, spacingNormal, spacingTangent):
        k = np.arange(0, N + 2)
        nn = np.arange(-N, N + 1)
        X, Y = np.meshgrid(nn * spacingTangent, k * spacingNormal, indexing='ij')
        return Vp * kernelFunction(np.hypot(X, Y), h).sum() + mu * wallIntegral(d, h, kernel)

    def solve(sn, st):                                # first-row distance d with density exactly 1: bisection (the density falls as the row moves away from the wall)
        a, b = 0.2 * sn, 1.5 * sn
        fa = rowDensity(a, sn, st) - 1.0
        for _ in range(60):
            c = 0.5 * (a + b)
            fc = rowDensity(c, sn, st) - 1.0
            if fa * fc <= 0:
                b = c
            else:
                a, fa = c, fc
        return 0.5 * (a + b)

    # With a normalized summation kernel, physical cell mass already gives
    # Vp*W in the density. Keep the analytic continuum scaled by mu; scaling
    # mass as well would apply the fluid correction twice.
    return dict(S=S, V=dx * dy if calibrateNormalization else Vp,
                mu=mu, dwallX=solve(dx, dy), dwallY=solve(dy, dx))
