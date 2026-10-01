"""Gresho-Chan rotating vortex initial state, used by `warpSPH.cases.greshoVortex`.

Samples a regular lattice and imposes the piecewise-analytic radial pressure
and angular-velocity profile of the vortex (three annuli, balanced so the
continuum solution is exactly steady), then derives internal energy, pressure,
and sound speed via `idealGasEOS` from the resulting density field.

`cuspWidth > 0` replaces the standard profile's two slope discontinuities
(r = 0.2 and r = 0.4, where the paper-standard v_phi is only C^0) by a smoothed
profile -- see `greshoProfile`. The standard (`cuspWidth = 0`) path is
unchanged.
"""

from ....sample import *
import numpy as np
from scipy.signal import fftconvolve
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *
from ....modules.timestep.compressible import computeTimestep
import math

__all__ = ['sampleGreshoVortex', 'greshoProfile']


_P_OUTER = 3 + 4 * math.log(2)   # the standard profile's pressure for r >= 0.4
_PROFILE_DR = 1e-5


def _cubicBSpline(x):
    """Unit-support cubic B-spline (C^2), x in [-2, 2], integral 1."""
    ax = np.abs(x)
    return np.where(ax < 1, 2 / 3 - ax ** 2 + ax ** 3 / 2,
                    np.where(ax < 2, (2 - ax) ** 3 / 6, 0.0))


def greshoProfile(r, cuspWidth=0.0):
    """(v_phi, P) of the Gresho-Chan vortex at radius `r` (torch tensor, rho = 1).

    `cuspWidth = 0` is the standard piecewise profile (v_phi = 5r, 2 - 5r, 0;
    its pressure is the closed form in the original paper). For `cuspWidth = w`
    the standard v_phi, odd-extended through r = 0 (it is linear there, so the
    origin is untouched), is convolved with a cubic B-spline of half-width w:
    the two slope discontinuities become C^3, everything away from r = 0.2 and
    0.4 (+- w) is unchanged. The pressure is then re-derived from the radial
    balance dP/dr = v_phi^2 / r, integrated inwards from the standard outer
    value P(r >= 0.4 + w) = 3 + 4 ln 2, so the smoothed vortex is again an
    exact steady state. Computed on a 1e-5 radial grid and interpolated.
    """
    if cuspWidth <= 0:
        vphi = torch.where(r < 0.2, 5 * r, torch.where(r < 0.4, 2 - 5 * r, torch.zeros_like(r)))
        rs = r.clamp(min=1e-30)
        P = torch.where(r < 0.2, 12.5 * r ** 2 + 5,
            torch.where(r < 0.4, 12.5 * r ** 2 - 20 * r + 4 * torch.log(5 * rs) + 9,
                        torch.full_like(r, _P_OUTER)))
        return vphi, P
    if cuspWidth >= 0.1:
        raise ValueError('cuspWidth must stay below 0.1 so the smoothed regions of the two cusps do not overlap')
    dr = _PROFILE_DR
    rg = np.arange(0.0, 1.5, dr)
    rfull = np.concatenate([-rg[:0:-1], rg])
    tent = np.where(np.abs(rfull) < 0.2, 5 * rfull,
                    np.where(np.abs(rfull) < 0.4, np.sign(rfull) * (2 - 5 * np.abs(rfull)), 0.0))
    kx = np.arange(-2 * cuspWidth, 2 * cuspWidth + dr / 2, dr)
    # kernel of half-width w: B-spline argument scaled so its support is [-w, w]
    k = _cubicBSpline(kx / (cuspWidth / 2)); k /= k.sum()
    vfull = fftconvolve(tent, k, mode="same")
    vg = vfull[len(rg) - 1:]
    integrand = np.where(rg > 0, vg ** 2 / np.maximum(rg, 1e-30), 0.0)
    # P(r) = P_outer - int_r^inf v^2/r' dr'  (trapezoid, cumulated from the outside)
    seg = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    Pg = _P_OUTER - tail
    rn = r.detach().double().cpu().numpy()
    vphi = torch.as_tensor(np.interp(rn, rg, vg), dtype=r.dtype, device=r.device)
    P = torch.as_tensor(np.interp(rn, rg, Pg), dtype=r.dtype, device=r.device)
    return vphi, P


def sampleGreshoVortex(nx, config, schemeConfig, SimulationState, SimulationSystem, cuspWidth=0.0):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    

    positions = compressibleSystem.state.positions
    r = torch.linalg.norm(positions, dim=-1)

    vinitial_angular, Pinitial = greshoProfile(r, cuspWidth)
    Pinitial = Pinitial.to(compressibleSystem.state.densities.dtype)
    vinitial_angular = vinitial_angular.to(compressibleSystem.state.densities.dtype)

    vinitial = torch.zeros_like(positions)
    vinitial[:,0] = -vinitial_angular * positions[:,1] / r
    vinitial[:,1] = vinitial_angular * positions[:,0] / r

    u = 1 / (schemeConfig.gamma - 1) * (Pinitial / compressibleSystem.state.densities)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = compressibleSystem.state.densities, gamma = schemeConfig.gamma)

    compressibleSystem.state.velocities = vinitial
    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s

    config.dt = computeTimestep(compressibleSystem, config, schemeConfig)
    return compressibleSystem