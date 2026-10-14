"""The MFV (meshless finite-volume) step: the MFM fluxes plus the GIZMO mass-flux limiter, which redistributes mass across faces (dm != 0).

`mfvStep` is `runMeshlessStep` in mode 'MFV'. ``temporalScheme='auto'`` resolves to 'hancock' (the D4 measurement). The 'hancock' update has a second,
EXPERIMENTAL form selected by ``mfvUpdate='primitive'`` (`gizmoPrimitiveUpdate`).
"""

import torch

from .step import runMeshlessStep
from .temporal import conservedHancockUpdate

__all__ = ['mfvStep', 'gizmoPrimitiveUpdate']


def gizmoPrimitiveUpdate(ctx):
    """GIZMO's update (hydro_toplevel.cc, kicks.cc): the velocity and the internal energy are kicked from the *full* (uncapped) fluxes; only the mass
    sees the per-pair cap (`massFluxLimit`, default 10 %), floored at -0.99 m. The momentum / energy flux are independent of the mass cap, so the
    capped and the full rates differ only in column 0. Returns ``(dQdt, Qn)``, ``dQdt = (Qn - Q) / dt``."""
    dt, Q, vel, u = ctx.dt, ctx.Q, ctx.vel, ctx.u
    full = ctx.rates(0.0)
    capped = ctx.rates(ctx.schemeConfig.massFluxLimit)
    m = Q[:, 0:1]
    Fm, Fp, Fe = full[:, 0:1], full[:, 1:-1], full[:, -1:]
    dm = torch.maximum(dt * capped[:, 0:1], -0.99 * m)
    mNew = m + dm
    a = (Fp - vel * Fm) / m
    dUdt = (Fe - (vel * Fp).sum(-1, keepdim=True) + 0.5 * (vel * vel).sum(-1, keepdim=True) * Fm
            - u[:, None] * Fm) / m
    vNew = vel + dt * a * (m / mNew)
    uNew = torch.maximum(u[:, None] + dt * dUdt, 0.5 * u[:, None])
    Qn = torch.cat([mNew, mNew * vNew, mNew * (uNew + 0.5 * (vNew * vNew).sum(-1, keepdim=True))], dim=-1)
    return (Qn - Q) / dt, Qn


def mfvStep(system, dt, config, schemeConfig, verbose = False):
    update = gizmoPrimitiveUpdate if schemeConfig.mfvUpdate == 'primitive' else conservedHancockUpdate
    return runMeshlessStep(system, dt, config, schemeConfig, mode='MFV', hancockUpdate=update)
