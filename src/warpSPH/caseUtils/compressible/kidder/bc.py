"""Driven boundary bands for the Kidder case, used by `warpSPH.cases.compressible.kidder`.

`buildKidderBCs` builds a dynamic `BoundaryCondition` that pins velocity,
density, and internal energy on the first/last `band` particles (the inner
and outer shell) to `KidderIsentropicCapsuleAnalyticSolution` at every step,
via `buffer_sdf`/`buffer_sdf_gradient` selecting those particles by index
rather than by an actual signed distance. The module-level `kidderSolution`
name is unused (each call closes over its own `kidderSolution` argument
instead); `kidderAcceleration` is defined but not wired into any BC function.
"""

import torch
kidderSolution = None

def kidderVelocity(t_, x, schemeConfig, kidderSolution):
    t = t_.cpu().numpy() if torch.is_tensor(t_) else t_
    r = torch.linalg.norm(x, dim = -1)
    # P = particles.pressures.cpu().numpy()
    # P = kidderSolution.P(t, r.cpu().numpy())
    Panalytic = kidderSolution.P(t, r.cpu().numpy())
    rhoAnalytic = kidderSolution.rho(t, r.cpu().numpy())

    uAnalytic = 1 / (schemeConfig.gamma - 1) * Panalytic / rhoAnalytic
    v = kidderSolution.vr(t, r.cpu().numpy())
    # print('Applying velocity, v = ', v, 't = ', t)
    return torch.tensor(v, dtype = x.dtype, device = x.device).view(-1,1)
def kidderDensity(t_, x, schemeConfig, kidderSolution):
    t = t_.cpu().numpy() if torch.is_tensor(t_) else t_
    r = torch.linalg.norm(x, dim = -1)
    # P = particles.pressures.cpu().numpy()
    # P = kidderSolution.P(t, r.cpu().numpy())
    Panalytic = kidderSolution.P(t, r.cpu().numpy())
    rhoAnalytic = kidderSolution.rho(t, r.cpu().numpy())

    uAnalytic = 1 / (schemeConfig.gamma - 1) * Panalytic / rhoAnalytic
    # print('Applying density')
    return torch.tensor(rhoAnalytic, dtype = x.dtype, device = x.device)
def kidderInternalEnergy(t_, x, schemeConfig, kidderSolution):
    t = t_.cpu().numpy() if torch.is_tensor(t_) else t_
    r = torch.linalg.norm(x, dim = -1)
    # P = particles.pressures.cpu().numpy()
    # P = kidderSolution.P(t, r.cpu().numpy())
    Panalytic = kidderSolution.P(t, r.cpu().numpy())
    rhoAnalytic = kidderSolution.rho(t, r.cpu().numpy())

    uAnalytic = 1 / (schemeConfig.gamma - 1) * Panalytic / rhoAnalytic
    # print('Applying internal energy')
    return torch.tensor(uAnalytic, dtype = x.dtype, device = x.device)

def kidderAcceleration(t_, x, schemeConfig, kidderSolution):
    t = t_.cpu().numpy() if torch.is_tensor(t_) else t_
    r = torch.linalg.norm(x, dim = -1)
    # P = particles.pressures.cpu().numpy()
    # P = kidderSolution.P(t, r.cpu().numpy())
    Panalytic = kidderSolution.P(t, r.cpu().numpy())
    rhoAnalytic = kidderSolution.rho(t, r.cpu().numpy())

    uAnalytic = 1 / (schemeConfig.gamma - 1) * Panalytic / rhoAnalytic
    v = kidderSolution.accelr(t, r.cpu().numpy())
    # print('Applying velocity, v = ', v, 't = ', t)
    return torch.tensor(v, dtype = x.dtype, device = x.device).view(-1,1)


def buffer_sdf(position, band):
    dist = torch.ones_like(position[:,0])
    dist[:band] = -1
    dist[-band:] = -1
    # dist[:] = -1
    return dist
def buffer_sdf_gradient(position, band):
    dist = torch.ones_like(position[:,0])
    dist[:band] = 1
    dist[-band:] = -1
    return dist

from warpSPH.modules.boundaryConditions import *
from warpSPH.configurations import *

__all__ = ['buildKidderBCs']


def buildKidderBCs(schemeConfig, kidderSolution, band):
    kidderBC = BoundaryCondition(
        type = BoundaryConditionType.dynamic,
        sdf = lambda x: (buffer_sdf(x, band), buffer_sdf_gradient(x, band)),
        dirichletFunctions = {
            'velocities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: kidderVelocity(t, positions, schemeCfg, kidderSolution),
            'densities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: kidderDensity(t, positions, schemeCfg, kidderSolution),
            'internalEnergies': lambda state, cfg, schemeCfg, positions, d, n, t, dt: kidderInternalEnergy(t, positions, schemeCfg, kidderSolution),
        },
        updateFunctions = {
            # 'dvdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: kidderAcceleration(t, positions),
            'dxdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: kidderVelocity(t, positions, schemeCfg, kidderSolution),
        }
    )
    return kidderBC