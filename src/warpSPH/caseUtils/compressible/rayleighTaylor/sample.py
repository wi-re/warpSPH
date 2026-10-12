"""Builds the Rayleigh-Taylor initial state for `cases/compressible/rayleighTaylor.py`:
heavy fluid (`rho_t`) over light (`rho_b`) in a tall periodic-in-x box, with a
smooth logistic density interface at ``y = 0.5`` (see `bcs.py`), a
hydrostatic pressure profile balancing the constant gravity `g`, and a small
seeded vertical velocity perturbation (a product of cosines in `x` and `y`,
nonzero only in the band ``0.3 <= y <= 0.7``) that seeds the instability.
Registers two `BoundaryCondition`s on `schemeConfig` -- `rayleighTaylorBC`,
which pins the top/bottom buffer bands to the unperturbed profile via
`bcs.py`'s Dirichlet functions, and `gravityBC`, which applies `forcing.py`'s
constant downward acceleration everywhere -- then relaxes supports/densities
with `evaluateOptimalSupport` and derives the EOS state from the analytic
pressure via `idealGasEOS`.
"""

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *
from ....modules.timestep.compressible import computeTimestep
import math
import numpy as np
from warpSPH import *

from .bcs import *
from .forcing import *
from .sdf import *

__all__ = ['sampleRayleighTaylor']


def sampleRayleighTaylor(rho_b, rho_t, delta, g, L, dx, aspect, nx, config, schemeConfig, SimulationState, SimulationSystem):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)

    gamma = schemeConfig.gamma
    P_0 = rho_t / gamma

    positions = compressibleSystem.state.positions
    x = positions[:,0]
    y = positions[:,1]

    # rho_y = rho_b + (rho_t - rho_b) * (1 + torch.exp(-(y - 0.5) / delta))**(-1)
    rho_y = rayleighTaylor_rho(positions, rho_b, rho_t, delta)

    delta_y = delta * 5
    v_y = delta_y * (1 + torch.cos(8 * np.pi * (x + 0.25))) * (1 + torch.cos(5 * np.pi * (y - 0.5)))
    v_y[~((y >= 0.3) & (y <= 0.7))] = 0

    P = P_0 - g * rho_y * (y - 1/2)

    Pinitial = P
    rhoInitial = rho_y
    vInitial = torch.stack([torch.zeros_like(v_y), v_y], dim=-1)

    particles = compressibleSystem.state


    rayleighTaylorBC = BoundaryCondition(
        type = BoundaryConditionType.dynamic,
        sdf = lambda x: (buffer_sdf(x, L), buffer_sdf_gradient(x, L)),
        dirichletFunctions = {
            'velocities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: RayleighTaylorVelocity(positions),
            'densities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: RayleighTaylorDensity(positions, rho_b, rho_t, delta),
            'internalEnergies': lambda state, cfg, schemeCfg, positions, d, n, t, dt: RayleighTaylorInternalEnergy(positions, rho_b, rho_t, delta, g, gamma),
        },
            updateFunctions = {
                'dvdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: RayleighTaylorAcceleration(positions),
                'dxdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: RayleighTaylorVelocity(positions),
            }
    )

    gravityBC = BoundaryCondition(
        type = BoundaryConditionType.dynamic,
        sdf = lambda x: (-1 * torch.ones_like(x[:,0]), torch.tensor([0,-1], device = x.device, dtype = x.dtype).expand(x.shape[0], -1)),
        forcingFunctions=[
            lambda state, cfg, schemeCfg, positions, d, n, t, dt: gravityForcing(state, cfg, schemeCfg, positions, d, n, t, dt, g),
        ]
    )
        
    schemeConfig.boundaryConditions.clear()
    schemeConfig.boundaryConditions.append(rayleighTaylorBC)
    schemeConfig.boundaryConditions.append(gravityBC)

    enforceDirichlet(compressibleSystem, compressibleSystem.t, config.dt, config, schemeConfig)

    
    compressibleSystem.state.masses = dx**config.dim / aspect**config.dim * rhoInitial

    compressibleSPHConfigAdaptiveH = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.Owen,
    )

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(compressibleSystem.state, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdaptiveH)

    compressibleSystem.state.supports = h_optimal
    compressibleSystem.state.densities = rho_optimal

    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = compressibleSystem.state.densities, gamma = gamma)
# v_initial = torch.zeros_like(particles_l.positions)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(vInitial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * compressibleSystem.state.masses

    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.totalEnergies = totalEnergy
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s
    compressibleSystem.state.velocities = vInitial
    # compressibleSystem.state.densities = rhoInitial

    # config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = None)
    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = None) * 2/3

    return compressibleSystem