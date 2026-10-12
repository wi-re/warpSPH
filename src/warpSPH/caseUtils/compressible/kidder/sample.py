"""Kidder 1D initial state, used by `warpSPH.cases.compressible.kidder`.

Samples a regular 1D lattice on `[r_inner, r_outer]` (periodic domain sized
`1.1 * r_outer`), evaluates `KidderIsentropicCapsuleAnalyticSolution` at
`t=0` for density and pressure, and derives internal energy/pressure/sound
speed via `idealGasEOS`. Velocity starts at zero even though the analytic
solution is not static at `t=0`; the driven boundary bands from
`buildKidderBCs` are what keep the shell converging once the case's
`postStep` hook takes over. Large stretches of print/debug diagnostics are
commented out.
"""

from .kidder import KidderIsentropicCapsuleAnalyticSolution
from ....modules.timestep.compressible import computeTimestep
from warpSPH import *
from warpSPHCore import *
import torch
import numpy as np

__all__ = ['buildKidder']


def buildKidder(
    config,
    schemeConfig,
    SimulationState,
    SimulationSystem,
    r_inner, r_outer, P_inner, P_outer, rho_outer, nu, gamma,
):
    device = config.device
    dtype = config.dtype
    nx = config.nx

    domain = buildDomainDescription(r_outer * 1.1, 1, periodic = True, device = device, dtype = dtype)
    dim = 1


    # particles_l = sampleRegularParticles(nx + 20, buildDomainDescription(r_outer - r_inner + dx * 20, dim, periodic = False, device = device, dtype = dtype), targetNeighbors, jitter = 0.0)
    # print(nx)
    particles_l = sampleRegularParticles(nx, buildDomainDescription(r_outer - r_inner, dim, periodic = True, device = device, dtype = dtype), config.targetNeighbors, jitter = 0.0)

    normalized_positions = (particles_l.positions - particles_l.positions.min()) / (particles_l.positions.max() - particles_l.positions.min())

    offset_positions = normalized_positions * (r_outer - r_inner) + r_inner

    particles_l = particles_l._replace(positions = offset_positions)


    dx = (particles_l.positions[1] - particles_l.positions[0]).cpu().numpy()

    # print(particles_l.positions[1] - particles_l.positions[0])
    # print(particles_l.positions.shape)
    # print(particles_l.positions.min(), particles_l.positions.max())

    kidderSolution = KidderIsentropicCapsuleAnalyticSolution(
        nu = nu,
        r0 = r_inner,
        r1 = r_outer,
        P0 = P_inner,
        P1 = P_outer,
        rho1 = rho_outer,
    )

    particles_l = particles_l._replace(
        densities = torch.tensor(kidderSolution.rho(0, torch.linalg.norm(particles_l.positions, dim = -1).cpu().numpy()), dtype = dtype, device = device),
        masses = torch.tensor(kidderSolution.rho(0, torch.linalg.norm(particles_l.positions, dim = -1).cpu().numpy()) * dx, dtype = dtype, device = device),
    )

    t = 0
    Pinitial = torch.tensor(kidderSolution.P(t, torch.linalg.norm(particles_l.positions, dim = -1).cpu().numpy()), dtype = dtype, device = device)
    # Pinitial[:10] = pInner
    # Pinitial[-10:] = pOuter
    print(Pinitial.shape)

    dx = (particles_l.positions[1] - particles_l.positions[0]).cpu().numpy()
    dx = (r_outer - r_inner) / (nx-1)

    particles_l = particles_l._replace(
        densities = torch.tensor(kidderSolution.rho(0, torch.linalg.norm(particles_l.positions, dim = -1).cpu().numpy()), dtype = dtype, device = device),
        masses = torch.tensor(kidderSolution.rho(0, torch.linalg.norm(particles_l.positions, dim = -1).cpu().numpy()) * dx, dtype = dtype, device = device),
    )

    simulationState = SimulationState(
        positions = particles_l.positions,
        supports = particles_l.supports,
        masses = particles_l.masses,
        densities = particles_l.densities,        
        velocities = torch.zeros_like(particles_l.positions),

        kinds = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_l.positions.shape[0], device = device, dtype = torch.int32),
        UIDcounter= particles_l.positions.shape[0],
        
        internalEnergies = None,
        totalEnergies = None,
        entropies = None,
        pressures = None,
        soundspeeds = None,

        alphas = torch.ones_like(particles_l.densities),
        alpha0s = torch.ones_like(particles_l.densities),
        divergence = torch.zeros_like(particles_l.densities)
    )

    densities = warpOperation(
        simulationState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather,
        ),
        domain = config.domain,
        adjacency=None
    )
    # simulationState.densities[band:-band] = densities[band:-band]

    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = simulationState.densities, gamma = schemeConfig.gamma)
    v_initial = torch.zeros_like(particles_l.positions)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(v_initial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles_l.masses



    simulationState.internalEnergies = u_
    simulationState.pressures = P_
    simulationState.soundspeeds = c_s
    simulationState.entropies = A_
    # total (not specific) energy: `totalEnergies` is integrated with the total rate `dEdt` (X3)
    simulationState.totalEnergies = (u_ + torch.linalg.norm(v_initial, dim = -1)**2/2) * simulationState.masses 

    simulationState.velocities = v_initial

    kineticEnergy = 0.5 * (torch.linalg.norm(simulationState.velocities, dim = -1) **2 * simulationState.masses).sum()
    thermalEnergy = (simulationState.internalEnergies * simulationState.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy

    kineticEnergy = 0.5 * (torch.linalg.norm(simulationState.velocities, dim = -1) **2 * simulationState.masses).sum()
    thermalEnergy = (simulationState.internalEnergies * simulationState.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy

    # err = kidderInternalEnergy(0, particles_l.positions).flatten() - u_
    # print('Initial total energy: ', totalEnergy.sum().item())
    # print('Initial kinetic energy: ', kineticEnergy.sum().item())
    # print('Initial thermal energy: ', thermalEnergy.sum().item())
    # print('Max internal energy error: ', err.abs().max().item())

    # err_rho = kidderSolution.rho(0, particles_l.positions.cpu().numpy()).flatten() - simulationState.densities.cpu().numpy()
    # print('Max density error: ', np.abs(err_rho).max().item())
    # err_pressure = kidderSolution.P(0, particles_l.positions.cpu().numpy()).flatten() - simulationState.pressures.cpu().numpy()
    # print('Max pressure error: ', np.abs(err_pressure).max().item())

    # sampled_rho = densities.cpu().numpy()[band:-band]
    # sampled_solution_rho = kidderSolution.rho(0, simulationState.positions.cpu().numpy())[band:-band].flatten()
    # print('Max sampled density error: ', np.abs(sampled_solution_rho - sampled_rho).max().item())

    # print(f'Current Density: {simulationState.densities}')
    # print(f'Target Density: {kidderSolution.rho(0, simulationState.positions.cpu().numpy()).flatten()}')

    # print(f'Current Pressure: {simulationState.pressures}')
    # print(f'Target Pressure: {kidderSolution.P(0, simulationState.positions.cpu().numpy()).flatten()}')

    # print(f'Current Internal Energy: {simulationState.internalEnergies}')
    # print(f'Target Internal Energy: {kidderInternalEnergy(0, simulationState.positions).flatten()}')

    # print(f'Initial total energy: {totalEnergy:.3g}, kinetic energy: {kineticEnergy:.3g}, thermal energy: {thermalEnergy:.3g}')

    # rho = warpOperation(
    #     simulationState,
    #     OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Density,
    #         supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
    #     ),
    #     domain = config.domain,
    #     adjacency = None,
    # )

    # print('Initial density: ', rho)


    compressibleSystem = SimulationSystem(
        state=simulationState, 
        adjacency = None, 
        domain = config.domain)

    # dx = simulationState.positions[1] - simulationState.positions[0]
    config.dx = dx
    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = config.dt)
    # config.dt = 1e-4
    print(f"Initial timestep: {config.dt}")
        
    return compressibleSystem, kidderSolution