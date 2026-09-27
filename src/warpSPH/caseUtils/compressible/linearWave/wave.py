"""Linear acoustic wave initial state and reference plot, used by `warpSPH.cases.linearWave`.

`sampleLinearWave` samples a regular 1D lattice with a sinusoidal pressure
perturbation of amplitude `A` and wavelength `lamda` on top of rest density
`rho0`, then iteratively (`nIters`) re-corrects particle mass against the
target density profile via `evaluateOptimalSupport` before setting velocity
from the linearized acoustic relation `v = c_s * delta_i` and deriving
energy/entropy through `idealGasEOS`. `plotState` overlays the current
density/pressure against that same analytic traveling-wave profile advected
at `c_s`.
"""

from ....modules.timestep.compressible import computeTimestep

# final import blocks that are generic
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
import numpy as np

# custom SPH libraries
from warpSPHIntegrators.integration import *
from warpSPHCore import *

# This library
from warpSPH import *
import torch

__all__ = ['sampleLinearWave', 'plotState']


def sampleLinearWave(
    nx,
    config,
    compressibleSPHConfig,
    SimulationState,
    SimulationSystem,
    A, lamda, c_s, rho0, gamma,
    nIters = 16,
    supportScheme = AdaptiveSupportScheme.NoScheme
):

    particles_l = sampleRegularParticles(nx, config.domain, config.targetNeighbors, jitter = 0.0)


    delta_i = A * torch.sin(2 * np.pi * particles_l.positions[:, 0] / lamda)
    P0 = c_s**2 * rho0 / gamma

    pressures = P0 + delta_i
    supports = volumeToSupport(1 / nx, dim = 1, targetNeighbors = config.targetNeighbors)

    # print(f'Initial Pressures: {pressures.min().item()} to {pressures.max().item()}')

    # Pinitial = torch.zeros_like(particles_l.densities)
    # r = torch.linalg.norm(particles_l.positions, dim = -1)
    # rhoInitial = torch.ones_like(particles_l.densities)
    # v_initial = - torch.nn.functional.normalize(particles_l.positions, dim = -1)
    # particles_l = particles_l._replace(masses = particles_l.masses * rhoInitial, densities = rhoInitial)

    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = particles_l.densities, gamma = gamma)

    # internalEnergy = u_ 
    # kineticEnergy = torch.linalg.norm(v_initial, dim = -1) **2/ 2
    # totalEnergy = (internalEnergy + kineticEnergy) * particles_l.masses


    simulationState = SimulationState(
        positions = particles_l.positions,
        supports = particles_l.supports,
        masses = particles_l.masses,
        densities = particles_l.densities,        
        velocities = torch.zeros_like(particles_l.positions),

        kinds = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_l.positions.shape[0], device = config.device, dtype = torch.int32),
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

    currScheme = compressibleSPHConfig.adaptiveSupportScheme
    compressibleSPHConfig.adaptiveSupportScheme = supportScheme

    adjacency = None

    for i in range(16):    


        adjacency = buildVerletList(
            simulationState, 
            config.domain, verletScale = 1.4, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)
        
        rho = warpOperation(
            simulationState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
            ),
            domain = config.domain,
            adjacency = adjacency,
        )
        simulationState.densities = rho
        rho, h_optimal, _, *_ = evaluateOptimalSupport(simulationState, config, compressibleSPHConfig, SupportScheme.Gather, adjacency)


        # apparentVolume, densities, crkState = computeCRKFactors(simulationState, config.domain, config.kernel, adjacency = adjacency)

        samplingError = torch.mean(rho - rho0)
        error = rho - (rho0 + delta_i)
        masses = simulationState.masses*( 1 - error)
        
        print(f'Iter {i+1}/{nIters}, Sampling Error: {samplingError:.3e}, Density Error: {error.abs().mean():.3e}, Mass correction factor: {masses.mean()/simulationState.masses.mean():.3f}')



        simulationState.supports = h_optimal
        simulationState.masses = masses
        simulationState.densities = rho
    compressibleSPHConfig.adaptiveSupportScheme = currScheme

    A_, u_, P_, c_ = idealGasEOS(A = None, u = None, P = pressures, rho = rho, gamma = gamma)

    # c_[:] = c_s
    v_ = (c_s * delta_i).view(-1,1)
    u_[:] = 1/(gamma - 1) - 1 / (gamma * (delta_i + 1))

    simulationState.internalEnergies = u_
    simulationState.pressures = P_
    simulationState.soundspeeds = c_
    simulationState.entropies = A_
    simulationState.totalEnergies = u_ + torch.linalg.norm(v_, dim = -1)**2/2 

    simulationState.velocities = v_

    compressibleSystem = SimulationSystem(
        state=simulationState, 
        adjacency = None, 
        domain = config.domain)

    dx = simulationState.masses.min()
    config.dx = dx
    config.dt = computeTimestep(compressibleSystem, config, compressibleSPHConfig, dt = config.dt)

    return compressibleSystem




def plotState(fig, axis, runningSystem, compressibleSystem, config, schemeConfig, rho0, A, lamda, c_s):
    for ax in axis.flatten():
        ax.cla()
    currentPositions = runningSystem.state.positions
    l = 1
    time = runningSystem.t
    reference = (rho0 +  A * torch.sin(2 * np.pi * currentPositions[:,0] / l - time * c_s / lamda * 2 * np.pi)).cpu().numpy()


    axis[0,0].scatter(runningSystem.state.positions.cpu(), runningSystem.state.densities.cpu(), s = 32/config.nx)
    axis[0,0].scatter(compressibleSystem.state.positions.cpu(), reference, s = 32/config.nx)
    axis[0,0].set_title('Density')
    axis[0,0].set_xlabel('x')
    axis[0,0].set_ylabel('Density')
    # axis[0,0].axhline(rho_s, color = 'r', linestyle = '--')

    axis[0,1].scatter(runningSystem.state.positions.cpu(), runningSystem.state.pressures.cpu(), s = 32/config.nx)
    axis[0,1].scatter(compressibleSystem.state.positions.cpu(), compressibleSystem.state.pressures.cpu(), s = 32/config.nx)
    axis[0,1].set_title('Pressure')
    axis[0,1].set_xlabel('x')
    axis[0,1].set_ylabel('Pressure')
    # axis[0,1].axhline(P_s, color = 'r', linestyle = '--')
    # axis[0,1].axvline(v_s * runningSystem.t, color = 'black', linestyle = ':', alpha = 0.5)
    # axis[0,1].axvline(-v_s * runningSystem.t, color = 'black', linestyle = ':', alpha = 0.5)

    axis[0,2].scatter(runningSystem.state.positions.cpu(), runningSystem.state.densities.cpu() - reference, s = 32/config.nx)
    # axis[0,2].scatter(compressibleSystem.state.positions.cpu(), compressibleSystem.state.velocities.cpu()[:,0], s = 32/config.nx)
    axis[0,2].set_title('Error')


    kineticEnergy = 0.5 * (torch.linalg.norm(runningSystem.state.velocities, dim = -1) **2 * runningSystem.state.masses).sum()
    thermalEnergy = (runningSystem.state.internalEnergies * runningSystem.state.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy
    fig.suptitle(f'{schemeConfig.schemeName}\n Linear Wave Problem, t = {runningSystem.t:2f}, dt = {config.dt:.3g}, ptcls = {len(runningSystem.state.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}')
    # fig.canvas.draw()
    for ax in axis.flatten():
        ax.set_xlim(-0.5, 0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()
