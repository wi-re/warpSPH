"""Yee isentropic vortex initial state, used by `warpSPH.cases.compressible.yeeVortex`.

Samples particles on concentric shells (`sampleShellv2`, `nr` rings) rather
than a lattice, imposes the Yee vortex's Gaussian velocity/temperature
perturbation (amplitude `beta`, centered at `(xc, yc)`) analytically, then
iteratively refines density/support/mass via `evaluateOptimalSupport` and CRK
renormalization before deriving pressure/energy through `idealGasEOS`. The
outermost `buffer_rings` shells are flagged (`indices`) and returned along
with a dynamic `BoundaryCondition` (`yeeBC`) that holds them at their `t=0`
state -- selecting particles by shell index rather than an actual SDF, mirroring
the pattern in `kidder/bc.py`. Contains extensive commented-out
diagnostic/plotting code.
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

__all__ = ['sampleYeeVortex']


def sampleYeeVortex(nr, nx, buffer_rings, config, schemeConfig, extraData, SimulationState, SimulationSystem):
    xc = extraData['xc']
    yc = extraData['yc']
    beta = extraData['beta']
    gamma = extraData['gamma']
    P_infty = extraData['P_infty']
    rho_infty = extraData['rho_infty']
    rho0 = extraData['rho0']
    particles_, shells = sampleShellv2(nr, config.domain, config.targetNeighbors)

    # fig, axis = plt.subplots(1, 2, figsize=(11,5), squeeze=False)

    indices = torch.hstack([torch.ones(s[4].shape[0], device = particles_.positions.device, dtype = torch.int32) * s[0] for s in shells])
    # colors = torch.rand((len(shells), 3), device = particles_.positions.device)
    # sampledColors = colors[indices.long()]

    # r = torch.norm(particles_.positions, dim=1)

    # axis[0,0].scatter(particles_.positions[:,0].cpu(), particles_.positions[:,1].cpu(), s=5, c=sampledColors.cpu(), cmap = 'tab20')
    # axis[0,0].set_title('Particle Positions')

    r = torch.norm(particles_.positions, dim=1)

    deltav_term = beta / (2 * math.pi) * torch.exp( ( 1 - r**2 ) / 2)
    deltav_x = deltav_term * (- particles_.positions[:, 1] + yc)
    deltav_y = deltav_term * (particles_.positions[:, 0] - xc)

    deltaT = - ( ( gamma - 1) * beta**2 ) / ( 8 * gamma * math.pi**2 ) * torch.exp( 1 - r**2 )

    T_infty = P_infty / rho_infty

    T = T_infty + deltaT
    rhoInitial = T**(1 / (gamma - 1))
    Pinitial = rhoInitial * T
    uInitial = 1 / (gamma - 1) * (Pinitial / rhoInitial)


    actualMasses = rhoInitial / rho0 * particles_.masses

    v_initial = torch.stack([deltav_x, deltav_y], dim = 1)


    # from optimalSupport import evaluateOptimalSupport

    # dim = 2
    # domain = buildDomainDescription(L, dim, periodic = periodicDomain, device = device, dtype = dtype)

    # particles_ = sampleRegularParticles(nx, config.domain, config.targetNeighbors)

    # particles_ = sampleShell(32, config.domain, config.targetNeighbors, circle = True, extraRings=0)
    # print(f'Number of particles: {particles_.positions.shape[0]}')

    # domain = buildDomainDescription(domainExtent * 1.5, dim, periodic = periodicDomain, device = device, dtype = dtype)
    particles_ = particles_._replace(masses = particles_.masses * schemeConfig.rho0)
    L = config.domain.max[0] - config.domain.min[0]
    config.dx = L / nx

    device = config.device

    particles = SimulationState(
        positions = particles_.positions,
        supports = particles_.supports,
        masses = actualMasses,
        densities = particles_.densities,
        velocities = v_initial,
        
        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        UIDcounter = particles_.positions.shape[0],
        
        internalEnergies = None,
        totalEnergies = None,
        entropies = None,
        pressures = None,
        soundspeeds = None,

        divergence=torch.zeros_like(particles_.densities),
        alpha0s= torch.ones_like(particles_.densities),
        alphas= torch.ones_like(particles_.densities),
    )


    densities = warpOperation(
        particles, 
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather,
            gradientMode = config.gradientMode,
            laplacianMode = config.laplacianMode,
        ),
        domain = config.domain,
    )
    particles.densities = densities

    compressibleSPHConfigAdapt = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme,
    )

    # neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    # numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    # config = {'targetNeighbors': targetNeighbors, 'domain': domain, 'support': {'iterations': 16, 'scheme': 'Monaghan'}, 'neighborhood': {'algorithm': 'compact'}}
    # rho, h, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)


    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particles, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdapt)


    adjacency = buildVerletList(particles, domain = config.domain, verletScale = 1.0, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)

    print(f'Optimal Support min: {h_optimal.min()}, max: {h_optimal.max()}, mean: {h_optimal.mean()}')
    print(f'Number of neighbors min: {adjacency.numNeighbors.min()}, max: {adjacency.numNeighbors.max()}, mean: {adjacency.numNeighbors.float().mean()}')



    # compressibleSPHConfig = CompressibleSPHConfig(
    #     adaptiveSupportIterations=16,
    #     adaptiveSupportThreshold=1e-3,
    #     adaptiveSupportScheme=AdaptiveSupportScheme.Owen,
    # )

    # neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    # numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    # config = {'targetNeighbors': targetNeighbors, 'domain': domain, 'support': {'iterations': 16, 'scheme': 'Monaghan'}, 'neighborhood': {'algorithm': 'compact'}}
    # rho, h, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particles, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdapt)



    adjacency = buildVerletList(particles, domain = config.domain, verletScale = 1.0, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)

    print(f'Optimal Support min: {h_optimal.min()}, max: {h_optimal.max()}, mean: {h_optimal.mean()}')
    print(f'Number of neighbors min: {adjacency.numNeighbors.min()}, max: {adjacency.numNeighbors.max()}, mean: {adjacency.numNeighbors.float().mean()}')



    # particleState.supports = h_optimal

    particles.densities = rho_optimal
    particles.supports = h_optimal

    # P_initial = torch.ones_like(particles.densities)
    u = 1 / (gamma - 1) * (Pinitial / rho_optimal)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = rho_optimal, gamma = gamma)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(torch.zeros_like(particles.positions), dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles.masses

    simulationState_ = SimulationState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities,        
        velocities = v_initial,

        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        UIDcounter = particles.positions.shape[0],
        
        internalEnergies = u_,
        totalEnergies = totalEnergy,
        entropies = A_,
        pressures = P_,
        soundspeeds = c_s,

        alphas = torch.ones_like(particles.densities),
        alpha0s = torch.ones_like(particles.densities),
        divergence=torch.zeros_like(particles.densities),
    )

    # area = L**dim / (nx**dim)
    # rho_low = 1
    # rho_high = 2

    # mask = torch.logical_and(particles.positions[:,0].abs() < L/4, particles.positions[:,1].abs() < L/4)

    # simulationState_.masses[mask] = area * rho_high
    # simulationState_.densities[mask] = rho_high
    # simulationState_.masses[~mask] = area * rho_low
    # simulationState_.densities[~mask] = rho_low

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particles, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdapt)
    # simulationState_.densities = rho_optimal
    adjacency = buildVerletList(
        simulationState_, 
        config.domain, verletScale = 1.4, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None,
        verbose = False)

    apparentVolume, simulationState_.densities, crkState = computeCRKFactors(simulationState_, config.domain, config.kernel, adjacency = adjacency)


    # P_initial = torch.ones_like(particles.densities)
    # u = 1 / (gamma - 1) * (P_initial / simulationState_.densities)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = simulationState_.densities, gamma = gamma)
    simulationState_.internalEnergies = u_
    simulationState_.pressures = P_
    simulationState_.soundspeeds = c_s


    print(f"min density: {simulationState_.densities.min()}, max density: {simulationState_.densities.max()}")
    print(f"min mass: {simulationState_.masses.min()}, max mass: {simulationState_.masses.max()}")
    print(f"min support: {simulationState_.supports.min()}, max support: {simulationState_.supports.max()}")

    adjacency = buildVerletList(simulationState_, 
                            domain = config.domain,
                            verletScale = 2**(1/config.dim), supportMode = config.supportMode)

    compressibleSystem = SimulationSystem(
        state=simulationState_, 
        adjacency = adjacency, 
        domain = config.domain)


    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = None)
    initialState = (v_initial, rhoInitial, uInitial)

    def YeeVelocity(t_, x, initialState):
        return initialState[0]
    def YeeDensity(t_, x, initialState):
        return initialState[1]
    def YeeInternalEnergy(t_, x, initialState):
        return initialState[2]
    def YeeAcceleration(t_, x, initialState):
        return torch.zeros_like(x)

    mask = indices >= (len(shells) - buffer_rings)
    print(f'Buffer Particles: {mask.sum()} out of {len(particles_.positions)} [ {mask.sum() / len(particles_.positions) * 100:.2f}% ]')

    def buffer_sdf(position):
        dist = torch.ones_like(position[:,0])
        dist[mask] = -1
        # dist[:] = -1
        return dist
    def buffer_sdf_gradient(position):
        # for the gradient we set the non buffer particles to point outwards, and the buffer particles to point inwards
        dist = position.clone()
        dist[mask] = -dist[mask]
        return torch.nn.functional.normalize(dist, dim=1)

    sdf = buffer_sdf(particles_.positions)
    sdf_gradient = buffer_sdf_gradient(particles_.positions)

    yeeBC = BoundaryCondition(
        type = BoundaryConditionType.dynamic,
        sdf = lambda x: (buffer_sdf(x), buffer_sdf_gradient(x)),
        dirichletFunctions = {
            'velocities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: YeeVelocity(t, positions, initialState),
            'densities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: YeeDensity(t, positions, initialState),
            'internalEnergies': lambda state, cfg, schemeCfg, positions, d, n, t, dt: YeeInternalEnergy(t, positions, initialState),
        },
        updateFunctions = {
            'dvdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: YeeAcceleration(t, positions, initialState),
            'dxdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: YeeVelocity(t, positions, initialState),
        }
    )


    return compressibleSystem, indices, yeeBC
        