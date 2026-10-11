"""1D Sod shock tube sampling for `cases/sod.py`: the left/right Riemann
state container (`sodInitialState`) and `buildSod1D`, which lays out two
independently-sampled periodic lattices (dense left half, coarser right half
by `samplingRatio`) and derives the EOS state from the given pressures via
`idealGasEOS`. When `smoothIC` is set, the internal energy is additionally
ramped smoothly across the interface instead of taking the sharp step, to
compensate for SPH's own smoothing of the density field near the
discontinuity; that block carries a fair amount of commented-out debug/dead
code from earlier iterations of the ramp, left untouched (separately tracked
cleanup item), which the live smoothIC path does not depend on.
"""

from typing import NamedTuple

from warpSPH.configurations import SimulationConfig
from warpSPH.configurations.compressibleConfig import CompressibleSPHConfig
from warpSPH.utils import *
from warpSPHCore import *
import torch
from ....systems import *
from ....modules import idealGasEOS, evaluateOptimalSupport
# from optimalSupport import evaluateOptimalSupport

__all__ = ['sodInitialState', 'buildSod1D', 'buildSod1DMeshless']


class sodInitialState(NamedTuple):
    p: float
    rho: float
    v: float

from warpSPH.sample import *

from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPH.modules.timestep.compressible import computeTimestep
def buildSod1D(
    # nx: int,
    SimulationSystem, SimulationState,
    samplingRatio: int,
    leftState: sodInitialState,
    rightState: sodInitialState,
    gamma: float,
    config: SimulationConfig,
    smoothIC: bool = False,
    adaptiveSupportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.Monaghan,
):
    
    nx = config.nx 
    actualRatio = nx / (nx // samplingRatio)

    particles_l = sampleRegularParticles(nx, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)
    particles_r = sampleRegularParticles(nx // samplingRatio, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)

    particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min())
    pos_r = particles_r.positions
    pos_r[pos_r[:,0] < 0, 0] -= 0.5
    pos_r[pos_r[:,0] > 0, 0] += 0.5
    particles_r = particles_r._replace(positions = pos_r)

    # print(f'Left particles: {particles_l.positions.shape[0]}, Right particles: {particles_r.positions.shape[0]}')

    combinedPositions = torch.cat([particles_l.positions, particles_r.positions], dim = 0)
    tags = torch.cat([
        torch.zeros(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.ones(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)

    # particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min() * rightState.rho * actualRatio)
    # particles_l = particles_l._replace(masses = particles_l.masses * leftState.rho)


    leftMass = particles_l.masses.min() * leftState.rho
    rightMass = particles_l.masses.min() * rightState.rho * actualRatio



    combinedMasses = torch.where(tags == 0, leftMass, rightMass).to(combinedPositions.dtype)
    combinedSupports = torch.where(tags[:, None] == 0, particles_l.supports.min(), particles_r.supports.min())[:,0]
    combinedDensities = torch.where(tags == 0, leftState.rho, rightState.rho).to(combinedPositions.dtype)
    combinedVelocities = torch.where(tags[:, None] == 0, float(leftState.v), float(rightState.v)).to(combinedPositions.dtype)
    combinedKinds = torch.zeros_like(tags)
    combinedMaterials = tags
    combinedUIDs = torch.cat([
        torch.arange(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.arange(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)


    particleState = SimulationState(
        positions = combinedPositions,
        velocities = combinedVelocities,
        supports = combinedSupports,
        masses = combinedMasses,
        densities = combinedDensities,

        kinds = combinedKinds,
        materials = combinedMaterials,
        UIDs = combinedUIDs,
        UIDcounter=combinedUIDs.max() + 1,

        internalEnergies=None,
        totalEnergies=None,
        entropies=None,
        pressures=None,
        soundspeeds=None,
        
        divergence=torch.zeros_like(combinedDensities),
        alpha0s=torch.ones_like(combinedDensities),
        alphas=torch.ones_like(combinedDensities)
    )

    # print(f'Left state: p={leftState.p}, rho={leftState.rho}, v={leftState.v}')
    # print(f'Right state: p={rightState.p}, rho={rightState.rho}, v={rightState.v}')
    # print(f'Actual sampling ratio: {combinedPositions.shape[0] / particles_l.positions.shape[0]}')

    # print(f'Initial particle count: {combinedPositions.shape[0]}')
    # print(f'Positions Shape: {combinedPositions.shape}')
    # print(f'Masses Shape: {combinedMasses.shape}')
    # print(f'Densities Shape: {combinedDensities.shape}')
    # print(f'Velocities Shape: {combinedVelocities.shape}')
    # print(f'Supports Shape: {combinedSupports.shape}')
    # print(f'Kinds Shape: {combinedKinds.shape}')


    densities = warpOperation(
        particleState, 
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather,
            gradientMode = config.gradientMode,
            laplacianMode = config.laplacianMode,
        ),
        domain = config.domain,
    )
    compParams = CompressibleSPHConfig(gamma=gamma, adaptiveSupportCorrections=True, adaptiveSupportIterations=16, adaptiveSupportThreshold=1e-3, adaptiveSupportScheme=adaptiveSupportScheme)

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particleState, config, compParams, supportScheme = SupportScheme.Gather)
    particleState.supports = h_optimal

    # rho_optimal = densities
    # h_optimal = particleState.supports

    eosRho = rho_optimal if smoothIC else particleState.densities
    P_initial = torch.where(particleState.materials == 0, leftState.p, rightState.p)
    # particleState.densities = eosRho

    u = 1 / (gamma - 1) * (P_initial / eosRho)
    # particleState.densities = eosRho
    if smoothIC:
        # Due to the smoothing inherent in SPH the initial conditions for the Sod shock tube problem are not perfectly captured, especially near the discontinuity. 
        # In the simulation we use the current density field to evaluate the pressure using the internal energy. However, this can lead to deviations as the smoothed density field is not the one the conditions were designed for. Consequently, we need to smooth the internal energy field as well to ensure that the initial pressure field matches the intended conditions. This is done by using the same smoothing operation on the internal energy as we do for the density, which allows us to maintain consistency in the initial conditions and better capture the behavior of the shock tube problem in our SPH simulation.

        # print('Smoothing initial internal energy to match smoothed density field for consistent initial conditions.')
        # print(f'Initial internal energy before smoothing: {u.min().item():6.4g} to {u.max().item():6.4g}')
        
        # particleState.densities = eosRho
        # print(f'Initial density: {particleState.densities.min().item():6.4g} to {particleState.densities.max().item():6.4g}')
        # print(f'Initial pressure before smoothing: {P_initial.min().item():6.4g} to {P_initial.max().item():6.4g}')
        # P = warpOperation(
        #     particleState,
        #     operationProperties=OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.KernelMeanSymmetric,
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = P_initial
        # )
        # print(f'Interpolated initial pressure: {P.min().item():6.4g} to {P.max().item():6.4g}')
        # u = 1 / (gamma - 1) * (P / eosRho)
        # particleState.densities = eosRho
        # u = warpOperation(
        #     particleState,
        #     operationProperties=OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather,
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = u
        # )

        print(f'Initial internal energy: {u.min().item():6.4g} to {u.max().item():6.4g}')

        # particleState.densities = eosRho
        
        dx = particles_l.positions[1,0] - particles_l.positions[0,0]
        dx = 0.5 * dx
        # x = torch.where(particleState.positions[:,0] > 0., particleState.positions[:,0] - 0.5, particleState.positions[:,0] + 0.5)
        # ramp = torch.exp(x/dx) / (1 + torch.exp(x/dx))
        # ramp =  / (torch.exp(x/dx) + 1)
        ramped = lambda a, b, x: (a - b) / (torch.exp(x/dx) + 1) + b

        # u_max = u.max()
        # u_min = u.min()
        # u[mask] = u_min * (1 - ratio[mask]) + u_max * ratio[mask] if u_max < u_min else u_min * (1 - ratio[mask]) + u_max * ratio[mask]

        left_p = leftState.p
        right_p = rightState.p

        left_u = 1 / (gamma - 1) * (leftState.p / leftState.rho)
        right_u = 1 / (gamma - 1) * (rightState.p / rightState.rho)

        x = particleState.positions[:,0]
        u = torch.where(particleState.positions[:,0] > 0., ramped(left_u, right_u, x - 0.5), u)
        u = torch.where(particleState.positions[:,0] < 0., ramped(right_u, left_u, x + 0.5), u).to(particleState.densities.dtype)
        p = torch.where(particleState.positions[:,0] > 0., ramped(left_p, right_p, x-0.5), P_initial)

        # left_A = leftState.p / leftState.rho**gamma
        # right_A = rightState.p / rightState.rho**gamma
        # A = torch.where(particleState.positions[:,0] > 0., ramped(left_A, right_A, x - 0.5), left_A)
        # A = torch.where(particleState.positions[:,0] < 0., ramped(right_A, left_A, x + 0.5), A)

        # u = A * particleState.densities**(gamma - 1) / (gamma - 1)

        # x = torch.where(particleState.positions[:,0] < 0., particleState.positions[:,0] + 0.5, particleState.positions[:,0] - 0.5)
        # u = torch.where(particleState.positions[:,0] < 0., ramped(right_u, left_u, x), u)
        p = torch.where(particleState.positions[:,0] < 0., ramped(right_p, left_p, x+0.5), p).to(particleState.densities.dtype)

        # u = 1 / (gamma - 1) * (p / eosRho)
        

    particleState.densities = eosRho
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = eosRho, gamma = gamma)

    internalEnergy = u_
    kineticEnergy = torch.linalg.norm(particleState.velocities, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particleState.masses

    particleState.internalEnergies = internalEnergy
    particleState.totalEnergies = totalEnergy
    particleState.pressures = P_
    particleState.soundspeeds = c_s
    particleState.entropies = A_

    adjacency = buildVerletList(particleState, 
                                domain = config.domain,
                                verletScale = 2**(1/config.dim), supportMode = config.supportMode)

    compressibleSystem = SimulationSystem(
        state=particleState,
        adjacency = adjacency,
        domain = config.domain)
    compParams.gamma = gamma

    dx = particleState.masses.min()
    config.dx = dx
    config.dt = computeTimestep(compressibleSystem, config, compParams, dt = None)
    return compressibleSystem


def buildSod1DMeshless(
    SimulationSystem,
    samplingRatio: int,
    leftState: sodInitialState,
    rightState: sodInitialState,
    gamma: float,
    config: SimulationConfig,
    schemeConfig,
):
    """The meshless (MFM / MFV) form of `buildSod1D` (COMPRESSIBLE_SPH_PLAN E2):
    the same double-Sod lattice, but the state is the conserved Q (the G1
    pattern, `caseUtils/meshless.py`) and the dt is the core signal timestep
    (`modules/timestep/meshless.py`) instead of the acoustic CFL one."""
    from warpSPH.modules.timestep import computeTimestep
    from ...meshless import buildMeshlessState

    nx = config.nx
    actualRatio = nx / (nx // samplingRatio)

    particles_l = sampleRegularParticles(nx, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)
    particles_r = sampleRegularParticles(nx // samplingRatio, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)
    particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min())
    pos_r = particles_r.positions
    pos_r[pos_r[:, 0] < 0, 0] -= 0.5
    pos_r[pos_r[:, 0] > 0, 0] += 0.5
    particles_r = particles_r._replace(positions = pos_r)

    combinedPositions = torch.cat([particles_l.positions, particles_r.positions], dim = 0)
    tags = torch.cat([
        torch.zeros(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.ones(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)

    leftMass = particles_l.masses.min() * leftState.rho
    rightMass = particles_l.masses.min() * rightState.rho * actualRatio

    combinedMasses = torch.where(tags == 0, leftMass, rightMass).to(combinedPositions.dtype)
    combinedSupports = torch.where(tags[:, None] == 0, particles_l.supports.min(), particles_r.supports.min())[:, 0]
    combinedDensities = torch.where(tags == 0, leftState.rho, rightState.rho).to(combinedPositions.dtype)
    combinedVelocities = torch.where(tags[:, None] == 0, float(leftState.v), float(rightState.v)).to(combinedPositions.dtype)
    combinedPressures = torch.where(tags == 0, float(leftState.p), float(rightState.p)).to(combinedPositions.dtype)
    combinedKinds = torch.zeros_like(tags)
    combinedMaterials = tags
    combinedUIDs = torch.cat([
        torch.arange(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.arange(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)

    particleState = buildMeshlessState(
        combinedPositions, combinedSupports, combinedMasses,
        combinedDensities, combinedVelocities, combinedPressures,
        gamma, combinedKinds, combinedMaterials, combinedUIDs)

    meshlessSystem = SimulationSystem(state=particleState, domain=config.domain)
    config.dt = computeTimestep(meshlessSystem, config, schemeConfig, dt=None)
    return meshlessSystem