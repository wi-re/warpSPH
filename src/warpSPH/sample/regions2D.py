"""2D regular-lattice sampler split into up to four axis-aligned regions by
one or two split lines, used to give each region a different resolution
(oversampled regions all get `masses.min()`, then rescaled by `rho0`, so
mass is uniform across the merged set even though particle spacing isn't).
`sampleRegionSystem` is the entry point -- it merges the regions, computes a
uniform density field, relaxes supports/density via `evaluateOptimalSupport`,
and derives pressure/energy through `idealGasEOS` for an at-rest gas; it is
used by `caseUtils.compressible.triplePoint.equalMass` for the triple-point
problem's per-region resolution. The `splitDomain`/`maskParticles`/
`sampleRegions`/`sampleRegionsSymmetric`/`mergeParticles`/`computeNs`/
`plotDomain` helpers below are internal to that pipeline (or plotting-only,
for `plotDomain`) and not used outside this file.
"""

from warpSPH.configurations.compute import neighborSearchOptions

import numpy as np
from ..configurations import *
from warpSPH.modules import *
import torch
from warpSPHCore import *
from ..enumTypes import *

__all__ = ['sampleRegionSystem']

def splitDomain(split_x, split_y, domain, nx):
    lx = domain.max[0] - domain.min[0]
    dx = lx / nx
    ly = domain.max[1] - domain.min[1]
    dy = ly / nx

    splitx_ = (split_x - domain.min[0]) / dx
    splity_ = (split_y - domain.min[1]) / dy

    rounded_splitx_ = torch.round(splitx_).long()
    rounded_splity_ = torch.round(splity_).long()

    splitx = rounded_splitx_ * dx + domain.min[0]
    splity = rounded_splity_ * dy + domain.min[1]

    return splitx.cpu().item(), splity.cpu().item()


def maskParticles(particles, split_x, split_y, domain, nx):
    device = particles.positions.device
    if isinstance(split_x, list):
        splitXLower, splitYLower = splitDomain(split_x[0], split_y[0], domain, nx)
        splitXUpper, splitYUpper = splitDomain(split_x[1], split_y[1], domain, nx)

        x = particles.positions[:, 0]
        y = particles.positions[:, 1]

        A_1 = (x < splitXLower) & (y < splitYLower)
        A_2 = (x >= splitXUpper) & (y < splitYLower)
        A_3 = (x < splitXLower) & (y >= splitYUpper)
        A_4 = (x >= splitXUpper) & (y >= splitYUpper)
        maskA = torch.logical_or(torch.logical_or(A_1, A_2), torch.logical_or(A_3, A_4))

        # B/C split the strip between the x edges (splitXLower <= x < splitXUpper)
        # by |y|: B is the outer band (y past splitYLower or before splitYUpper,
        # i.e. the union of the top and bottom strips), C is the inner band
        # between them. The previous version ANDed per-corner x/y terms taken
        # from the A pattern, which made B's condition self-contradictory
        # (never true) and left B's particles falling through to C's slot.
        midX = (x >= splitXLower) & (x < splitXUpper)
        maskB = midX & torch.logical_or(y >= splitYLower, y < splitYUpper)
        maskC = midX & (y < splitYLower) & (y >= splitYUpper)

        mask = torch.ones_like(particles.positions[:, 0], dtype=torch.int64, device=device)*3
        mask[maskA] = 0
        mask[maskB] = 1
        mask[maskC] = 2

        return mask, [splitXLower, splitXUpper], [splitYLower, splitYUpper]

    else:

        splitx, splity = splitDomain(split_x, split_y, domain, nx)

        maskA = (particles.positions[:, 0] < splitx) & (particles.positions[:, 1] < splity)
        maskB = (particles.positions[:, 0] >= splitx) & (particles.positions[:, 1] < splity)
        maskC = (particles.positions[:, 0] < splitx) & (particles.positions[:, 1] >= splity)
        maskD = (particles.positions[:, 0] >= splitx) & (particles.positions[:, 1] >= splity)

        mask = torch.zeros_like(particles.positions[:, 0], dtype=torch.int64, device=device)
        mask[maskA] = 0
        mask[maskB] = 1
        mask[maskC] = 2
        mask[maskD] = 3

        return mask, splitx, splity

def maskParticlesSymmetric(particles, split_x, split_y, domain, nx):
    return maskParticles(particles, [split_x, domain.max[0] - split_x], [split_y, domain.max[1] - split_y], domain, nx)


from ..geometry import ParticleSet
from .regular import sampleRegularParticles

def sampleRegions(domain, nx, targetNeighbors, splitX, splitY, jitter = 0.0):
    print(f'Sampling regions with nx = {nx}, splitX = {splitX}, splitY = {splitY}, jitter = {jitter}')
    if isinstance(nx, int):
        particles = sampleRegularParticles(nx, domain, targetNeighbors, jitter = jitter)

        # split_x, split_y = splitDomain(splitX, splitY, domain, nx)
        mask, split_x, split_y = maskParticles(particles, splitX, splitY, domain, nx)

        particlesA = ParticleSet(
            positions = particles.positions[mask == 0],
            supports = particles.supports[mask == 0],
            masses = particles.masses[mask == 0],
            densities = particles.densities[mask == 0]
        ) 
        particlesB = ParticleSet(
            positions = particles.positions[mask == 1],
            supports = particles.supports[mask == 1],
            masses = particles.masses[mask == 1],
            densities = particles.densities[mask == 1]
        ) 
        particlesC = ParticleSet(
            positions = particles.positions[mask == 2],
            supports = particles.supports[mask == 2],
            masses = particles.masses[mask == 2],
            densities = particles.densities[mask == 2]
        ) 
        particlesD = ParticleSet(
            positions = particles.positions[mask == 3],
            supports = particles.supports[mask == 3],
            masses = particles.masses[mask == 3],
            densities = particles.densities[mask == 3]
        )

        return particlesA, particlesB, particlesC, particlesD, split_x, split_y
    else:
        particlesA = sampleRegularParticles(nx[0], domain, targetNeighbors, jitter = jitter)
        particlesB = sampleRegularParticles(nx[1], domain, targetNeighbors, jitter = jitter)
        particlesC = sampleRegularParticles(nx[2], domain, targetNeighbors, jitter = jitter)
        particlesD = sampleRegularParticles(nx[3], domain, targetNeighbors, jitter = jitter)

        # splitx, splity = splitDomain(splitX, splitY, domain, np.min(nx))

        maskA, splitx, splity = maskParticles(particlesA, splitX, splitY, domain, np.min(nx))
        maskB, *_ = maskParticles(particlesB, splitX, splitY, domain, np.min(nx))
        maskC, *_ = maskParticles(particlesC, splitX, splitY, domain, np.min(nx))
        maskD, *_ = maskParticles(particlesD, splitX, splitY, domain, np.min(nx))

        particlesA = ParticleSet(
            positions = particlesA.positions[maskA == 0],
            supports = particlesA.supports[maskA == 0],
            masses = particlesA.masses[maskA == 0],
            densities = particlesA.densities[maskA == 0]
        ) 
        particlesB = ParticleSet(
            positions = particlesB.positions[maskB == 1],
            supports = particlesB.supports[maskB == 1],
            masses = particlesB.masses[maskB == 1],
            densities = particlesB.densities[maskB == 1]
        ) 
        particlesC = ParticleSet(
            positions = particlesC.positions[maskC == 2],
            supports = particlesC.supports[maskC == 2],
            masses = particlesC.masses[maskC == 2],
            densities = particlesC.densities[maskC == 2]
        )
        particlesD = ParticleSet(
            positions = particlesD.positions[maskD == 3],
            supports = particlesD.supports[maskD == 3],
            masses = particlesD.masses[maskD == 3],
            densities = particlesD.densities[maskD == 3]
        )

        return particlesA, particlesB, particlesC, particlesD, splitx, splity
    
def sampleRegionsSymmetric(domain, nx, targetNeighbors, splitX, splitY, jitter = 0.0):
    lx = domain.max[0].item() - domain.min[0].item()
    ly = domain.max[1].item() - domain.min[1].item()

    minX = domain.min[0].item()
    minY = domain.min[1].item()
    maxX = domain.max[0].item()
    maxY = domain.max[1].item()

    splitXUpper = minX + (maxX - minX) * (1 - (splitX - minX) / (maxX - minX))
    splitYUpper = minY + (maxY - minY) * (1 - (splitY - minY) / (maxY - minY))

    return sampleRegions(domain, nx, targetNeighbors, [splitX, splitXUpper], [splitY, splitYUpper], jitter = jitter)


def plotDomain(axis, domain):
    axis.set_xlim(domain.min[0].cpu(), domain.max[0].cpu())
    axis.set_ylim(domain.min[1].cpu(), domain.max[1].cpu())
    axis.set_aspect('equal')

    axis.plot([domain.min[0].cpu(), domain.max[0].cpu()], [domain.min[1].cpu(), domain.min[1].cpu()], 'k-')
    axis.plot([domain.min[0].cpu(), domain.max[0].cpu()], [domain.max[1].cpu(), domain.max[1].cpu()], 'k-')
    axis.plot([domain.min[0].cpu(), domain.min[0].cpu()], [domain.min[1].cpu(), domain.max[1].cpu()], 'k-')
    axis.plot([domain.max[0].cpu(), domain.max[0].cpu()], [domain.min[1].cpu(), domain.max[1].cpu()], 'k-')

def mergeParticles(particles_list):
    positions = torch.cat([p.positions for p in particles_list], dim=0)
    supports = torch.cat([p.supports for p in particles_list], dim=0)
    masses = torch.cat([p.masses for p in particles_list], dim=0)
    densities = torch.cat([p.densities for p in particles_list], dim=0)
    index = torch.cat([torch.ones_like(p.masses) * i for i, p in enumerate(particles_list)], dim=0)

    return ParticleSet(
        positions = positions,
        supports = supports,
        masses = masses,
        densities = densities
    ), index


def computeNs(domain, nx, band, shortEdge = True):    
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        # x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nx + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        # spaces.append(x)
        dxs.append(dx)

    spaces = []
    if shortEdge:
        dx = torch.min(torch.tensor(dxs))
    else:
        dx = torch.max(torch.tensor(dxs))
    # print(dxs, dx, nx)
    ns = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        nd = (torch.ceil(l/dx)).to(torch.int32)
        dn = l / (nd if periodicity[d] else nd-1)
        offset = dx/2 if periodicity[d] else 0
        offset -= dx * band

        # print(f'Dimension {d}: l: {l}, dx: {dx}, nd: {nd}, dn: {dn}, offset: {offset}')
        x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nd + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        spaces.append(x)
        ns.append(nd + band * 2)
    
    return ns



def sampleRegionSystem(
    nxs: list[int],
    splitLineX: float,
    splitLineY: float,
    config: SimulationConfig,
    schemeConfig: CompressibleSPHConfig,
    SimulationState: type,
    SimulationSystem: type,
):
    device = config.device
    # splitLineX = 1
    # splitLineY = -1.5
    particlesA, particlesB, particlesC, particlesD, splitx, splity = sampleRegionsSymmetric(config.domain, nxs, config.targetNeighbors, splitLineX, splitLineY)


    # config.domain.min[0] = -L/2/aspect
    # config.domain.max[0] = L/2/aspect
    # config.domain.min[1] = 0 - band * dx
    # config.domain.max[1] = L + band * dx
    # particles_ = sampleRegularParticles(nx, config.domain, config.targetNeighbors)
    particles_, setIndex = mergeParticles([particlesA, particlesB, particlesC, particlesD])
    print(f"Sampled {particles_.positions.shape[0]} particles.")

    particles_.masses[:] = particles_.masses.min() # we oversample the density regions with rho = 1 so we need to use the min here
    # domain = buildDomainDescription(domainExtent * 1.5, dim, periodic = periodicDomain, device = device, dtype = dtype)
    particles_ = particles_._replace(masses = particles_.masses * schemeConfig.rho0)


    print(f"Sampled {particles_.positions.shape[0]} particles. From {particles_.positions.min(dim=0).values} to {particles_.positions.max(dim=0).values}")
    print(f' Support: Min: {particles_.supports.min().item()}, Max: {particles_.supports.max().item()}, Mean: {particles_.supports.mean().item()}')

    particles = SimulationState(
        positions = particles_.positions,
        supports = particles_.supports,
        masses = particles_.masses,
        densities = particles_.densities,
        velocities = torch.zeros_like(particles_.positions),
        
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

    schemeConfigAdaptiveH = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.Owen,
    )


    # neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    # numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    # config = {'targetNeighbors': targetNeighbors, 'domain': domain, 'support': {'iterations': 16, 'scheme': 'Monaghan'}, 'neighborhood': {'algorithm': 'compact'}}
    # rho, h, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particles, config, supportScheme = SupportScheme.Gather, compParams = schemeConfigAdaptiveH)
    # particleState.supports = h_optimal

    particles.densities = rho_optimal
    particles.supports = h_optimal

    print(f' Support After Optimization: Min: {particles.supports.min().item()}, Max: {particles.supports.max().item()}, Mean: {particles.supports.mean().item()}')

    P_initial = torch.zeros_like(particles.densities)
    u = 1 / (schemeConfig.gamma - 1) * (P_initial / rho_optimal)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = rho_optimal, gamma = schemeConfig.gamma)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(torch.zeros_like(particles.positions), dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles.masses

    simulationState_ = SimulationState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities,        
        velocities = torch.zeros_like(particles.positions),

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
        


    adjacency = buildVerletList(
        simulationState_,
        domain=config.domain,
        verletScale=2 ** (1 / config.dim),
        supportMode=config.supportMode,
        **neighborSearchOptions(config))

    compressibleSystem = SimulationSystem(
        state=simulationState_, 
        adjacency = adjacency, 
        domain = config.domain)
    
    config.dx = simulationState_.masses.min() ** (1.0 / config.dim)

    return compressibleSystem
