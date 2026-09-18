"""Dispatch initial particle sampling by `config.samplingScheme`
(`geometry.SamplingScheme`), used only by `warpSPH.systems.waveSystem`'s
wave-equation setup (not by any compressible/weakly-compressible case).

`regular`/`jittered`/`random` all start from `sampleRegularParticles` and
differ only in how much position noise is layered on afterwards; `optimal`
relaxes a lattice via `sampleOptimal`; `densest` lays out the densest
periodic packing (1D uniform, 2D hexagonal, 3D FCC) via
`sampleDensestParticles`; `glass` loads a pre-relaxed particle
configuration from a local `position_samples_*.h5` file sized to the
requested particle count. `config.samplingScheme` defaults to
`SamplingScheme.regular` and no case in this repo overrides it, so the other
branches are wired but currently unexercised.
"""

from ..configurations import SimulationConfig
from ..geometry import *
from .regular import sampleRegularParticles
from .optimal import sampleOptimal
from .densest import sampleDensestParticles
from warpSPHCore import KernelFunctions
import h5py
import torch

__all__ = ['sampleParticles']

def sampleParticles(nx: int,config : SimulationConfig):
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # dtype = torch.float32
    # L = 2
    # dim = 2

    # kernel = KernelType.Wendland4
    # targetNeighbors = n_h_to_nH(4, dim)

    # domain = buildDomainDescription(L, dim, True, device, dtype)
    
    particles = sampleRegularParticles(
                nx = nx,
                targetNeighbors=config.targetNeighbors,
                domain=config.domain,
            ) 

    # config = {
    #     'domain': domain,
    #     # 'kernel': kernel,
    #     'targetNeighbors': targetNeighbors,
    #     'neighborhood':{
    #         'verletScale': 1.0,
    #         'computeDkDh': True
    #     }
    # }

    # config['gradientMode'] = GradientMode.Difference
    # config['laplacianMode'] = LaplacianMode.Brookshaw
    # config['supportScheme'] = SupportScheme.Gather
    # config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    
    L = config.domain.max[0] - config.domain.min[0]
    if config.samplingScheme == SamplingScheme.regular:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=config.targetNeighbors,
            domain=config.domain,
        )
    elif config.samplingScheme == SamplingScheme.jittered:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=config.targetNeighbors,
            domain=config.domain,
            jitter=0
        )
        dx = 2 * L / nx
        jitterAmount = dx * 0.25
        particles = particles._replace(
            positions = particles.positions + (torch.rand_like(particles.positions) - 0.5) * jitterAmount
        )
        
    elif config.samplingScheme == SamplingScheme.random:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=config.targetNeighbors,
            domain=config.domain,
            jitter=0.5
        )
        particles = particles._replace(
            positions = torch.rand_like(particles.positions) * L * 2 - L
        )
        
    elif config.samplingScheme == SamplingScheme.optimal:
        particles = sampleOptimal(
            nx = nx,
            domain=config.domain,
            targetNeighbors=config.targetNeighbors,
            kernel=KernelFunctions.Wendland4,
            jitter = 0.5,
            shiftScheme='delta',
            shiftIters=128
        )
    elif config.samplingScheme == SamplingScheme.densest:
        particles = sampleDensestParticles(
            nx = nx,
            domain=config.domain,
            targetNeighbors=config.targetNeighbors,
        )
    elif config.samplingScheme == SamplingScheme.glass:
        files = ['position_samples_1024.h5', 'position_samples_4096.h5', 'position_samples_16384.h5', 'position_samples_65536.h5']
        numParticles = nx ** config.dim
        selectedFile = None
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) >= numParticles:
                selectedFile = file
                break
        if selectedFile is None:
            raise ValueError('No suitable glass file found for the given number of particles.')
        data = None
        with h5py.File(selectedFile, 'r') as f:
            data = f['positions']
            numSamples = data.shape[0]
            randomIndex = torch.randint(0, numSamples, (1,)).item()
            positions = torch.tensor(data[randomIndex], device=config.device, dtype=config.dtype)
            densities = f['densities'][randomIndex]
            supports = f['supports'][randomIndex]
            numNeighbors = f['numNeighbors'][randomIndex]
            counter = f['counter'][randomIndex]
            
            particles = ParticleSet(
                positions=positions,
                supports=torch.tensor(supports, device=config.device, dtype=config.dtype),
                masses=torch.ones_like(positions[:,0], device=config.device, dtype=config.dtype) * (L**config.dim / positions.shape[0]),
                densities=torch.tensor(densities, device=config.device, dtype=config.dtype),
            )
        # particles#, torch.tensor(numNeighbors, device=config.device, dtype=config.dtype), torch.tensor(counter, device=config.device, dtype=config.dtype)

    # particleState = BasicState(particles.positions, particles.supports, particles.masses, particles.densities, torch.zeros_like(particles.positions), torch.zeros(particles.positions.shape[0], device = config.device, dtype = torch.int64), torch.zeros(particles.positions.shape[0], device = config.device, dtype = torch.int64), torch.arange(particles.positions.shape[0], device = config.device), particles.positions.shape[0])
    # neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    # particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries
    # particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    # neighs = neighbors.get('noghost')
    # nnx = 63
    # ddx = 2 / (nnx)
    # hij = (particleState.supports[neighs[0].row] + particleState.supports[neighs[0].col]) / 2

    # positions = neighs[1].x_ij / hij.view(-1,1)
    # index = ((positions + 1) / ddx).to(torch.int64)
    # linIdx = index[:,0] * nnx + index[:,1]
    # counter = scatter_sum(torch.ones_like(linIdx), dim = 0, index = linIdx, dim_size = nnx**2).reshape(nnx,nnx).cpu()#.numpy()
        
    # particles = particles._replace(densities = particleState.densities)    
    
    particles = ParticleSet(
        positions = particles.positions.to(config.device, config.dtype),
        supports = particles.supports.to(config.device, config.dtype),
        masses = particles.masses.to(config.device, config.dtype),
        densities = particles.densities.to(config.device, config.dtype)
    )


    return particles #, particleState.numNeighbors, counter


