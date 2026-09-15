import sys, copy, math, json
sys.path.insert(0, '/home/lu26029/dev/diffSPH/examples/weaklyCompressible')
import torch

from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.integration import getIntegrator
from diffSPH.util import volumeToSupport
from diffSPH.boundary import sampleDomainSDF
from diffSPH.kernels import Kernel_Scale
from diffSPH.sdf import getSDF, sdfFunctions, operatorDict, sampleSDF
from diffSPH.regions import buildRegion, filterRegion
from diffSPH.modules.timestep import computeTimestep
from diffSPH.schemes.initializers import initializeSimulation, updateBodyParticles
from diffSPH.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *

simulationName = 'Dam Break'
L = 2
nx = 64
dx = L / nx
targetDt = 0.0005
rho0 = 1
freeSurface = True
band = 7
fps = 50
timeLimit = 4

kernel = KernelType.Wendland4
scheme = SimulationScheme.DeltaSPH
integrationScheme = IntegrationSchemeType.symplecticEuler

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
targetNeighbors = n_h_to_nH(4, 2)
c_s = 0.3 * volumeToSupport(dx**2, targetNeighbors, 2) / Kernel_Scale(kernel, 2) / targetDt

dim = 2
CFL = 0.3

domain = buildDomainDescription(l = L + dx * (band) * 2, dim = dim, periodic = False, device = device, dtype = dtype)
domain.min = torch.tensor([-L, -L/2], device = device, dtype = dtype)
domain.max = torch.tensor([L, L/2], device = device, dtype = dtype)

interiorDomain = copy.deepcopy(domain)

domain.min -= dx * band
domain.max += dx * band

wrappedKernel = kernel

simulator, SimulationSystem, config, integrator = getSimulationScheme(
     scheme, kernel, integrationScheme,
     1.0, targetNeighbors, domain)
integrationScheme = getIntegrator(integrationScheme)

config['particle'] = {
    'nx': nx + 2 * band,
    'dx': L/nx,
    'targetNeighbors': targetNeighbors,
    'band': band
}
config['fluid'] = {
    'rho0': rho0,
    'c_s': c_s
}
config['surfaceDetection']['active'] = freeSurface
config['shifting']['freeSurface'] = freeSurface
config['pressure']['term'] = 'Antuono'

config['gravity'] = {
    'active': True,
    'magnitude': 10,
    'mode': 'directional',
    'direction': torch.tensor([0, -1], device = device, dtype = dtype)
}

fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)

W = L*5/6
H = L/2
box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([W/2,H/2]).to(points.device)), torch.tensor([interiorDomain.min[0]+W/2,interiorDomain.min[1] + H/2]).to(points.device)), invert = False)
config['particle']['shortEdge'] = True

regions = []
regions.append(buildRegion(sdf = domain_sdf, config = config, type = 'boundary', kind = 'constant'))
regions.append(buildRegion(sdf = box_sdf, config = config, type = 'fluid'))

for region in regions:
    region = filterRegion(region, regions)

particleState, config, rigidBodies = initializeSimulation(scheme, config, regions)
config['shifting']['active'] = True

dt = computeTimestep(scheme, 1e-2, particleState, config, None)

kinds = particleState.kinds.detach().cpu()
pos = particleState.positions.detach().cpu()
dens = particleState.densities.detach().cpu()
mass = particleState.masses.detach().cpu()
supp = particleState.supports.detach().cpu() if getattr(particleState, 'supports', None) is not None else None

out = {}
out['nx'] = nx
out['dx'] = dx
out['band'] = band
out['targetNeighbors'] = float(targetNeighbors)
out['c_s'] = float(c_s)
out['dt_initial'] = float(dt)
out['domain_min'] = domain.min.detach().cpu().tolist()
out['domain_max'] = domain.max.detach().cpu().tolist()
out['interior_min'] = interiorDomain.min.detach().cpu().tolist()
out['interior_max'] = interiorDomain.max.detach().cpu().tolist()
out['W_box'] = W
out['H_box'] = H
out['box_min'] = [float(interiorDomain.min[0]), float(interiorDomain.min[1])]
out['box_max'] = [float(interiorDomain.min[0])+W, float(interiorDomain.min[1])+H]
out['n_total'] = int(kinds.shape[0])
for k in sorted(set(kinds.tolist())):
    mask = kinds == k
    out[f'n_kind{int(k)}'] = int(mask.sum())
fluidMask = kinds == 0
out['fluid_pos_min'] = pos[fluidMask].min(dim=0).values.tolist()
out['fluid_pos_max'] = pos[fluidMask].max(dim=0).values.tolist()
out['fluid_density_min'] = float(dens[fluidMask].min())
out['fluid_density_mean'] = float(dens[fluidMask].mean())
out['fluid_density_max'] = float(dens[fluidMask].max())
out['fluid_mass_mean'] = float(mass[fluidMask].mean())
if supp is not None:
    out['fluid_support_mean'] = float(supp[fluidMask].mean())
out['pressure_term'] = config['pressure']['term']
out['shifting_active'] = config['shifting']['active']
out['shifting_freeSurface'] = config['shifting']['freeSurface']
out['gravity'] = {'magnitude': config['gravity']['magnitude'], 'mode': config['gravity']['mode']}
out['kernel'] = kernel.name
out['integrationScheme'] = 'symplecticEuler'

print(json.dumps(out, indent=2))
with open('/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/diffsph_ic.json', 'w') as f:
    json.dump(out, f, indent=2)
