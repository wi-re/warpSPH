"""diffSPH's own dambreak trajectory, faithfully reproducing 15_Dambreak.ipynb
(nx=64, band=7, H=L/2, Wendland4, symplecticEuler, fixed dt computed once at
setup) to t=4, for a rough trace/diagnostics comparison against the matched
warpSPH run. Lightweight: no live matplotlib window, sparse PNG scatter
export + per-step diagnostics only.
"""
import sys, os, copy, math, json, time
sys.path.insert(0, '/home/lu26029/dev/diffSPH/examples/weaklyCompressible')
import numpy as np
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

OUT = '/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/diffsph_traj'
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, 'frames'), exist_ok=True)

L = 2
nx = 64
dx = L / nx
targetDt = 0.0005
rho0 = 1
freeSurface = True
band = 7
timeLimit = 4

kernel = KernelType.Wendland4
scheme = SimulationScheme.DeltaSPH
integrationScheme = IntegrationSchemeType.symplecticEuler

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
targetNeighbors = n_h_to_nH(4, 2)
c_s = 0.3 * volumeToSupport(dx**2, targetNeighbors, 2) / Kernel_Scale(kernel, 2) / targetDt

domain = buildDomainDescription(l=L + dx * band * 2, dim=2, periodic=False, device=device, dtype=dtype)
domain.min = torch.tensor([-L, -L/2], device=device, dtype=dtype)
domain.max = torch.tensor([L, L/2], device=device, dtype=dtype)
interiorDomain = copy.deepcopy(domain)
domain.min -= dx * band
domain.max += dx * band

simulator, SimulationSystem, config, integrator = getSimulationScheme(
    scheme, kernel, integrationScheme, 1.0, targetNeighbors, domain)
integrationScheme = getIntegrator(integrationScheme)

config['particle'] = {'nx': nx + 2 * band, 'dx': L/nx, 'targetNeighbors': targetNeighbors, 'band': band}
config['fluid'] = {'rho0': rho0, 'c_s': c_s}
config['surfaceDetection']['active'] = freeSurface
config['shifting']['freeSurface'] = freeSurface
config['pressure']['term'] = 'Antuono'
config['gravity'] = {'active': True, 'magnitude': 10, 'mode': 'directional',
                     'direction': torch.tensor([0, -1], device=device, dtype=dtype)}

fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert=True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert=False)
W = L * 5 / 6
H = L / 2
box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](
    lambda x: getSDF('box')['function'](x, torch.tensor([W/2, H/2]).to(points.device)),
    torch.tensor([interiorDomain.min[0]+W/2, interiorDomain.min[1]+H/2]).to(points.device)), invert=False)
config['particle']['shortEdge'] = True

regions = [
    buildRegion(sdf=domain_sdf, config=config, type='boundary', kind='constant'),
    buildRegion(sdf=box_sdf, config=config, type='fluid'),
]
for region in regions:
    region = filterRegion(region, regions)

particleState, config, rigidBodies = initializeSimulation(scheme, config, regions)
config['shifting']['active'] = True

dt = computeTimestep(scheme, 1e-2, particleState, config, None)
dt = float(dt)
print(f'fixed dt = {dt:.6g}')

particleSystem = DeltaPlusSPHSystem(config['domain'], None, 0., copy.deepcopy(particleState),
                                    'momentum', None, rigidBodies=config['rigidBodies'],
                                    regions=config['regions'], config=config)
for rigidBody in config['rigidBodies']:
    particleState = updateBodyParticles(scheme, particleState, rigidBody)

timesteps = int(timeLimit / dt)
print(f'{timesteps} steps to t={timeLimit}')

fluidMask0 = (particleSystem.systemState.kinds == 0).detach().cpu().numpy()
rho0v = config['fluid']['rho0']

frameEvery = max(1, timesteps // 200)   # ~200 rough snapshots
diagEvery = 1
traj = []

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0 = time.time()
diverged = False
for i in range(timesteps):
    particleSystem, currentState, updates = integrationScheme.function(
        particleSystem, dt, simulator, config, priorStep=None, verbose=False)

    st = particleSystem.systemState
    vel = st.velocities[fluidMask0]
    dens = st.densities[fluidMask0]
    speed = torch.linalg.norm(vel, dim=-1)
    if i % diagEvery == 0 or i == timesteps - 1:
        row = dict(
            step=i, t=float(particleSystem.t),
            maxVelocity=float(speed.max().detach().cpu()),
            kineticEnergy=float((0.5 * st.masses[fluidMask0] * (speed**2)).sum().detach().cpu()),
            minDensity=float(dens.min().detach().cpu()),
            maxDensity=float(dens.max().detach().cpu()),
        )
        traj.append(row)

    if not torch.isfinite(st.velocities).all():
        print(f'non-finite velocities at step {i}; stopping.')
        diverged = True
        break

    if i % frameEvery == 0 or i == timesteps - 1:
        pos = st.positions.detach().cpu().numpy()
        kinds = st.kinds.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(pos[kinds == 1, 0], pos[kinds == 1, 1], s=1, c='gray')
        sc = ax.scatter(pos[kinds == 0, 0], pos[kinds == 0, 1], s=2,
                        c=np.linalg.norm(st.velocities.detach().cpu().numpy()[kinds == 0], axis=-1),
                        cmap='viridis', vmin=0, vmax=3)
        ax.set_aspect('equal')
        ax.set_xlim(float(domain.min[0]), float(domain.max[0]))
        ax.set_ylim(float(domain.min[1]), float(domain.max[1]))
        ax.set_title(f't={particleSystem.t:.4f}')
        fig.colorbar(sc, ax=ax, label='|v|')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'frames', f'frame_{i:05d}.png'), dpi=90)
        plt.close(fig)

    if i % 200 == 0:
        print(f'step {i}/{timesteps}  t={particleSystem.t:.4f}  '
             f'maxV={traj[-1]["maxVelocity"]:.4f}  '
             f'rho=[{traj[-1]["minDensity"]:.4f},{traj[-1]["maxDensity"]:.4f}]  '
             f'({time.time()-t0:.1f}s elapsed)', flush=True)

print(f'finished in {time.time()-t0:.1f}s, diverged={diverged}, steps run={len(traj)}')
with open(os.path.join(OUT, 'trajectory.json'), 'w') as fh:
    json.dump(dict(dt=dt, timesteps=timesteps, diverged=diverged, traj=traj), fh)
print('saved', os.path.join(OUT, 'trajectory.json'))
