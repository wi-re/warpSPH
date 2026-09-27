"""Cross-engine single-step force comparison at a MID-TRAJECTORY snapshot
(not the trivial t=0 IC, where pressure/velocity fields are exactly zero
everywhere and both engines trivially agree). Loads a frame from the already
-completed warpSPH diffSPH-matched dambreak run, strips it to fluid-only
(dropping walls/ghosts so both engines see an identical particle set with no
boundary-bookkeeping asymmetry -- symmetric for both sides), and calls each
engine's own raw force function once on that exact snapshot.
"""
import sys
import copy
import json
import numpy as np
import h5py

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

TRAJ = '/home/lu26029/dev/warpSPH/export/dambreak-diffsphMatch_2026-09-14_08-26-40/trajectory.h5'
FRAME = 'frame_00200'  # t ~ 0.1s: gravity has acted, column is compressing/starting to
                        # collapse, well before any wall impact -- non-trivial pressure
                        # + velocity field, but nothing violent yet.

f = h5py.File(TRAJ, 'r')
t_frame = float(f['times'][FRAME][()][0])
pos_all = f['positions'][FRAME][:]
vel_all = f['velocities'][FRAME][:]
dens_all = f['densities'][FRAME][:]
kinds_all = f['combinedKinds'][:]
mass_all = f['combinedMasses'][:]
supp_all = f['combinedSupports'][:]
f.close()
print(f'loaded {FRAME} @ t={t_frame:.4f}, {pos_all.shape[0]} particles total')

fluidMask = kinds_all == 0
n = int(fluidMask.sum())
print(f'fluid particles: {n}')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

pos_np = pos_all[fluidMask]
vel_np = vel_all[fluidMask]
dens_np = dens_all[fluidMask]
mass_np = mass_all[fluidMask]
supp_np = supp_all[fluidMask]
print('velocity magnitude: mean', np.linalg.norm(vel_np, axis=1).mean(),
     'max', np.linalg.norm(vel_np, axis=1).max())
print('density: min', dens_np.min(), 'mean', dens_np.mean(), 'max', dens_np.max())

# ---------------------------------------------------------------------------
# Rebuild the matched-config context (kernel/gravity/c0/schemeConfig) exactly
# as before, just to get ctx.config / ctx.schemeConfig / ctx.config.domain and
# dt -- cheap (nSteps=0), no need to re-derive these by hand.
# ---------------------------------------------------------------------------
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import caseMain

args = [
    '--nx', '64', '--L', '2.0', '--n_h', '4.0',
    '--kernel', 'Wendland4', '--integrationScheme', 'symplecticEuler',
    '--supportMode', 'SuperSymmetric', '--tLimit', '4.0', '--targetDt', '0.0005',
    '--band', '7', '--fillRatio', '0.5', '--fluidWidth', str(5.0 / 12.0),
    '--gravityMagnitude', '10.0', '--wallBC', 'constant',
    '--nSteps', '0', '--no-plot', '--no-store', '--no-video', '--quiet',
    '--precision', 'float32',
]
result = caseMain(dambreakCase, args)
ctx = result.ctx
dt = float(ctx.config.dt)

posT = torch.tensor(pos_np, device=device, dtype=dtype)
velT = torch.tensor(vel_np, device=device, dtype=dtype)
densT = torch.tensor(dens_np, device=device, dtype=dtype)
massT = torch.tensor(mass_np, device=device, dtype=dtype)
suppT = torch.tensor(supp_np, device=device, dtype=dtype)

from warpSPH.systems.weaklyCompressible import (WeaklyCompressibleState as WState_w,
                                                WeaklyCompressibleSystem as WSystem_w)

fluidState_w = WState_w(
    positions=posT.clone(), velocities=velT.clone(), supports=suppT.clone(),
    masses=massT.clone(), densities=densT.clone(),
    kinds=torch.zeros(n, dtype=torch.int32, device=device),
    materials=torch.zeros(n, dtype=torch.int32, device=device),
    UIDs=torch.arange(n, dtype=torch.int32, device=device),
    UIDcounter=n,
    pressures=None, soundspeeds=None,
)
fluidSystem_w = WSystem_w(state=fluidState_w, adjacency=None, domain=ctx.config.domain, t=t_frame)

schemeConfig_w = copy.deepcopy(ctx.schemeConfig)
schemeConfig_w.regions = []
schemeConfig_w.boundaryConditions = []

from warpSPH.schemes.deltaSPH import deltaSPH_step

update_w, adjacency_w, currentState_w = deltaSPH_step(fluidSystem_w, dt, ctx.config, schemeConfig_w)
dvdt_w = update_w.dvdt.detach().cpu().numpy()
drhodt_w = update_w.drhodt.detach().cpu().numpy()

print('warpSPH dvdt: mean', dvdt_w.mean(0), 'max|.|', np.linalg.norm(dvdt_w, axis=1).max())
print('warpSPH drhodt: min/mean/max', drhodt_w.min(), drhodt_w.mean(), drhodt_w.max())

# ---------------------------------------------------------------------------
# diffSPH side, same snapshot.
# ---------------------------------------------------------------------------
from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *
from diffSPH.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d

kernelD = KernelType.Wendland4
schemeD = SimulationScheme.DeltaSPH
integrationD = IntegrationSchemeType.symplecticEuler

domMin = ctx.config.domain.min.detach().cpu().tolist()
domMax = ctx.config.domain.max.detach().cpu().tolist()
domainD = buildDomainDescription(float(domMax[0] - domMin[0]), 2, False, device, dtype)
domainD.min = torch.tensor(domMin, device=device, dtype=dtype)
domainD.max = torch.tensor(domMax, device=device, dtype=dtype)

targetNeighborsD = n_h_to_nH(4, 2)
simulatorD, SystemD, configD, integratorD = getSimulationScheme(
    schemeD, kernelD, integrationD, 1.0, targetNeighborsD, domainD)

c_s = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
configD['particle'] = {'nx': ctx.spec.nx, 'dx': ctx.config.dx,
                       'targetNeighbors': targetNeighborsD, 'band': 7}
configD['fluid'] = {'rho0': 1.0, 'c_s': c_s}
configD['surfaceDetection']['active'] = True
configD['shifting']['freeSurface'] = True
configD['pressure']['term'] = 'Antuono'
configD['gravity'] = {
    'active': True, 'magnitude': 10.0, 'mode': 'directional',
    'direction': torch.tensor([0, -1.0], device=device, dtype=dtype),
}
configD['regions'] = []

kindsD = torch.zeros(n, device=device, dtype=torch.int64)
materialsD = torch.zeros(n, device=device, dtype=torch.int64)
UIDsD = torch.arange(n, device=device, dtype=torch.int64)
soundD = torch.full((n,), c_s, device=device, dtype=dtype)
pressD = torch.zeros(n, device=device, dtype=dtype)

stateD = WState_d(
    positions=posT.clone(), supports=suppT.clone(), masses=massT.clone(),
    densities=densT.clone(), velocities=velT.clone(),
    pressures=pressD, soundspeeds=soundD, kinds=kindsD, materials=materialsD,
    UIDs=UIDsD, UIDcounter=n,
)
systemD = DeltaPlusSPHSystem(systemState=stateD, domain=domainD, neighborhoodInfo=None,
                             t=t_frame, scheme='momentum', rigidBodies=[], regions=[], config=configD)

updateD, particlesD, neighborhoodD = deltaPlusSPHScheme(systemD, dt, configD)
dvdt_d = updateD.velocities.detach().cpu().numpy()
drhodt_d = updateD.densities.detach().cpu().numpy()

print('diffSPH dvdt: mean', dvdt_d.mean(0), 'max|.|', np.linalg.norm(dvdt_d, axis=1).max())
print('diffSPH drhodt: min/mean/max', drhodt_d.min(), drhodt_d.mean(), drhodt_d.max())

# ---------------------------------------------------------------------------
# Compare.
# ---------------------------------------------------------------------------
dv_diff = dvdt_w - dvdt_d
dv_diff_mag = np.linalg.norm(dv_diff, axis=1)
dv_ref_mag = np.linalg.norm(dvdt_w, axis=1)
drho_diff = drhodt_w - drhodt_d

# Distance to nearest domain wall (the interior AABB), to see whether
# disagreement concentrates near the (now wall-less) patch edges.
lo = np.array(domMin) if False else np.array([ctx.scratch['interiorDomain'].min[0].item(),
                                               ctx.scratch['interiorDomain'].min[1].item()])
hi = np.array([ctx.scratch['interiorDomain'].max[0].item(), ctx.scratch['interiorDomain'].max[1].item()])
distToWall = np.minimum(pos_np - lo, hi - pos_np).min(axis=1)
dx = float(ctx.config.dx)
interior = distToWall > 4 * dx  # outside one kernel support of any former wall

out = dict(
    frame=FRAME, t=t_frame, n=n,
    velMagMean=float(np.linalg.norm(vel_np, axis=1).mean()),
    velMagMax=float(np.linalg.norm(vel_np, axis=1).max()),
    densMin=float(dens_np.min()), densMax=float(dens_np.max()),
    dv_diff_mag_mean=float(dv_diff_mag.mean()),
    dv_diff_mag_max=float(dv_diff_mag.max()),
    dv_diff_mag_p50=float(np.median(dv_diff_mag)),
    dv_diff_mag_p95=float(np.percentile(dv_diff_mag, 95)),
    dv_relerr_mean=float((dv_diff_mag / np.clip(dv_ref_mag, 1e-8, None)).mean()),
    dv_relerr_median=float(np.median(dv_diff_mag / np.clip(dv_ref_mag, 1e-8, None))),
    drho_diff_absmean=float(np.abs(drho_diff).mean()),
    drho_diff_absmax=float(np.abs(drho_diff).max()),
    interior_only=dict(
        n=int(interior.sum()),
        dv_diff_mag_mean=float(dv_diff_mag[interior].mean()),
        dv_diff_mag_max=float(dv_diff_mag[interior].max()),
        drho_diff_absmean=float(np.abs(drho_diff[interior]).mean()),
        drho_diff_absmax=float(np.abs(drho_diff[interior]).max()),
    ),
    nearwall_only=dict(
        n=int((~interior).sum()),
        dv_diff_mag_mean=float(dv_diff_mag[~interior].mean()) if (~interior).any() else None,
        dv_diff_mag_max=float(dv_diff_mag[~interior].max()) if (~interior).any() else None,
    ),
)
print(json.dumps(out, indent=2))

np.savez('/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/cross_engine_step_midtraj.npz',
        pos=pos_np, vel=vel_np, dens=dens_np, dvdt_w=dvdt_w, dvdt_d=dvdt_d,
        drhodt_w=drhodt_w, drhodt_d=drhodt_d, interior=interior)
with open('/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/cross_engine_step_midtraj.json', 'w') as fh:
    json.dump(out, fh, indent=2)
