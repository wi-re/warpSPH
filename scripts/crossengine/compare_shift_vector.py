"""Cross-engine RAW shift-vector comparison, mirroring
cross_engine_step_compare_midtraj.py but for the shift term instead of the
force term: warpSPH's `computeDeltaShift` vs diffSPH's `computeDeltaShifting`
(the core kernel-gradient concentration sum each codebase's `solveShifting`
wrapper calls once per iteration), on the IDENTICAL mid-trajectory fluid-only
state (dropping walls symmetrically, as before).
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
FRAME = 'frame_00200'  # t ~ 0.1s, same snapshot as the earlier force comparison

f = h5py.File(TRAJ, 'r')
t_frame = float(f['times'][FRAME][()][0])
pos_all = f['positions'][FRAME][:]
vel_all = f['velocities'][FRAME][:]
dens_all = f['densities'][FRAME][:]
kinds_all = f['combinedKinds'][:]
mass_all = f['combinedMasses'][:]
supp_all = f['combinedSupports'][:]
f.close()

fluidMask = kinds_all == 0
n = int(fluidMask.sum())
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

pos_np = pos_all[fluidMask]
vel_np = vel_all[fluidMask]
dens_np = dens_all[fluidMask]
mass_np = mass_all[fluidMask]
supp_np = supp_all[fluidMask]
print(f'loaded {FRAME} @ t={t_frame:.4f}, {n} fluid particles')

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
c_s = float(ctx.schemeConfig.fluid.fixedSoundSpeed)

posT = torch.tensor(pos_np, device=device, dtype=dtype)
velT = torch.tensor(vel_np, device=device, dtype=dtype)
densT = torch.tensor(dens_np, device=device, dtype=dtype)
massT = torch.tensor(mass_np, device=device, dtype=dtype)
suppT = torch.tensor(supp_np, device=device, dtype=dtype)

# ---------------------------------------------------------------------------
# warpSPH's own raw computeDeltaShift, called directly (bypassing the
# solveShifting wrapper's free-surface projection/thresholding entirely).
# ---------------------------------------------------------------------------
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState as WState_w
from warpSPHCore import SupportScheme, buildVerletList
from warpSPH.modules.shifting.delta import computeDeltaShift

fluidState_w = WState_w(
    positions=posT.clone(), velocities=velT.clone(), supports=suppT.clone(),
    masses=massT.clone(), densities=densT.clone(),
    kinds=torch.zeros(n, dtype=torch.int32, device=device),
    materials=torch.zeros(n, dtype=torch.int32, device=device),
    UIDs=torch.arange(n, dtype=torch.int32, device=device),
    UIDcounter=n, pressures=None, soundspeeds=None,
)
adjacency_w = buildVerletList(fluidState_w, ctx.config.domain, verletScale=ctx.config.verletScale,
                              supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)

update_w, _adj = computeDeltaShift(fluidState_w, ctx.config, ctx.schemeConfig, ctx.config.domain,
                                   adjacency_w, iters=1)
update_w = update_w.detach().cpu().numpy()
print('warpSPH raw shift: mean|.|', np.linalg.norm(update_w, axis=1).mean(),
     'max|.|', np.linalg.norm(update_w, axis=1).max())

# ---------------------------------------------------------------------------
# diffSPH's own raw computeDeltaShifting, called directly.
# ---------------------------------------------------------------------------
from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d
from diffSPH.neighborhood import evaluateNeighborhood, SupportScheme as DSupportScheme
from diffSPH.modules.shifting.deltaPlusShifting import computeDeltaShifting

kernelD = KernelType.Wendland4
schemeD = SimulationScheme.DeltaSPH
integrationD = IntegrationSchemeType.symplecticEuler

domMin = ctx.config.domain.min.detach().cpu().tolist()
domMax = ctx.config.domain.max.detach().cpu().tolist()
domainD = buildDomainDescription(float(domMax[0] - domMin[0]), 2, False, device, dtype)
domainD.min = torch.tensor(domMin, device=device, dtype=dtype)
domainD.max = torch.tensor(domMax, device=device, dtype=dtype)

targetNeighborsD = n_h_to_nH(4, 2)
_, _, configD, _ = getSimulationScheme(schemeD, kernelD, integrationD, 1.0, targetNeighborsD, domainD)
configD['particle'] = {'nx': ctx.spec.nx, 'dx': ctx.config.dx, 'targetNeighbors': targetNeighborsD,
                       'band': 7, 'support': float(suppT.mean())}
configD['fluid'] = {'rho0': 1.0, 'c_s': c_s}
configD['shifting']['CFL'] = float(ctx.schemeConfig.shiftProperties.CFL)
configD['shifting']['computeMach'] = bool(ctx.schemeConfig.shiftProperties.computeMach)

stateD = WState_d(
    positions=posT.clone(), supports=suppT.clone(), masses=massT.clone(),
    densities=densT.clone(), velocities=velT.clone(),
    pressures=torch.zeros(n, device=device, dtype=dtype),
    soundspeeds=torch.full((n,), c_s, device=device, dtype=dtype),
    kinds=torch.zeros(n, device=device, dtype=torch.int64),
    materials=torch.zeros(n, device=device, dtype=torch.int64),
    UIDs=torch.arange(n, device=device, dtype=torch.int64), UIDcounter=n,
)
neighborhoodInfo, neighbors = evaluateNeighborhood(
    stateD, domainD, kernelD, verletScale=configD['neighborhood']['verletScale'],
    mode=DSupportScheme.SuperSymmetric, priorNeighborhood=None,
    computeHessian=configD['neighborhood']['computeHessian'],
    computeDkDh=configD['neighborhood']['computeDkDh'])
noghost = neighbors.get('noghost')[0]
raw_d = computeDeltaShifting(stateD, domainD, kernelD, noghost, configD)
# Net-position-additive convention check: diffSPH's own caller does
# `update = -computeDeltaShifting(...)` then `positions -= update`, i.e. the
# actual net effect on positions is `positions += raw_d` (the two negations
# cancel) -- the same "gets added directly" convention warpSPH's `shift`
# uses (`positions += shift` in both `delta.py` and `wrapper.py`). So the
# apples-to-apples comparable quantity is `raw_d` itself, NOT `-raw_d`.
update_d = raw_d.detach().cpu().numpy()
update_d_negated = (-raw_d).detach().cpu().numpy()  # kept for reference/sanity
print('diffSPH raw shift: mean|.|', np.linalg.norm(update_d, axis=1).mean(),
     'max|.|', np.linalg.norm(update_d, axis=1).max())

# ---------------------------------------------------------------------------
# Compare.
# ---------------------------------------------------------------------------
diff = update_w - update_d
diffMag = np.linalg.norm(diff, axis=1)
wMag = np.linalg.norm(update_w, axis=1)
dMag = np.linalg.norm(update_d, axis=1)
cos = np.sum(update_w * update_d, axis=1) / (wMag * dMag + 1e-30)

out = dict(
    t=t_frame, n=n,
    warp_mean_mag=float(wMag.mean()), warp_max_mag=float(wMag.max()),
    diff_mean_mag=float(dMag.mean()), diff_max_mag=float(dMag.max()),
    ratio_mean_mag=float((wMag.mean() / dMag.mean())) if dMag.mean() > 0 else None,
    diffvec_mean_mag=float(diffMag.mean()), diffvec_max_mag=float(diffMag.max()),
    cosine_mean=float(np.nanmean(cos)), cosine_median=float(np.nanmedian(cos)),
    cosine_p05=float(np.nanpercentile(cos, 5)),
)
print(json.dumps(out, indent=2))
