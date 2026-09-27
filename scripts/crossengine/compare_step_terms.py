"""Term-decomposed cross-engine force comparison (DELTASPH_VALIDATION_PLAN
Part 8.8's open question): does the tight AGGREGATE dvdt/drhodt agreement
(compare_step_force.py, <1% relative) hide larger per-TERM disagreements
that partially cancel in the sum?

Technique: monkey-patch each engine's own step function's imported term
functions with transparent "spy" wrappers (call straight through to the
real implementation, stash the result) so the REAL `deltaSPH_step` /
`deltaPlusSPHScheme` runs completely unmodified -- all the internal wiring
(adjacency, mDBC density, EOS, surface detection, gradRho prerequisites)
happens exactly as normal; only the intermediate per-term tensors are
additionally captured.

Run at multiple snapshot times to also check the "does it drift over time"
half of 8.8's open question.
"""
import sys
import json
import numpy as np
import h5py

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

TRAJ = '/home/lu26029/dev/warpSPH/export/dambreak-diffsphMatch_2026-09-14_08-26-40/trajectory.h5'
FRAMES = ['frame_00200', 'frame_01000', 'frame_02000']  # t ~ 0.1, 0.5, 1.0

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

# ---------------------------------------------------------------------------
# Spy-patch warpSPH's deltaSPH_step's own term functions.
# ---------------------------------------------------------------------------
import warpSPH.schemes.deltaSPH as w_deltaSPH_mod

_W_TERM_NAMES = ['computePressureForceSurfaceAware', 'computeVelocityDiffusion',
                 'computeDensityDiffusion', 'computeMomentum', 'computeGravity']
_w_orig = {name: getattr(w_deltaSPH_mod, name) for name in _W_TERM_NAMES}
_w_store = {}


def _makeSpyW(name, orig):
    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        _w_store[name] = r
        return r
    return wrapped


for name in _W_TERM_NAMES:
    setattr(w_deltaSPH_mod, name, _makeSpyW(name, _w_orig[name]))

# ---------------------------------------------------------------------------
# Spy-patch diffSPH's deltaPlusSPHScheme's own term functions. (Importing
# `diffSPH.schemes.deltaSPH` before diffSPH's other modules have loaded hits
# a circular import -- `diffSPH.sampling` first, as every other script here
# already does, pulls the module graph in in the order that avoids it.)
from diffSPH.sampling import buildDomainDescription as _warm_import  # noqa: F401
import diffSPH.schemes.deltaSPH as d_deltaSPH_mod

_D_TERM_NAMES = ['computePressureForce', 'computeViscosity_deltaSPH_inviscid',
                 'computeDensityDeltaTerm', 'computeMomentum', 'computeGravity']
_d_orig = {name: getattr(d_deltaSPH_mod, name) for name in _D_TERM_NAMES}
_d_store = {}


def _makeSpyD(name, orig):
    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        _d_store[name] = r
        return r
    return wrapped


for name in _D_TERM_NAMES:
    setattr(d_deltaSPH_mod, name, _makeSpyD(name, _d_orig[name]))

from warpSPH.systems.weaklyCompressible import (WeaklyCompressibleState as WState_w,
                                                WeaklyCompressibleSystem as WSystem_w)
from warpSPH.schemes.deltaSPH import deltaSPH_step

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
device = ctx.config.domain.min.device
dtype = ctx.config.domain.min.dtype
domainD = buildDomainDescription(float(domMax[0] - domMin[0]), 2, False, device, dtype)
domainD.min = torch.tensor(domMin, device=device, dtype=dtype)
domainD.max = torch.tensor(domMax, device=device, dtype=dtype)

targetNeighborsD = n_h_to_nH(4, 2)
_, _, configD, _ = getSimulationScheme(schemeD, kernelD, integrationD, 1.0, targetNeighborsD, domainD)
configD['particle'] = {'nx': ctx.spec.nx, 'dx': ctx.config.dx, 'targetNeighbors': targetNeighborsD,
                       'band': 7, 'support': None}
configD['fluid'] = {'rho0': 1.0, 'c_s': c_s}
configD['surfaceDetection']['active'] = True
configD['shifting']['freeSurface'] = True
configD['pressure']['term'] = 'Antuono'
configD['gravity'] = {
    'active': True, 'magnitude': 10.0, 'mode': 'directional',
    'direction': torch.tensor([0, -1.0], device=device, dtype=dtype),
}
configD['regions'] = []

f = h5py.File(TRAJ, 'r')
kinds_all = f['combinedKinds'][:]
mass_all = f['combinedMasses'][:]
supp_all = f['combinedSupports'][:]
fluidMask = kinds_all == 0
n = int(fluidMask.sum())
mass_np = mass_all[fluidMask]
supp_np = supp_all[fluidMask]
configD['particle']['support'] = float(supp_np.mean())

allResults = []
for FRAME in FRAMES:
    t_frame = float(f['times'][FRAME][()][0])
    pos_np = f['positions'][FRAME][:][fluidMask]
    vel_np = f['velocities'][FRAME][:][fluidMask]
    dens_np = f['densities'][FRAME][:][fluidMask]

    posT = torch.tensor(pos_np, device=device, dtype=dtype)
    velT = torch.tensor(vel_np, device=device, dtype=dtype)
    densT = torch.tensor(dens_np, device=device, dtype=dtype)
    massT = torch.tensor(mass_np, device=device, dtype=dtype)
    suppT = torch.tensor(supp_np, device=device, dtype=dtype)

    _w_store.clear()
    fluidState_w = WState_w(
        positions=posT.clone(), velocities=velT.clone(), supports=suppT.clone(),
        masses=massT.clone(), densities=densT.clone(),
        kinds=torch.zeros(n, dtype=torch.int32, device=device),
        materials=torch.zeros(n, dtype=torch.int32, device=device),
        UIDs=torch.arange(n, dtype=torch.int32, device=device),
        UIDcounter=n, pressures=None, soundspeeds=None,
    )
    fluidSystem_w = WSystem_w(state=fluidState_w, adjacency=None, domain=ctx.config.domain, t=t_frame)
    import copy as _copy
    schemeConfig_w = _copy.deepcopy(ctx.schemeConfig)
    schemeConfig_w.regions = []
    schemeConfig_w.boundaryConditions = []
    deltaSPH_step(fluidSystem_w, dt, ctx.config, schemeConfig_w)

    _d_store.clear()
    stateD = WState_d(
        positions=posT.clone(), supports=suppT.clone(), masses=massT.clone(),
        densities=densT.clone(), velocities=velT.clone(),
        pressures=torch.zeros(n, device=device, dtype=dtype),
        soundspeeds=torch.full((n,), c_s, device=device, dtype=dtype),
        kinds=torch.zeros(n, device=device, dtype=torch.int64),
        materials=torch.zeros(n, device=device, dtype=torch.int64),
        UIDs=torch.arange(n, device=device, dtype=torch.int64), UIDcounter=n,
    )
    systemD = DeltaPlusSPHSystem(domainD, None, t_frame, stateD, 'momentum', None,
                                 rigidBodies=[], regions=[], config=configD)
    deltaPlusSPHScheme(systemD, dt, configD)

    def mag(t):
        t = t.detach()
        return torch.linalg.norm(t, dim=-1) if t.ndim > 1 else t.abs()

    termPairs = [
        ('pressure', 'computePressureForceSurfaceAware', 'computePressureForce'),
        ('viscosity/dissipation', 'computeVelocityDiffusion', 'computeViscosity_deltaSPH_inviscid'),
        ('density-diffusion (DDT)', 'computeDensityDiffusion', 'computeDensityDeltaTerm'),
        ('continuity (momentum)', 'computeMomentum', 'computeMomentum'),
        ('gravity', 'computeGravity', 'computeGravity'),
    ]
    row = {'frame': FRAME, 't': t_frame}
    for label, wKey, dKey in termPairs:
        if wKey not in _w_store or dKey not in _d_store:
            row[label] = 'MISSING'
            continue
        vW = _w_store[wKey]
        vD = _d_store[dKey]
        if isinstance(vW, tuple):
            vW = vW[0]
        if isinstance(vD, tuple):
            vD = vD[0]
        magW = mag(vW)
        magD = mag(vD)
        diff = (vW - vD).detach()
        diffMag = mag(diff)
        row[label] = dict(
            w_mean=float(magW.mean()), w_max=float(magW.max()),
            d_mean=float(magD.mean()), d_max=float(magD.max()),
            ratio_mean=float(magW.mean() / magD.mean()) if float(magD.mean()) != 0 else None,
            diff_mean=float(diffMag.mean()), diff_max=float(diffMag.max()),
        )
    allResults.append(row)
    print(f'--- {FRAME} (t={t_frame:.4f}) ---')
    print(json.dumps(row, indent=2))

f.close()
with open('/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/compare_step_terms.json', 'w') as fh:
    json.dump(allResults, fh, indent=2)
