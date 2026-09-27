"""Level 2a (reverse): run warpSPH's OWN case/runner/integrator/force-step
pipeline completely unchanged, but monkey-patch ONLY the shift/PST step
(`warpSPH.systems.weaklyCompressible`'s bound name `solveShifting`) to
compute the shift vector via diffSPH's own `solveShifting` instead of
warpSPH's `computeDeltaShift`-based one. `deltaSPH_step` (the raw force/BC
term) stays warpSPH's own, untouched.

Combination matrix so far:
  physics=warp + shift=warp  -> STABLE (pure warpSPH run, wide bands)
  physics=diff + shift=diff  -> STABLE (Stage 1, tight bands matching diffSPH)
  physics=diff + shift=warp  -> DIVERGES catastrophically by t~0.12
  physics=warp + shift=diff  -> this script
"""
import sys
import json
import argparse

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

ap = argparse.ArgumentParser()
ap.add_argument('--nSteps', type=int, default=None)
ap.add_argument('--out', type=str, required=True)
cliArgs = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

MATCHED_ARGS = [
    '--nx', '64', '--L', '2.0', '--n_h', '4.0',
    '--kernel', 'Wendland4', '--integrationScheme', 'symplecticEuler',
    '--supportMode', 'SuperSymmetric', '--tLimit', '4.0', '--targetDt', '0.0005',
    '--band', '7', '--fillRatio', '0.5', '--fluidWidth', str(5.0 / 12.0),
    '--gravityMagnitude', '10.0', '--wallBC', 'constant',
    '--no-plot', '--no-store', '--no-video', '--quiet', '--precision', 'float32',
]
if cliArgs.nSteps is not None:
    MATCHED_ARGS += ['--nSteps', str(cliArgs.nSteps)]

from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import caseMain

_probe = caseMain(dambreakCase, MATCHED_ARGS[:MATCHED_ARGS.index('--wallBC') + 2] +
                  ['--nSteps', '0', '--no-plot', '--no-store', '--no-video', '--quiet',
                   '--precision', 'float32'])
_ctx = _probe.ctx
c_s = float(_ctx.schemeConfig.fluid.fixedSoundSpeed)
dxVal = float(_ctx.config.dx)
supportValue = float(_probe.state.state.supports.mean())

from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *
from diffSPH.schemes.deltaSPH import DeltaPlusSPHSystem
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d
from diffSPH.modules.particleShifting import solveShifting as diffSolveShifting

kernelD = KernelType.Wendland4
schemeD = SimulationScheme.DeltaSPH
integrationD = IntegrationSchemeType.symplecticEuler

domMin = _ctx.config.domain.min.detach().cpu().tolist()
domMax = _ctx.config.domain.max.detach().cpu().tolist()
device = _ctx.config.domain.min.device
dtype = _ctx.config.domain.min.dtype
domainD = buildDomainDescription(float(domMax[0] - domMin[0]), 2, False, device, dtype)
domainD.min = torch.tensor(domMin, device=device, dtype=dtype)
domainD.max = torch.tensor(domMax, device=device, dtype=dtype)

targetNeighborsD = n_h_to_nH(4, 2)
_, _, configD, _ = getSimulationScheme(schemeD, kernelD, integrationD, 1.0, targetNeighborsD, domainD)
configD['particle'] = {'nx': _ctx.spec.nx, 'dx': dxVal, 'targetNeighbors': targetNeighborsD,
                       'band': 7, 'support': supportValue}
configD['fluid'] = {'rho0': 1.0, 'c_s': c_s}
configD['surfaceDetection']['active'] = True
configD['shifting']['freeSurface'] = True
configD['shifting']['active'] = True
configD['pressure']['term'] = 'Antuono'
configD['gravity'] = {
    'active': True, 'magnitude': 10.0, 'mode': 'directional',
    'direction': torch.tensor([0, -1.0], device=device, dtype=dtype),
}
configD['regions'] = []

print(f'diffSPH shift config ready: c_s={c_s:.6g}, support={supportValue:.6g}, dx={dxVal:.6g}')

nShiftCalls = [0]


def diffShiftWrapper(systemState, config, schemeConfig, adjacency, dt, verbose=False):
    nShiftCalls[0] += 1
    n = systemState.positions.shape[0]
    pdt = systemState.positions.dtype
    stateD = WState_d(
        positions=systemState.positions.detach().clone(),
        supports=systemState.supports.detach().clone(),
        masses=systemState.masses.detach().clone(),
        densities=systemState.densities.detach().clone(),
        velocities=systemState.velocities.detach().clone(),
        pressures=torch.zeros(n, device=systemState.positions.device, dtype=pdt),
        soundspeeds=torch.full((n,), c_s, device=systemState.positions.device, dtype=pdt),
        kinds=systemState.kinds.detach().clone().to(torch.int64),
        materials=systemState.materials.detach().clone().to(torch.int64),
        UIDs=systemState.UIDs.detach().clone().to(torch.int64),
        UIDcounter=n,
        ghostIndices=systemState.ghostIndices.detach().clone().to(torch.int64)
        if systemState.ghostIndices is not None else None,
        ghostOffsets=systemState.ghostOffsets.detach().clone()
        if systemState.ghostOffsets is not None else None,
    )
    diffSystem = DeltaPlusSPHSystem(domainD, None, 0.0, stateD, 'momentum', None,
                                    rigidBodies=[], regions=[], config=configD)
    # diffSPH's own `DeltaPlusSPHSystem.finalize` calls this with a fixed 0.1
    # scale (not the physical dt) -- reusing that same convention here.
    dx, *_rest = diffSolveShifting(diffSystem, 0.1, configD, verbose=False)
    if nShiftCalls[0] % 200 == 0:
        fluid = systemState.kinds == 0
        print(f'  [shift call {nShiftCalls[0]}] max|dx|={float(dx[fluid].norm(dim=-1).max()):.4g} '
             f'mean|dx|={float(dx[fluid].norm(dim=-1).mean()):.4g}', flush=True)
    return dx.to(pdt)


import warpSPH.systems.weaklyCompressible as wc_mod
wc_mod.solveShifting = diffShiftWrapper

# --- same diffSPH pinv2x2 threshold-mismatch workaround as the other hybrid
# runs (see run_hybrid_physics_swap.py's comment for the full explanation).
import diffSPH.modules.renorm as renorm_mod


def _pinv2x2_fixed(M):
    a, b, c, d = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    theta = 0.5 * torch.atan2(2 * a * c + 2 * b * d, a**2 + b**2 - c**2 - d**2)
    cosTheta, sinTheta = torch.cos(theta), torch.sin(theta)
    U = torch.zeros_like(M)
    U[:, 0, 0], U[:, 0, 1], U[:, 1, 0], U[:, 1, 1] = cosTheta, -sinTheta, sinTheta, cosTheta
    S1 = a**2 + b**2 + c**2 + d**2
    S2 = torch.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4 * (a * c + b * d)**2)
    o1 = torch.sqrt((S1 + S2) / 2)
    o2 = torch.sqrt(torch.clamp(S1 - S2, min=1e-9) / 2)
    phi = 0.5 * torch.atan2(2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2)
    cosPhi, sinPhi = torch.cos(phi), torch.sin(phi)
    s11 = torch.sign((a * cosTheta + c * sinTheta) * cosPhi + (b * cosTheta + d * sinTheta) * sinPhi)
    s22 = torch.sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)
    V = torch.zeros_like(M)
    V[:, 0, 0], V[:, 0, 1], V[:, 1, 0], V[:, 1, 1] = cosPhi * s11, -sinPhi * s22, sinPhi * s11, cosPhi * s22
    eps = 1e-5
    o1_1 = torch.where(torch.abs(o1) > eps, 1.0 / o1.clamp_min(eps), torch.zeros_like(o1))
    o2_1 = torch.where(torch.abs(o2) > eps, 1.0 / o2.clamp_min(eps), torch.zeros_like(o2))
    o = torch.vstack((o1_1, o2_1))
    S_1 = torch.diag_embed(o.mT, dim1=2, dim2=1)
    eigVals = torch.vstack((o1, o2)).mT
    swap = torch.abs(eigVals[:, 1]) > torch.abs(eigVals[:, 0])
    eigVals = eigVals.clone()
    eigVals[swap, :] = torch.flip(eigVals[swap, :], [1])
    return torch.matmul(torch.matmul(V, S_1), U.mT), eigVals


renorm_mod.pinv2x2 = _pinv2x2_fixed

try:
    result = caseMain(dambreakCase, MATCHED_ARGS)
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {e}")
    import sys
    sys.exit(1)

print(f'shift wrapper calls: {nShiftCalls[0]}')
print('diverged:', result.diverged, 'nSteps:', result.nSteps)

import numpy as np
if result.trajectory:
    maxV = np.array([r['maxVelocity'] for r in result.trajectory if 'maxVelocity' in r])
    ke = np.array([r['kineticEnergy'] for r in result.trajectory if 'kineticEnergy' in r])
    minD = np.array([r['minDensity'] for r in result.trajectory if 'minDensity' in r])
    maxD = np.array([r['maxDensity'] for r in result.trajectory if 'maxDensity' in r])
    summary = dict(
        diverged=result.diverged, nSteps=result.nSteps,
        maxVelocity_final=float(maxV[-1]), maxVelocity_max=float(maxV.max()),
        kineticEnergy_final=float(ke[-1]), kineticEnergy_max=float(ke.max()),
        minDensity_final=float(minD[-1]), minDensity_min=float(minD.min()),
        maxDensity_final=float(maxD[-1]), maxDensity_max=float(maxD.max()),
    )
    print(json.dumps(summary, indent=2))
    with open(cliArgs.out + '.summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    with open(cliArgs.out + '.traj.json', 'w') as fh:
        json.dump(result.trajectory, fh)
