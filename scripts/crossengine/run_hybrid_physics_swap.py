"""Level 2a narrowing: run warpSPH's OWN case/runner/integrator/shift/export
pipeline completely unchanged, but monkey-patch the single function that
computes the raw per-step force/BC update
(`warpSPH.schemes.builder`'s bound name `deltaSPH_step`) to instead compute it
via diffSPH's `deltaPlusSPHScheme` on a translated copy of the state, then
translate the result back into warpSPH's `WeaklyCompressibleSystemUpdate`
shape. Everything else -- symplecticEuler's kick-drift-kick, the delta+ PST
shift, diagnostics, trajectory export -- stays warpSPH's own code, unmodified.
This isolates: does swapping ONLY the force/BC term (holding warpSPH's own
integrator+shift+IC fixed) narrow the density/velocity bands toward diffSPH's?
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

# ---------------------------------------------------------------------------
# 1. Build the matching diffSPH config/domain ONCE, from a throwaway nSteps=0
#    warpSPH IC (cheap) -- reused by the wrapper on every call.
# ---------------------------------------------------------------------------
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
from diffSPH.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d

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
configD['shifting']['active'] = False   # shift stays warpSPH's own job in this stage
configD['pressure']['term'] = 'Antuono'
configD['gravity'] = {
    'active': True, 'magnitude': 10.0, 'mode': 'directional',
    'direction': torch.tensor([0, -1.0], device=device, dtype=dtype),
}
configD['regions'] = []

print(f'diffSPH physics config ready: c_s={c_s:.6g}, support={supportValue:.6g}, dx={dxVal:.6g}')

# ---------------------------------------------------------------------------
# 2. The monkey-patched force function -- warpSPH call signature in, warpSPH
#    call signature out; diffSPH does the actual work in between.
# ---------------------------------------------------------------------------
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystemUpdate
from warpSPHCore import SupportScheme, buildVerletList

nCalls = [0]


def diffPhysicsWrapper(system, dt, config, schemeConfig, verbose=False, stageIndex=None):
    nCalls[0] += 1
    st = system.state
    n = st.positions.shape[0]
    pdt = st.positions.dtype

    stateD = WState_d(
        positions=st.positions.detach().clone(),
        supports=st.supports.detach().clone(),
        masses=st.masses.detach().clone(),
        densities=st.densities.detach().clone(),
        velocities=st.velocities.detach().clone(),
        pressures=torch.zeros(n, device=st.positions.device, dtype=pdt),
        soundspeeds=torch.full((n,), c_s, device=st.positions.device, dtype=pdt),
        kinds=st.kinds.detach().clone().to(torch.int64),
        materials=st.materials.detach().clone().to(torch.int64),
        UIDs=st.UIDs.detach().clone().to(torch.int64),
        UIDcounter=n,
        ghostIndices=st.ghostIndices.detach().clone().to(torch.int64) if st.ghostIndices is not None else None,
        ghostOffsets=st.ghostOffsets.detach().clone() if st.ghostOffsets is not None else None,
    )
    diffSystem = DeltaPlusSPHSystem(domainD, None, float(system.t), stateD, 'momentum', None,
                                    rigidBodies=[], regions=[], config=configD)
    updateD, particlesD, neighborhoodD = deltaPlusSPHScheme(diffSystem, float(dt), configD)

    updateW = WeaklyCompressibleSystemUpdate(
        dxdt=updateD.positions.to(pdt),
        dvdt=updateD.velocities.to(pdt),
        drhodt=updateD.densities.to(pdt),
        passive=torch.zeros(n, dtype=torch.bool, device=st.positions.device),
    )
    # Refresh a genuine warpSPH adjacency (diffSPH's own neighborhood object is
    # a different type) so downstream warpSPH code -- the shift step,
    # `particleDistributionMetrics` -- keeps working normally.
    adjacency = buildVerletList(st, config.domain, verletScale=config.verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=system.adjacency, verbose=False)
    system.adjacency = adjacency
    if nCalls[0] % 400 == 0:
        fluid = st.kinds == 0
        vmag = torch.linalg.norm(st.velocities[fluid], dim=-1)
        print(f'  [wrapper call {nCalls[0]}] t={float(system.t):.4f} '
             f'maxV={float(vmag.max()):.4f} '
             f'rho=[{float(st.densities[fluid].min()):.4f},{float(st.densities[fluid].max()):.4f}] '
             f'dt={float(dt):.3g}', flush=True)
    return updateW, adjacency, st


import warpSPH.schemes.builder as builder_mod
builder_mod.deltaSPH_step = diffPhysicsWrapper

# --- work around a real diffSPH bug (not a warpSPH issue): `math.pinv2x2`
# masks the assignment with `|o1| > 1e-5` but reads the reciprocal with
# `|o1| > 1e-7` -- whenever a covariance eigenvalue lands in (1e-7, 1e-5], the
# two masks have different cardinalities and the assignment shape-mismatches.
# diffSPH's own sparser wall/ghost sampling apparently never lands an
# eigenvalue in that narrow band; warpSPH's denser mDBC complement did, ~1200
# steps into the physics-only hybrid run. Patched here with a single
# consistent threshold so the ablation can proceed; not a fix to the diffSPH
# repo itself.
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

# ---------------------------------------------------------------------------
# 3. Run warpSPH's OWN full pipeline, unmodified otherwise.
# ---------------------------------------------------------------------------
try:
    result = caseMain(dambreakCase, MATCHED_ARGS)
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {e}")
    import sys; sys.exit(1)
print(f'diffSPH physics wrapper was called {nCalls[0]} times')
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
