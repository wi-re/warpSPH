"""Full diffSPH physics + integration (incl. shifting/PST), run on warpSPH's
own diffSPH-matched dambreak IC -- fluid AND the real boundary/ghost mDBC
complement (not stripped), reusing warpSPH's own `ghostIndices`/`ghostOffsets`
arrays directly (same per-particle convention in both codebases: ghost g's
`ghostIndices[g]` is its paired boundary particle, `ghostOffsets[g]` the
ghost->boundary offset -- confirmed by reading both `modules/mdbc/*.py` and
diffSPH's `boundary.py`).

Diagnostics ("bands") are computed by calling warpSPH's OWN
`dambreakCase.diagnostics` on a warpSPH-shaped state built each step from the
diffSPH-advanced positions/velocities/densities -- the same measurement code
that produced the pure-warpSPH run's bands, so the comparison is apples to
apples ("same import/export harness" in spirit: same diagnostic harness).

Usage: --nSteps N to bound the run (pilot), omit for the full t=4 run.
"""
import sys
import copy
import json
import time
import argparse

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

ap = argparse.ArgumentParser()
ap.add_argument('--nSteps', type=int, default=None)
ap.add_argument('--diagEvery', type=int, default=1)
ap.add_argument('--out', type=str, required=True)
cliArgs = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

# ---------------------------------------------------------------------------
# 1. Build warpSPH's diffSPH-matched dambreak IC (nSteps=0 -> just the IC),
#    full particle complement (fluid + boundary + ghost).
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
fullState = result.state.state
device = fullState.positions.device
dtype = fullState.positions.dtype
nTotal = fullState.positions.shape[0]
print(f'total particles: {nTotal}  '
     f'(fluid {int((fullState.kinds==0).sum())}, '
     f'boundary {int((fullState.kinds==1).sum())}, '
     f'ghost {int((fullState.kinds==2).sum())})')
print('ghostIndices set:', fullState.ghostIndices is not None,
     'ghostOffsets set:', fullState.ghostOffsets is not None)

# ---------------------------------------------------------------------------
# 2. Build the matching diffSPH config (same as the earlier matched runs).
# ---------------------------------------------------------------------------
from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *
from diffSPH.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d
from diffSPH.modules.timestep import computeTimestep
from diffSPH.integration import getIntegrator

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
# `computeTimestepWCSPH` reads `config['particle']['support']` directly (NOT
# `state.supports.min()`) -- `initializeSimulation`/`buildRegion` normally
# populate it as a side effect of sampling, which we skip here since we build
# the state by hand from warpSPH's own IC. Leaving it unset silently falls
# back to that function's own `default=1` and blows the acoustic/viscous dt
# constraints up to the `maxDt` ceiling (found via a 50-step pilot: dt landed
# on exactly 0.001 = maxDt instead of ~0.0005, and fluid particles were
# already at max|v|=1.16 by t=0.05s under gravity alone).
supportValue = float(fullState.supports.mean())
configD['particle'] = {'nx': ctx.spec.nx, 'dx': ctx.config.dx,
                       'targetNeighbors': targetNeighborsD, 'band': 7,
                       'support': supportValue}
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

# ---------------------------------------------------------------------------
# 3. Translate warpSPH's full IC state into a diffSPH WeaklyCompressibleState,
#    reusing ghostIndices/ghostOffsets verbatim (same convention, same array
#    layout -- no reordering happened, so the indices stay valid).
# ---------------------------------------------------------------------------
def toI64(t):
    return t.detach().clone().to(torch.int64) if t is not None else None


stateD = WState_d(
    positions=fullState.positions.detach().clone(),
    supports=fullState.supports.detach().clone(),
    masses=fullState.masses.detach().clone(),
    densities=fullState.densities.detach().clone(),
    velocities=fullState.velocities.detach().clone(),
    pressures=torch.zeros(nTotal, device=device, dtype=dtype),
    soundspeeds=torch.full((nTotal,), c_s, device=device, dtype=dtype),
    kinds=toI64(fullState.kinds),
    materials=toI64(fullState.materials),
    UIDs=toI64(fullState.UIDs),
    UIDcounter=nTotal,
    ghostIndices=toI64(fullState.ghostIndices),
    ghostOffsets=fullState.ghostOffsets.detach().clone() if fullState.ghostOffsets is not None else None,
)
particleSystem = DeltaPlusSPHSystem(domainD, None, 0.0, stateD, 'momentum', None,
                                    rigidBodies=[], regions=[], config=configD)

dt = float(computeTimestep(schemeD, 1e-2, stateD, configD, None))
print(f'fixed dt = {dt:.6g}')
integrationScheme = getIntegrator(integrationD)

tLimit = 4.0
timesteps = cliArgs.nSteps if cliArgs.nSteps is not None else int(tLimit / dt)
print(f'running {timesteps} steps')

# ---------------------------------------------------------------------------
# 4. Loop, using warpSPH's OWN `dambreakCase.diagnostics` for the bands.
# ---------------------------------------------------------------------------
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState as WState_w


class _StateWrapper:
    pass


def toF32(t):
    return t.detach().clone().to(torch.int32) if t.dtype != torch.float32 else t.detach().clone()


def buildWrapper(stD):
    ws = WState_w(
        positions=stD.positions.detach(),
        velocities=stD.velocities.detach(),
        supports=stD.supports.detach(),
        masses=stD.masses.detach(),
        densities=stD.densities.detach(),
        kinds=stD.kinds.detach().to(torch.int32),
        materials=stD.materials.detach().to(torch.int32),
        UIDs=stD.UIDs.detach().to(torch.int32),
        UIDcounter=int(stD.UIDcounter),
        pressures=stD.pressures.detach() if stD.pressures is not None else None,
        soundspeeds=stD.soundspeeds.detach() if stD.soundspeeds is not None else None,
    )
    w = _StateWrapper()
    w.state = ws
    return w


traj = []
diverged = False
t0 = time.time()
for i in range(timesteps):
    particleSystem, currentState, updates = integrationScheme.function(
        particleSystem, dt, simulatorD, configD, priorStep=None, verbose=False)

    st = particleSystem.systemState
    if not torch.isfinite(st.velocities).all() or not torch.isfinite(st.densities).all():
        print(f'non-finite state at step {i}; stopping.')
        diverged = True
        break

    if i % cliArgs.diagEvery == 0 or i == timesteps - 1:
        wrapper = buildWrapper(st)
        row = dambreakCase.diagnostics(ctx, wrapper)
        row['step'] = i
        row['t'] = float(particleSystem.t)
        traj.append(row)

    if i % 200 == 0:
        last = traj[-1] if traj else {}
        print(f'step {i}/{timesteps}  t={float(particleSystem.t):.4f}  '
             f'maxV={last.get("maxVelocity", float("nan")):.4f}  '
             f'rho=[{last.get("minDensity", float("nan")):.4f},'
             f'{last.get("maxDensity", float("nan")):.4f}]  '
             f'({time.time()-t0:.1f}s elapsed)', flush=True)

wallTime = time.time() - t0
print(f'finished in {wallTime:.1f}s, diverged={diverged}, steps run={len(traj)}')

import numpy as np
if traj:
    maxV = np.array([r['maxVelocity'] for r in traj])
    ke = np.array([r['kineticEnergy'] for r in traj])
    minD = np.array([r['minDensity'] for r in traj])
    maxD = np.array([r['maxDensity'] for r in traj])
    p05 = np.array([r.get('densityP05', float('nan')) for r in traj])
    p99 = np.array([r.get('densityP99', float('nan')) for r in traj])
    summary = dict(
        dt=dt, timesteps=timesteps, stepsRun=len(traj), diverged=diverged, wallTime=wallTime,
        maxVelocity_final=float(maxV[-1]), maxVelocity_max=float(maxV.max()),
        kineticEnergy_final=float(ke[-1]), kineticEnergy_max=float(ke.max()),
        minDensity_final=float(minD[-1]), minDensity_min=float(minD.min()),
        maxDensity_final=float(maxD[-1]), maxDensity_max=float(maxD.max()),
        densityP05_min=float(np.nanmin(p05)), densityP99_max=float(np.nanmax(p99)),
    )
    print(json.dumps(summary, indent=2))
    with open(cliArgs.out + '.summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

with open(cliArgs.out + '.json', 'w') as fh:
    json.dump(dict(dt=dt, timesteps=timesteps, diverged=diverged, traj=traj), fh)
print('saved', cliArgs.out + '.json')
