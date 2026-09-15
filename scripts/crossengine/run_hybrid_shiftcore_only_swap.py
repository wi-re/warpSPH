"""Level 2c narrowing (narrowest yet): run warpSPH's OWN case/runner/
integrator/force-step/shift-*wrapper* pipeline completely unchanged, and
monkey-patch ONLY the innermost raw kernel-gradient shift-vector computation
(`warpSPH.modules.shifting.wrapper`'s bound name `computeDeltaShift`, called
once per shift iteration at wrapper.py:216) with diffSPH's equivalent raw
computation (`computeDeltaShifting`). Everything AROUND it in
`solveShifting` -- free-surface detection, dot/mat/surfaceNormal projection,
the Sun 2019 Eq.(14) velocity-fraction cap, the per-component threshold
clamp, zeroing non-fluid particles, even the OUTER shift iteration loop --
stays warpSPH's own code, completely untouched. This isolates the raw
shift-magnitude/direction question in isolation from every other knob.

Sign convention (verified by `cross_engine_shift_compare.py`): the quantity
that gets ADDED directly to positions is warpSPH's own `shift`/`update`
return value, and it is diffSPH's raw `computeDeltaShifting(...)` output
UN-negated (diffSPH's caller does `update = -computeDeltaShifting(...)` then
`positions -= update`, so the double negation cancels). Direction agrees
well between engines (cosine ~0.95 mean/0.998 median on a real mid-
trajectory snapshot); magnitude differs ~4x (warpSPH's default/historical
scaling is the weaker one).
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
shiftCFL = float(_ctx.schemeConfig.shiftProperties.CFL)
shiftComputeMach = bool(_ctx.schemeConfig.shiftProperties.computeMach)

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
configD['shifting']['CFL'] = shiftCFL
configD['shifting']['computeMach'] = shiftComputeMach

print(f'diffSPH raw-shift config ready: c_s={c_s:.6g}, support={supportValue:.6g}, '
     f'CFL={shiftCFL}, computeMach={shiftComputeMach}')

nCalls = [0]


def diffComputeDeltaShiftWrapper(currentState, config, schemeConfig, domain, adjacency, iters=1):
    """Drop-in replacement for `warpSPH.modules.shifting.delta.computeDeltaShift`
    (as bound in `wrapper.py`'s namespace): same `(update, adjacency)` return
    shape, `update` computed via diffSPH's `computeDeltaShifting` instead.
    """
    nCalls[0] += 1
    n = currentState.positions.shape[0]
    pdt = currentState.positions.dtype
    stateD = WState_d(
        positions=currentState.positions.detach().clone(),
        supports=currentState.supports.detach().clone(),
        masses=currentState.masses.detach().clone(),
        densities=currentState.densities.detach().clone(),
        velocities=currentState.velocities.detach().clone(),
        pressures=torch.zeros(n, device=currentState.positions.device, dtype=pdt),
        soundspeeds=torch.full((n,), c_s, device=currentState.positions.device, dtype=pdt),
        kinds=currentState.kinds.detach().clone().to(torch.int64),
        materials=currentState.materials.detach().clone().to(torch.int64),
        UIDs=currentState.UIDs.detach().clone().to(torch.int64),
        UIDcounter=n,
    )
    neighborhoodInfo, neighbors = evaluateNeighborhood(
        stateD, domainD, kernelD, verletScale=configD['neighborhood']['verletScale'],
        mode=DSupportScheme.SuperSymmetric, priorNeighborhood=None,
        computeHessian=configD['neighborhood']['computeHessian'],
        computeDkDh=configD['neighborhood']['computeDkDh'])
    noghost = neighbors.get('noghost')[0]
    raw_d = computeDeltaShifting(stateD, domainD, kernelD, noghost, configD)
    update = raw_d.to(pdt)   # NOT negated -- see module docstring above.
    if nCalls[0] % 200 == 0:
        fluid = currentState.kinds == 0
        mag = torch.linalg.norm(update[fluid], dim=-1)
        print(f'  [shiftcore call {nCalls[0]}] max|dx|={float(mag.max()):.4g} '
             f'mean|dx|={float(mag.mean()):.4g}', flush=True)
    return update, adjacency


import warpSPH.modules.shifting.wrapper as wrapper_mod
wrapper_mod.computeDeltaShift = diffComputeDeltaShiftWrapper

try:
    result = caseMain(dambreakCase, MATCHED_ARGS)
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {e}")
    import sys
    sys.exit(1)

print(f'shiftcore wrapper calls: {nCalls[0]}')
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
