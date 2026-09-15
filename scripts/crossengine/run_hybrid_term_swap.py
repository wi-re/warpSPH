"""2x2 term-swap ablation (DELTASPH_VALIDATION_PLAN §8.9's next step): does
running warpSPH with diffSPH's artificial-viscosity and/or density-diffusion
(DDT) term(s) swapped in change stability/bands? Everything else -- warpSPH's
own runner/integrator/pressure-force/continuity/gravity/shift/diagnostics/
export -- stays exactly as-is; `--swap` selects which of the two dissipative
terms (found genuinely divergent in §8.9, unlike pressure/continuity which
were bit-exact) gets replaced by a call into diffSPH.

`--swap none` is a baseline/integrity check -- with no patches applied at
all it must reproduce the original pure-warpSPH numbers.
"""
import sys
import json
import argparse

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

ap = argparse.ArgumentParser()
ap.add_argument('--swap', choices=['none', 'viscosity', 'ddt', 'both'], default='none')
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
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState as WState_d
from diffSPH.neighborhood import evaluateNeighborhood, SupportScheme as DSupportScheme
from diffSPH.modules.velocityDiffusion import computeViscosity_deltaSPH_inviscid
from diffSPH.modules.densityDiffusion import computeDensityDeltaTerm
from diffSPH.modules.density import computeGradRho, computeGradRhoL
from diffSPH.modules.renorm import computeCovarianceMatrices
from diffSPH.neighborhood import coo_to_csr, filterNeighborhoodByKind

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
configD['diffusion'] = {'alpha': float(_ctx.schemeConfig.diffusionParams.inviscidAlpha),
                        'delta': 0.1}  # matches warpSPH's own default (both banners agree, §8.9)

print(f'diffSPH term config ready: c_s={c_s:.6g}, support={supportValue:.6g}, '
     f'alpha={configD["diffusion"]["alpha"]}, delta={configD["diffusion"]["delta"]}, swap={cliArgs.swap}')


def buildDiffState(st):
    """Full-state translation (all kinds -- fluid/boundary/ghost), reusing
    warpSPH's own ghostIndices/ghostOffsets verbatim (same convention in both
    codebases, §8.4)."""
    n = st.positions.shape[0]
    pdt = st.positions.dtype
    return WState_d(
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
        ghostIndices=st.ghostIndices.detach().clone().to(torch.int64)
        if st.ghostIndices is not None else None,
        ghostOffsets=st.ghostOffsets.detach().clone()
        if st.ghostOffsets is not None else None,
    )


nViscCalls = [0]
nDdtCalls = [0]


def diffViscosityWrapper(currentState, config, schemeConfig, adjacency, approachOnly=False):
    nViscCalls[0] += 1
    pdt = currentState.positions.dtype
    stateD = buildDiffState(currentState)
    _, neighbors = evaluateNeighborhood(
        stateD, domainD, kernelD, verletScale=configD['neighborhood']['verletScale'],
        mode=DSupportScheme.SuperSymmetric, priorNeighborhood=None,
        computeHessian=configD['neighborhood']['computeHessian'],
        computeDkDh=configD['neighborhood']['computeDkDh'])
    # diffSPH's own call site uses the 'fluid' neighbour filter for viscosity
    # (deltaSPH.py's deltaPlusSPHScheme, "Velocity Diffusion" section).
    dvdt_diss = computeViscosity_deltaSPH_inviscid(
        stateD, kernelD, neighbors.get('fluid'), DSupportScheme.Gather, configD)
    if nViscCalls[0] % 400 == 0:
        fluid = currentState.kinds == 0
        mag = torch.linalg.norm(dvdt_diss[fluid], dim=-1)
        print(f'  [viscosity call {nViscCalls[0]}] mean|.|={float(mag.mean()):.4g} '
             f'max|.|={float(mag.max()):.4g}', flush=True)
    return dvdt_diss.to(pdt)


def diffDdtWrapper(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL):
    nDdtCalls[0] += 1
    pdt = currentState.densities.dtype
    stateD = buildDiffState(currentState)
    _, neighbors = evaluateNeighborhood(
        stateD, domainD, kernelD, verletScale=configD['neighborhood']['verletScale'],
        mode=DSupportScheme.SuperSymmetric, priorNeighborhood=None,
        computeHessian=configD['neighborhood']['computeHessian'],
        computeDkDh=configD['neighborhood']['computeDkDh'])
    noghost = neighbors.get('noghost')
    # diffSPH's own pipeline sets `numNeighbors` right after the neighbour
    # search, then covariance/renormalization matrices (step "07") before
    # `computeGradRhoL` (step "09.1") -- omitting either crashed the pilot
    # ("Gradient renormalization matrix is None", then "Particles must
    # Number of Neighbors computed").
    stateD.numNeighbors = coo_to_csr(filterNeighborhoodByKind(stateD, neighbors.neighbors, which='fluid')).rowEntries
    stateD.covarianceMatrices, stateD.gradCorrectionMatrices, stateD.eigenValues = \
        computeCovarianceMatrices(stateD, kernelD, neighbors.get('normal'), DSupportScheme.Scatter, configD)
    stateD.gradRhoL = computeGradRhoL(stateD, kernelD, noghost, DSupportScheme.Gather, configD)
    stateD.gradRho = computeGradRho(stateD, kernelD, noghost, DSupportScheme.Gather, configD)
    drhodt_diss = computeDensityDeltaTerm(stateD, kernelD, noghost, DSupportScheme.Gather, configD)
    if nDdtCalls[0] % 400 == 0:
        fluid = currentState.kinds == 0
        mag = drhodt_diss[fluid].abs()
        print(f'  [ddt call {nDdtCalls[0]}] mean|.|={float(mag.mean()):.4g} '
             f'max|.|={float(mag.max()):.4g}', flush=True)
    return drhodt_diss.to(pdt)


import warpSPH.schemes.deltaSPH as w_deltaSPH_mod

if cliArgs.swap in ('viscosity', 'both'):
    w_deltaSPH_mod.computeVelocityDiffusion = diffViscosityWrapper
if cliArgs.swap in ('ddt', 'both'):
    w_deltaSPH_mod.computeDensityDiffusion = diffDdtWrapper

# --- diffSPH pinv2x2 threshold-mismatch workaround, same as the earlier
# hybrid runs (renorm.computeCovarianceMatrices is not on this code path for
# viscosity/DDT specifically, but evaluateNeighborhood's Hessian option is
# off here so this is likely a no-op precaution; kept for safety/parity).
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

print(f'swap={cliArgs.swap}  viscosity calls: {nViscCalls[0]}  ddt calls: {nDdtCalls[0]}')
print('diverged:', result.diverged, 'nSteps:', result.nSteps)

import numpy as np
if result.trajectory:
    maxV = np.array([r['maxVelocity'] for r in result.trajectory if 'maxVelocity' in r])
    ke = np.array([r['kineticEnergy'] for r in result.trajectory if 'kineticEnergy' in r])
    minD = np.array([r['minDensity'] for r in result.trajectory if 'minDensity' in r])
    maxD = np.array([r['maxDensity'] for r in result.trajectory if 'maxDensity' in r])
    summary = dict(
        swap=cliArgs.swap, diverged=result.diverged, nSteps=result.nSteps,
        maxVelocity_final=float(maxV[-1]), maxVelocity_max=float(maxV.max()),
        kineticEnergy_final=float(ke[-1]), kineticEnergy_max=float(ke.max()),
        minDensity_final=float(minD[-1]), minDensity_min=float(minD.min()),
        maxDensity_final=float(maxD[-1]), maxDensity_max=float(maxD.max()),
    )
    print(json.dumps(summary, indent=2))
    with open(cliArgs.out + '.summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
