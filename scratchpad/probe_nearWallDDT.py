#!/usr/bin/env python
"""A/B the delta-SPH DDT pair direction against the near-wall density profile.

The uncommitted change restricts the DDT outer sum to FluidToFluid (DualSPHysics
disables DDT for boundary neighbours). Near a wall roughly half a particle's
stencil IS boundary, so excluding it removes most of the diffusion exactly where
the lattice-scale density error is largest. Does the near-wall gradient anomaly
track it?
"""
import sys, numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run
from warpSPHCore import OperationDirection
import warpSPH.modules.deltaSPH.densityDiffusion as dd
import warpSPH.schemes.deltaSPH as sch

MODE = OperationDirection[sys.argv[1]] if len(sys.argv) > 1 else OperationDirection.FluidToFluid
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

_orig = dd.computeDensityDiffusion
def patched(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL):
    delta = schemeConfig.diffusionParams.densityDelta
    from warpSPHCore import sphKernel_xi
    xi = sphKernel_xi(config.kernel.value, config.dim)
    scale = delta * currentState.supports / xi * schemeConfig.fluid.fixedSoundSpeed
    return scale * dd.computeScalarFieldDiffusion(
        currentState, config, adjacency,
        schemeConfig.diffusionParams.densityDiffusionTerm,
        gradField=gradRho, gradFieldL=gradRhoL, operationMode=MODE)
dd.computeDensityDiffusion = patched
sch.computeDensityDiffusion = patched

r = run(getCase('sloshingTank'), scheme='deltaSPH', nx=200, nSteps=steps,
        tLimit=1e9, quiet=True, store=False, progress=False, plot=False)
st = r.state.state
pos = st.positions.detach().cpu().numpy(); rho = st.densities.detach().cpu().numpy()
k = st.kinds.detach().cpu().numpy(); dx = float(r.ctx.config.dx)
g, cs = 9.81, float(r.ctx.schemeConfig.fluid.fixedSoundSpeed)
exp = g * dx / cs**2

band = (np.abs(pos[:, 0]) < 0.15)
f = band & (k == 0)
ys = np.round(pos[f, 1] / dx * 2) / 2
rows = []
for yq in np.unique(ys)[:9]:
    s = np.abs(ys - yq) < 1e-6
    if s.sum() >= 3:
        rows.append((yq, rho[f][s].mean()))
print(f'=== DDT direction = {MODE.name}   steps={steps} ===')
print(f'{"rows":>13} {"d(rho)/dx":>12} {"x hydrostatic":>14}')
for (y0, r0), (y1, r1) in zip(rows, rows[1:]):
    print(f'{y0:5.1f}->{y1:5.1f} {r0-r1:12.3e} {(r0-r1)/exp:13.2f}x')
b = band & (k == 1) & (pos[:, 1] < 0.05)
yb = pos[b, 1]
top = np.abs(yb - yb.max()) < 0.25 * dx
print(f'interface step (top floor bnd - lowest fluid) = '
      f'{rho[b][top].mean() - rows[0][1]:+.6f}')
