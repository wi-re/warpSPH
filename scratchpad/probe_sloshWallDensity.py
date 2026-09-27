#!/usr/bin/env python
"""Density across the fluid->boundary interface in sloshingTank, at rest.

Reports, for the flat bottom wall mid-span, the density of each fluid lattice
row and each boundary lattice row against the hydrostatic expectation
`rho0 * (1 + g*(h_fill - y)/c^2)`. A step at y = 0 is the mDBC extrapolation
disagreeing with the fluid it is supposed to continue.
"""
import sys, numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

nSteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
case = getCase('sloshingTank')
r = run(case, scheme='deltaSPH', nx=200, nSteps=nSteps, tLimit=1e9,
        quiet=True, store=False, progress=False, plot=False)

st = r.state.state
pos = st.positions.detach().cpu().numpy()
rho = st.densities.detach().cpu().numpy()
kinds = st.kinds.detach().cpu().numpy()
dx = float(r.ctx.config.dx)
g = float(r.ctx.schemeConfig.gravityConfig.magnitude)
cs = float(r.ctx.schemeConfig.fluid.fixedSoundSpeed)
fill = float(r.ctx.param('fillDepth'))
rho0 = float(r.ctx.schemeConfig.fluid.restDensity)

print(f'steps={nSteps}  dx={dx:.5g}  c_s={cs:g}  g={g:g}  fill={fill:g}  rho0={rho0:g}')
print(f'hydrostatic span over the fluid depth: '
      f'{rho0 * g * fill / cs**2:.6f}  (rho at the floor = {rho0*(1+g*fill/cs**2):.6f})')

band = np.abs(pos[:, 0]) < 0.15          # mid-span, away from the corners
print(f'\n mid-span column (|x| < 0.15), y-row averages')
print(f'{"y/dx":>8} {"kind":>6} {"n":>5} {"rho_mean":>10} {"rho_std":>9} '
      f'{"hydrostatic":>12} {"err":>10}')
for k, lbl in ((0, 'fluid'), (1, 'bnd')):
    m = band & (kinds == k)
    ys = pos[m, 1]; rs = rho[m]
    for yq in np.unique(np.round(ys / dx * 2) / 2)[:9 if k == 1 else 8]:
        sel = np.abs(np.round(ys / dx * 2) / 2 - yq) < 1e-6
        if sel.sum() < 3:
            continue
        hyd = rho0 * (1 + g * (fill - yq * dx) / cs**2)
        print(f'{yq:8.2f} {lbl:>6} {sel.sum():5d} {rs[sel].mean():10.6f} '
              f'{rs[sel].std():9.2e} {hyd:12.6f} {rs[sel].mean()-hyd:+10.6f}')

# the step itself: lowest fluid row vs highest boundary row, same |x| band
f = band & (kinds == 0); b = band & (kinds == 1) & (pos[:, 1] < 0.5 * fill)
yf = pos[f, 1]; yb = pos[b, 1]
lowF = np.abs(yf - yf.min()) < 0.25 * dx
topB = np.abs(yb - yb.max()) < 0.25 * dx
print(f'\n lowest fluid row  y={yf.min()/dx:+.2f} dx : rho = {rho[f][lowF].mean():.6f}')
print(f' top FLOOR bnd row y={yb.max()/dx:+.2f} dx : rho = {rho[b][topB].mean():.6f}')
print(f' JUMP across the interface = {rho[b][topB].mean() - rho[f][lowF].mean():+.6f}'
      f'  ({abs(rho[b][topB].mean() - rho[f][lowF].mean())/(rho0*g*fill/cs**2)*100:.0f}% '
      f'of the full hydrostatic span)')
