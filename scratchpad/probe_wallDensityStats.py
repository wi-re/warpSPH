#!/usr/bin/env python
"""Boundary vs fluid density distribution at an early step, and where the
boundary value comes from (stored vs a fresh mDBC extrapolation)."""
import sys, numpy as np, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run
from warpSPH.modules.mdbc import computeMdbcDensity

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
r = run(getCase('sloshingTank'), scheme='deltaSPH', nx=200, nSteps=steps,
        tLimit=1e9, quiet=True, store=False, progress=False, plot=False)
st, cfg, sc, adj = r.state.state, r.ctx.config, r.ctx.schemeConfig, r.state.adjacency
pos = st.positions.detach().cpu().numpy()
rho = st.densities.detach().cpu().numpy()
k = st.kinds.detach().cpu().numpy()
dx = float(cfg.dx); fill = float(r.ctx.param('fillDepth'))

def stat(lbl, m):
    v = rho[m]
    print(f'  {lbl:34s} n={m.sum():5d}  min={v.min():.6f} p05={np.percentile(v,5):.6f} '
          f'med={np.median(v):.6f} max={v.max():.6f}')

print(f'steps={steps}  t={steps*float(cfg.dt):.4f}')
stat('fluid', k == 0)
stat('boundary (all)', k == 1)
wet = (k == 1) & (pos[:, 1] < fill)
stat('boundary, below waterline', wet)
stat('boundary, above waterline', (k == 1) & (pos[:, 1] >= fill))
stat('floor band |x|<0.15, y<0', (k == 1) & (np.abs(pos[:,0]) < 0.15) & (pos[:,1] < 0))

# Is the STORED boundary density what a fresh mDBC call would give right now?
fresh = computeMdbcDensity(st, cfg, sc, adj).detach().cpu().numpy()
d = fresh[k == 1] - rho[k == 1]
print(f'\n  fresh computeMdbcDensity - stored, on boundary rows:')
print(f'    max|diff| = {np.abs(d).max():.6e}   mean|diff| = {np.abs(d).mean():.6e}')
print(f'    stored  median = {np.median(rho[k==1]):.6f}')
print(f'    fresh   median = {np.median(fresh[k==1]):.6f}')
wetm = wet[k == 1]
if wetm.any():
    print(f'    wetted: stored median {np.median(rho[k==1][wetm]):.6f}  '
          f'fresh median {np.median(fresh[k==1][wetm]):.6f}')
