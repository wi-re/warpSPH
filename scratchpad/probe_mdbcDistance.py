#!/usr/bin/env python
"""Does the mDBC extrapolation travel the right DISTANCE?

Freeze the real sloshingTank configuration, overwrite the FLUID density with an
exact linear field rho = 1 + a.y, run `computeMdbcDensity` once, and compare the
boundary result against the same analytic field at the boundary positions.
English Eq. (12) is exact on a linear field, so any residual is the operator's
own error -- and if the extrapolation uses the wrong multiple of (r_b - r_g) the
error grows linearly with the layer's depth, with slope (lambda - 1) per unit
of |r_b - r_g|.
"""
import numpy as np, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run
from warpSPH.modules.mdbc import computeMdbcDensity

case = getCase('sloshingTank')
r = run(case, scheme='deltaSPH', nx=200, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False, plot=False)
st, cfg, sc = r.state.state, r.ctx.config, r.ctx.schemeConfig
adj = r.state.adjacency
dx = float(cfg.dx)

a = -0.05                                     # exact linear field rho = 1 + a*y
y = st.positions[:, 1]
analytic = 1.0 + a * y
st.densities = analytic.clone()               # fluid AND wall set to truth
rho_b = computeMdbcDensity(st, cfg, sc, adj)

k = st.kinds.detach().cpu().numpy()
pos = st.positions.detach().cpu().numpy()
got = rho_b.detach().cpu().numpy()
want = analytic.detach().cpu().numpy()
off = st.ghostOffsets.detach().cpu().numpy()  # boundary rows: r_b - r_g

bnd = (k == 1) & (np.abs(pos[:, 0]) < 0.15) & (pos[:, 1] < 0.5 * 0.093)  # flat floor
print(f'linear field rho = 1 + {a}*y   dx={dx:.5g}   floor boundary n={bnd.sum()}')
print(f'{"y/dx":>7} {"n":>4} {"|r_b-r_g|/dx":>13} {"rho_got":>10} {"rho_want":>10} '
      f'{"err":>10} {"implied lambda":>15}')
ys = np.round(pos[bnd, 1] / dx * 2) / 2
for yq in np.unique(ys):
    s = np.abs(ys - yq) < 1e-6
    if s.sum() < 3:
        continue
    d = np.linalg.norm(off[bnd][s], axis=1).mean()          # |r_b - r_g|
    g, w = got[bnd][s].mean(), want[bnd][s].mean()
    # rho_got = rho_g + lambda*(r_b-r_g).grad ; rho_g = 1 + a*y_g ; grad = a*yhat
    # (r_b - r_g).grad = a * (y_b - y_g) = a * (-d)  for a downward-pointing offset
    y_g = pos[bnd][s][:, 1].mean() - off[bnd][s][:, 1].mean()
    rho_g = 1.0 + a * y_g
    denom = a * (pos[bnd][s][:, 1].mean() - y_g)
    lam = (g - rho_g) / denom if abs(denom) > 1e-12 else float('nan')
    print(f'{yq:7.2f} {s.sum():4d} {d/dx:13.2f} {g:10.6f} {w:10.6f} {g-w:+10.6f} '
          f'{lam:15.3f}')
print('\nlambda = 1.000 everywhere -> English Eq. (12) exactly.')
print('lambda = 0.5 -> travels only to the surface;  2.0 -> travels twice too far.')
