"""CEILING_STICKING_PLAN.md §7.2: energy budget of the right-wall run-up film
from ceiling_forensics.py --region rightWall (<tag>_fin.pkl: every watched row,
every step: state at step start / after the integrator / after finalize, noPen
correction). Per window of steps, summed over the rows (mass-weighted, per unit
depth), E = m (1/2|v|^2 + c0^2 (rho-rho0)^2 / (2 rho0^2) + g y):

  dE int    E(after integrator) - E(step start)   (forces, incl. wall pressure)
  dE noPen  E(after finalize) - E(after integrator) on rows noPen corrected
  fired     row-steps noPen corrected;  tip = max y of the film; vy max

  python scratchpad/ceiling_film_budget.py s2_rw_finalize s2_rw_impulse [--window 100]
"""
import argparse
import os
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+')
ap.add_argument('--window', type=int, default=100)
ap.add_argument('--c0', type=float, default=97.04)
ap.add_argument('--yMin', type=float, default=-0.1, help='film rows: only those above this y [m] count')
a = ap.parse_args()
base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'out_ceiling', 'forensics')
c2, g = a.c0 ** 2, 9.81


def E(m, v, rho, x):
    return m * (0.5 * (v ** 2).sum(-1) + c2 * (rho - 1.0) ** 2 / 2 + g * x[:, 1])


for tag in a.tags:
    F = pickle.load(open(os.path.join(base, tag + '_fin.pkl'), 'rb'))
    ts = np.sqrt(9.81 / 0.6)
    print(f'\n{tag}: {len(F)} steps   (rows above y = {a.yMin} m; sums per {a.window} steps)')
    print('   t*     rows  fired   dE int      dE noPen    noPen/|int|   tip y   tip vy   max vy   P2-band rows')
    for w0 in range(0, len(F), a.window):
        dI = dN = 0.0; fired = rows = 0; tip = -9; tipv = 0; vmax = 0; nP2 = 0
        for f in F[w0:w0 + a.window]:
            sel = f['xPost'][:, 1] > a.yMin
            if not sel.any():
                continue
            m = f['m'][sel]
            e0 = E(m, f['v0'][sel], f['rho0'][sel], f['x0'][sel])
            e1 = E(m, f['vPre'][sel], f['rhoPre'][sel], f['xPre'][sel])
            e2 = E(m, f['vPost'][sel], f['rhoPost'][sel], f['xPost'][sel])
            on = np.abs(f['nopen'][sel]).sum(1) > 0
            dI += float((e1 - e0).sum()); dN += float((e2 - e1)[on].sum())
            fired += int(on.sum()); rows += int(sel.sum())
            k = np.argmax(f['xPost'][sel][:, 1])
            if f['xPost'][sel][k, 1] > tip:
                tip = float(f['xPost'][sel][k, 1]); tipv = float(f['vPost'][sel][k, 1])
            vmax = max(vmax, float(f['vPost'][sel][:, 1].max()))
            nP2 += int((np.abs(f['xPost'][sel][:, 1] - 0.0915) < 0.03).sum())
        t0 = F[w0]['t'] * ts
        print(f'{t0:6.3f} {rows:7d} {fired:6d} {dI:+10.3e} {dN:+10.3e} {abs(dN) / max(abs(dI), 1e-30):9.2f}   {tip:+.3f} {tipv:+7.2f} {vmax:+7.2f} {nP2:8d}')
