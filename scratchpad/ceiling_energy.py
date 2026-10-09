"""CEILING_STICKING_PLAN.md §7: per-cycle energy budget of one watched particle
from ceiling_forensics.py output (rows.npz + wall.pkl).

Per unit mass, at each step's midpoint RHS evaluation (the one symplecticEuler
applies to v; the density rate is applied at v-bar by the drift path, the
difference is O(dt) per step and is reported as `residual`):

  kinetic   v.aWallP  (wall's own pressure P_j)
            v.aSelfW  (i's pressure x wall truncation, -(P_i/rho_i) sum_wall V_j gradW)
            v.aFluid, v.visc, v.g
  internal  (P_i/rho_i^2) drho for: wall pairs as integrated (mirrored wall velocity),
            fluid pairs, DDT
  adjoint   (P_i/rho_i^2) drhoW0 = -v.aSelfW identically (wall at rest), checked;
            the mirrored booking's excess over it, (P/rho^2)(drhoW - drhoW0), is the
            energy the wall pair creates. Split: normal (y) / tangential (x) using
            G = -rho_i aW1 = sum_wall V_j gradW:  drhoW0 = rho v.G.

  python scratchpad/ceiling_energy.py scripts/out_ceiling/forensics/s2_u80_rows.npz --uid 80
"""
import argparse
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('rows')
ap.add_argument('--uid', type=int, default=80)
ap.add_argument('--c0', type=float, default=None, help='sound speed (default: from P/(rho-rho0))')
ap.add_argument('--wall', action='store_true', help='english2025 rows next to the uid, per cycle')
ap.add_argument('--g', type=float, default=-9.81)
a = ap.parse_args()
d = np.load(a.rows)
dx, yC = float(d['dx']), float(d['yCeil'])
m = np.nonzero(d['uid'] == a.uid)[0]
calls, t = d['call'][m], d['t'][m]
# two RHS calls per step share the diagnostics time; keep the second (midpoint) one
_, first = np.unique(t, return_index=True)
keep = np.ones(m.size, bool)
keep[first] = False
# a step with a single watched call (e.g. the resume's first) has no midpoint
m = m[keep]
t = d['t'][m]
dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))  # the step that ENDS at t (diag time is after the step)
rho, P = d['rho'][m], d['P'][m]
v = d['v'][m]
rP = P / rho ** 2
aW1 = d['aW1'][m]
G = -rho[:, None] * aW1

terms = {
    'KE wallP':    (v * d['aWallP'][m]).sum(1),
    'KE selfW':    (v * d['aSelfW'][m]).sum(1),
    'KE fluid':    (v * (d['a'][m] - d['aWallP'][m] - d['aSelfW'][m])).sum(1),
    'KE visc':     (v * d['visc'][m]).sum(1),
    'KE grav':     v[:, 1] * a.g,
    'IE wall(as)': rP * d['drhoW'][m],
    'IE fluid':    rP * (d['drho'][m] - d['drhoW'][m]),
    'IE ddt':      rP * d['ddt'][m],
}
adjErr = terms['KE selfW'] + rP * d['drhoW0'][m]
excess = rP * (d['drhoW'][m] - d['drhoW0'][m])
exN = excess + rP * rho * v[:, 1] * G[:, 1] - rP * d['drhoW0'][m] + rP * rho * v[:, 0] * G[:, 0]  # placeholder, recomputed below
# excess = rP*(drhoW - rho v.G): normal part = rP*(drhoW - rho v_y G_y); tangential = -rP rho v_x G_x
exT = -rP * rho * v[:, 0] * G[:, 0]
exN = excess - exT
print(f'uid {a.uid}: {m.size} midpoint calls, t {t[0]:.4f}..{t[-1]:.4f}; adjoint identity |err| max '
      f'{np.abs(adjErr).max():.3g} vs |KE selfW| max {np.abs(terms["KE selfW"]).max():.3g}')

# state energy per mass (linear EOS): 1/2|v|^2 + c^2 (rho-rho0)^2/(2 rho0^2)  [e = int P/rho^2]
rho0 = 1.0
c2 = a.c0 ** 2 if a.c0 else np.median((P / (rho - rho0))[np.abs(rho - rho0) > 1e-3])
E = 0.5 * (v ** 2).sum(1) + c2 * (rho - rho0) ** 2 / (2 * rho0 ** 2) - a.g * d['y'][m]

# cycles: upward zero crossings of P
up = np.nonzero((P[:-1] < 0) & (P[1:] >= 0))[0] + 1
print(f'c^2 = {c2:.4g}  (c = {np.sqrt(c2):.2f});  per cycle, integrated rate x dt [J/kg]:')
hdr = ['k0', 't*', 'steps', '|P|max', 'nF', 's/dx', 'vx'] + list(terms) + ['excess', 'exN', 'exT', 'dE state', 'sum-res']
print(' '.join(f'{h:>10s}' for h in hdr))
ts = np.sqrt(9.81 / 0.6)
nF = d['nF'][m]
for c in range(len(up) - 1):
    s = slice(up[c], up[c + 1])
    I = {k: float((r[s] * dt[s]).sum()) for k, r in terms.items()}
    ex = float((excess[s] * dt[s]).sum()); en = float((exN[s] * dt[s]).sum()); et = float((exT[s] * dt[s]).sum())
    dE = float(E[up[c + 1]] - E[up[c]])
    tot = sum(I.values())
    row = [up[c], t[up[c]] * ts, up[c + 1] - up[c], np.abs(P[s]).max(), nF[s].mean(),
           ((yC - d['y'][m][s]) / dx).mean(), v[s, 0].mean()] + list(I.values()) + [ex, en, et, dE, dE - tot]
    print(' '.join(f'{x:10.4g}' for x in row))

if a.wall:
    W = pickle.load(open(a.rows.replace('_rows.npz', '_wall.pkl'), 'rb'))
    byCall = {w['call']: w for w in W}
    cl = d['call'][m]
    print('\nenglish2025 rows within 2.5 dx (x) of the uid, first/second layer; at each cycle start:')
    print('  call   x_i     buid    xb-xi/dx  yb    nNb  alpha     Pg      hydro      Pb')
    for c in range(0, len(up) - 1, max(1, (len(up) - 1) // 12)):
        k = up[c]
        w = byCall.get(int(cl[k]))
        if w is None:
            continue
        xi = d['x'][m][k]
        sel = np.nonzero((np.abs(w['bpos'][:, 0] - xi) < 2.5 * dx) & (w['bpos'][:, 1] < yC + 2 * dx))[0]
        for j in sel[np.argsort(w['bpos'][sel, 0])]:
            print(f"  {int(cl[k]):5d} {xi:+.3f} {w['buid'][j]:6d} {(w['bpos'][j,0]-xi)/dx:+6.2f} {w['bpos'][j,1]:.4f} "
                  f"{int(w['nNb'][j]):3d} {w['alpha'][j]:.5f} {w['Pg'][j]:+9.2f} {w['hydro'][j]:+8.3f} {w['Pb'][j]:+9.2f}")
        print()
