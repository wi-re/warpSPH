"""CEILING_STICKING_PLAN.md §7: per-step *discrete* energy budget of one watched
particle, from ceiling_forensics.py's <tag>_rows.npz (stage-1 = midpoint RHS
terms) and <tag>_fin.npz (state at step start, after the integrator, after
finalize; the PST displacement; the noPen correction).

E per mass = 1/2|v|^2 + c^2 (rho-rho0)^2 / (2 rho0^2) - g.y   (linear EOS, e = int P/rho^2)

  integrator  E(pre) - E(start):  per term dt vbar.a_term (exact for a kick with
              the stage-1 acceleration), internal c^2 (rhobar-rho0) drho / rho0^2
              (exact); 'int res' = what the terms miss (drift-vs-kick mismatch)
  selfW+wallC dt vbar.aSelfW + c^2(rhobar-rho0) drho: the wall pair's net energy
              (drho is all wall-pair for a lone row: nF = 0, DDT = 0)
  noPen       E(post) - E(pre) on steps where the correction fired
  shift       -(P/rho) G.dx_shift: the change of internal energy the PST move
              should have booked (rho = rho(x) at the wall, drho = rho G.dx) and did not

  python scratchpad/ceiling_energy_step.py scripts/out_ceiling/forensics/s2_u80b --uid 80
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('prefix')
ap.add_argument('--uid', type=int, default=80)
ap.add_argument('--c0', type=float, default=97.04)
ap.add_argument('--g', type=float, default=-9.81)
ap.add_argument('--steps', action='store_true', help='print every step, not per cycle')
a = ap.parse_args()
d = np.load(a.prefix + '_rows.npz')
F = np.load(a.prefix + '_fin.npz')
c2, rho0 = a.c0 ** 2, 1.0
dx, yC = float(d['dx']), float(d['yCeil'])

m = np.nonzero(d['uid'] == a.uid)[0]
t = d['t'][m]
_, first = np.unique(t, return_index=True)
keep = np.ones(m.size, bool); keep[first] = False
m = m[keep]
col = np.nonzero(F['uid'][0] == a.uid)[0][0]
f = {k: (F[k][:, col] if F[k].ndim > 1 else F[k]) for k in F.files}
n = min(m.size, f['dt'].size)
m = m[:n]
f = {k: v[:n] for k, v in f.items()}
dt = f['dt']


def E(v, rho, x):
    return 0.5 * (v ** 2).sum(-1) + c2 * (rho - rho0) ** 2 / (2 * rho0 ** 2) - a.g * x[..., 1]


vbar = 0.5 * (f['v0'] + f['vPre'])
rbar = 0.5 * (f['rho0'] + f['rhoPre'])
drho = f['rhoPre'] - f['rho0']
aS, aP, aT, vi = d['aSelfW'][m], d['aWallP'][m], d['a'][m], d['visc'][m]
k = {
    'wallP': dt * (vbar * aP).sum(1),
    'selfW': dt * (vbar * aS).sum(1),
    'fluid': dt * (vbar * (aT - aP - aS)).sum(1),
    'visc': dt * (vbar * vi).sum(1),
    'grav': dt * vbar[:, 1] * a.g,
    'IE': c2 * (rbar - rho0) * drho / rho0 ** 2,
}
dEint = E(f['vPre'], f['rhoPre'], f['xPre']) - E(f['v0'], f['rho0'], f['x0'])
# E includes the potential -g.y; its change over the step cancels the 'grav' work
k['int res'] = dEint - sum(k.values()) - (-a.g) * (f['xPre'][:, 1] - f['x0'][:, 1])
fired = np.abs(f['nopen']).sum(1) > 0
k['noPen'] = np.where(fired, E(f['vPost'], f['rhoPost'], f['xPost']) - E(f['vPre'], f['rhoPre'], f['xPre']), 0.0)
P, rho = d['P'][m], d['rho'][m]
G = -rho[:, None] * d['aW1'][m]
k['shift'] = -(P / rho) * (G * f['shiftDx']).sum(1)
k['fin other'] = E(f['vPost'], f['rhoPost'], f['xPost']) - E(f['vPre'], f['rhoPre'], f['xPre']) - k['noPen']
Etot = E(f['vPost'], f['rhoPost'], f['xPost']) - E(f['v0'], f['rho0'], f['x0'])
pair = k['selfW'] + k['IE']

ts = np.sqrt(9.81 / 0.6)
print(f'uid {a.uid}: {n} steps; noPen fired on {fired.sum()} steps; |shift| max {np.abs(f["shiftDx"]).max()/dx:.3g} dx')
cols = list(k) + ['pair', 'dE step']
if a.steps:
    print(' step    t*    s/dx    vy      P     nop ' + ' '.join(f'{c:>9s}' for c in cols))
    for i in range(n):
        print(f'{i:5d} {t[0]*0 + d["t"][m][i]*ts:7.4f} {(yC-f["xPre"][i,1])/dx:5.2f} {f["vPost"][i,1]:+6.2f} {P[i]:+7.1f} {int(fired[i])} '
              + ' '.join(f'{k[c][i]:+9.2e}' for c in k) + f' {pair[i]:+9.2e} {Etot[i]:+9.2e}')
else:
    up = np.nonzero((P[:-1] < 0) & (P[1:] >= 0))[0] + 1
    print('   k0     t*  steps |P|max nop ' + ' '.join(f'{c:>9s}' for c in cols))
    for c in range(len(up) - 1):
        s = slice(up[c], up[c + 1])
        print(f'{up[c]:5d} {d["t"][m][up[c]]*ts:6.3f} {up[c+1]-up[c]:4d} {np.abs(P[s]).max():6.1f} {fired[s].sum():3d} '
              + ' '.join(f'{k[q][s].sum():+9.2e}' for q in k) + f' {pair[s].sum():+9.2e} {Etot[s].sum():+9.2e}')
