"""Print the force/continuity decomposition from ceiling_forensics.py for given UIDs."""
import argparse
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('rows')
ap.add_argument('--uid', type=int, nargs='+', required=True)
ap.add_argument('--every', type=int, default=0, help='print every Nth call (0 = ~50 lines)')
ap.add_argument('--calls', type=int, nargs=2, default=None)
ap.add_argument('--wall', action='store_true', help='also print english2025 wall rows near the uid')
a = ap.parse_args()
d = np.load(a.rows)
dx, yC = float(d['dx']), float(d['yCeil'])
for u in a.uid:
    m = np.nonzero(d['uid'] == u)[0]
    if a.calls:
        m = m[(d['call'][m] >= a.calls[0]) & (d['call'][m] <= a.calls[1])]
    if not m.size:
        print(f'uid {u}: not watched'); continue
    ev = a.every or max(1, m.size // 50)
    print(f'--- uid {u}  (a in m/s^2, y components; aFl = fluid-fluid pairs; drho: total / wall pairs / DDT)')
    print(' call     t*     s/dx   vy     rho      P     fs sw nF nW  lam | aP_y   aWallP aSelfW  aFl  | visc_y | drho    drhoW    ddt')
    for i in m[::ev]:
        aP = d['a'][i, 1]; aw = d['aWallP'][i, 1]; asw = d['aSelfW'][i, 1]
        print(f"{d['call'][i]:5d} {d['t'][i]*np.sqrt(9.81/0.6):7.4f} {(yC-d['y'][i])/dx:5.2f} {d['v'][i,1]:+6.2f} "
              f"{d['rho'][i]:.4f} {d['P'][i]:+8.1f} {d['fs'][i]} {int(d['sw'][i])} {int(d['nF'][i]):2d} {int(d['nW'][i]):2d} {d['lam'][i]:.2f} | "
              f"{aP:+7.1f} {aw:+7.1f} {asw:+7.1f} {aP-aw-asw:+7.1f} | {d['visc'][i,1]:+7.1f} | "
              f"{d['drho'][i]:+8.2f} {d['drhoW'][i]:+8.2f} {d['ddt'][i]:+7.2f}")
if a.wall:
    W = pickle.load(open(a.rows.replace('_rows.npz', '_wall.pkl'), 'rb'))
    print('wall rows (english2025): call  buid  bpos  nNb alpha  Pg  hydro  Pb  vb')
    for w in W[:: max(1, len(W) // 20)]:
        for k in range(min(len(w['buid']), 6)):
            print(f"  {w['call']:5d} {w['buid'][k]:6d} ({w['bpos'][k,0]:+.3f},{w['bpos'][k,1]:+.3f}) {int(w['nNb'][k]):2d} "
                  f"{w['alpha'][k]:.4f} {w['Pg'][k]:+8.2f} {w['hydro'][k]:+6.2f} {w['Pb'][k]:+8.2f} "
                  f"{'' if w['vb'] is None else np.round(w['vb'][k], 2)}")
