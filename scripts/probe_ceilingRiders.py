"""List the ceiling riders of a stored ACSPH run (`--store` checkpoints) and what holds them.

A rider is a fluid row with a wall row within 1.5 dx on its gravity-up side
(the `probe_contactLine` definition). For each rider and checkpoint it prints
uid, own pressure / speed / density, fluid and wall neighbour counts, the wall
rows' pressures above it and the wall-fluid column [nF of the wall rows' fluid
neighbours], so a rider can be classed as: lone (nF = 0), pair/clump, or a
thin film attached to a bulk.

With `--balance` (needs frames), also the vertical force balance on each thin /
lone rider recomputed from the stored state: the pressure acceleration
`-V sum_j (p_i + p_j) gradW_ij` split into wall and fluid partners (wall
pressure by Eq. (61) from the stored fluid pressures, Wendland C2, `supports`
as the kernel radius) against gravity. `sum_y ~ 0` with `v_y ~ 0` = held by the
wall's tension; `sum_y << 0` = falling off.

Usage: python -I scripts/probe_ceilingRiders.py [--balance] <trajectoryDir> [frame ...]
"""
import glob
import os
import sys

import h5py
import numpy as np


def load(path):
    h = h5py.File(path, 'r')
    s = h['state']
    d = {k: s[k][()] for k in ('UIDs', 'kinds', 'positions', 'velocities', 'pressures', 'densities',
                               'supports', 'surfaceLambdas', 'surfaceIndicators')}
    d['t'] = float(h.attrs['time'])
    d['dx'] = float(h.attrs['dx'])
    d['g'] = float(h.attrs['gravityMagnitude'])
    return d


def riders(d):
    fluid = d['kinds'] == 0
    wall = d['kinds'] == 1
    x = d['positions']
    fi, wi = np.nonzero(fluid)[0], np.nonzero(wall)[0]
    up = np.array([0.0, 1.0])
    dv = x[wi][None, :, :] - x[fi][:, None, :]
    dUp = dv @ up
    dSide = np.linalg.norm(dv - dUp[..., None] * up, axis=-1)
    close = np.linalg.norm(dv, axis=-1) < 1.5 * d['dx']
    above = (dUp > 0.5 * d['dx']) & (dSide < 0.75 * d['dx'])
    return fi[(close & above).any(axis=1)]


def describe(d, rows):
    x, p, v = d['positions'], d['pressures'], d['velocities']
    fluid = d['kinds'] == 0
    wall = d['kinds'] == 1
    out = []
    for i in rows:
        r = np.linalg.norm(x - x[i], axis=1)
        inSup = r < d['supports'][i]
        nF = int((inSup & fluid).sum()) - 1
        nW = int((inSup & wall).sum())
        pw = p[inSup & wall]
        out.append(dict(uid=int(d['UIDs'][i]), x=x[i].round(3).tolist(), p=float(p[i]), speed=float(np.linalg.norm(v[i])),
                        vy=float(v[i, 1]), rho=float(d['densities'][i]), lam=float(d['surfaceLambdas'][i]),
                        nF=nF, nW=nW, pwMin=float(pw.min()) if pw.size else 0.0, pwMean=float(pw.mean()) if pw.size else 0.0))
    return out


def wendland2(r, H):
    q = np.clip(r / H, 0, 1)
    return 7 / (np.pi * H ** 2) * (1 - q) ** 4 * (4 * q + 1)


def dWendland2(r, H):
    q = np.clip(r / H, 0, 1)
    return 7 / (np.pi * H ** 3) * (-20 * q) * (1 - q) ** 3


def balance(d, rows, info):
    x, p = d['positions'].astype(np.float64), d['pressures'].astype(np.float64)
    k, H = d['kinds'], d['supports'].astype(np.float64)
    V, g = d['dx'] ** 2, np.array([0.0, -d['g']])
    fl = np.nonzero(k == 0)[0]

    def wallPressure(w):                     # De Courcy 2024 Eq. (61)
        r = np.linalg.norm(x[fl] - x[w], axis=1)
        m = r < H[w]
        ww, f = wendland2(r[m], H[w]) * V, fl[m]
        if ww.sum() < 1e-9:
            return 0.0
        return ((p[f] * ww).sum() + (ww * ((x[w] - x[f]) @ g)).sum()) / ww.sum()

    for i, r in zip(rows, info):
        if r['nF'] > 4:
            continue
        rr = np.linalg.norm(x - x[i], axis=1)
        ayW = ayF = 0.0
        for j in np.nonzero((rr < H[i]) & (rr > 0))[0]:
            if k[j] == 2:
                continue
            pj = p[j] if k[j] == 0 else wallPressure(j)
            ay = (-V * (p[i] + pj) * dWendland2(rr[j], H[i]) * (x[i] - x[j]) / rr[j])[1]
            if k[j] == 0:
                ayF += ay
            else:
                ayW += ay
        print(f"    uid {r['uid']:5d} nF={r['nF']} p={r['p']:+.3f} vy={d['velocities'][i][1]:+.3f} "
              f"a_wall_y={ayW:+.2f} a_fluid_y={ayF:+.2f} sum_y={ayW + ayF - d['g']:+.2f}")


def main():
    args = [a for a in sys.argv[1:] if a != '--balance']
    doBalance = '--balance' in sys.argv
    traj = args[0]
    frames = [int(a) for a in args[1:]]
    files = sorted(glob.glob(os.path.join(traj, 'state_*.h5')))
    for f in files:
        n = int(os.path.basename(f)[6:10])
        if frames and n not in frames:
            continue
        d = load(f)
        rows = riders(d)
        info = describe(d, rows)
        lone = sum(1 for r in info if r['nF'] == 0)
        thin = sum(1 for r in info if 0 < r['nF'] <= 4)
        print(f'frame {n:5d} t={d["t"]:.3f}: {len(info)} riders  lone(nF=0)={lone}  thin(1<=nF<=4)={thin}  '
              f'bulk(nF>4)={len(info) - lone - thin}')
        if frames and not doBalance:
            for r in sorted(info, key=lambda r: r['nF']):
                print('   ', r)
        if doBalance:
            balance(d, rows, info)


if __name__ == '__main__':
    main()
