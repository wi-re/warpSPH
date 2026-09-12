"""Track ceiling residents and the right-wall event in the clean `finalize` run.

A hovering particle does not accelerate, so a speed filter misses it. Its
signature is instead: sitting in the ceiling band, detached from the fluid, with
`y` not changing over many checkpoints while gravity points away from the wall.
"""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=None)
ap.add_argument('--what', default='hover', choices=('hover', 'rightwall'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import h5py
import numpy as np
import torch

from warpSPH.io.hdf5 import loadState
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
from warpSPH.cases.sloshingTank import sloshingTankCase
from warpSPH.runner import run

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'),
                            key=os.path.getmtime)[-1]
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
cfg = r.ctx.config
dev = cfg.domain.min.device
dx = float(cfg.dx)

files = sorted(glob.glob(os.path.join(runDir, 'trajectory', 'state_*.h5')),
               key=lambda p: int(os.path.basename(p)[6:-3]))


def load(f):
    h = h5py.File(f, 'r')
    return loadState(h['state'], dev, WeaklyCompressibleState), float(h.attrs['time'])


if args.what == 'hover':
    # ceiling residents in the last checkpoint, whatever their speed
    st, t = load(files[-1])
    fl = st.kinds == 0
    idx = torch.where(fl & (st.positions[:, 1] > 0.498))[0]
    print(f'final checkpoint t={t:.3f}: {len(idx)} fluid particles with y > 0.498')
    for i in idx.tolist():
        nFl = int(((st.positions[fl] - st.positions[i]).norm(dim=1)
                   < float(st.supports[i])).sum()) - 1
        print(f'  UID {int(st.UIDs[i]):6d} y={float(st.positions[i,1]):.5f} '
              f'x={float(st.positions[i,0]):+.5f} |v|={float(st.velocities[i].norm()):7.4f} '
              f'v=({float(st.velocities[i,0]):+.4f},{float(st.velocities[i,1]):+.4f}) '
              f'rho={float(st.densities[i]):.5f} nFluidNb={nFl}')
    uids = [int(st.UIDs[i]) for i in idx.tolist()][:4]
    print(f'\ntracing {uids} back through checkpoints:')
    print(f'{"t":>7} ' + ' '.join(f'{"UID"+str(u):>26}' for u in uids))
    for f in files[::4]:
        s, tt = load(f)
        cells = []
        for u in uids:
            m = (s.UIDs == u).nonzero()
            if m.numel() == 0:
                cells.append(f'{"-":>26}'); continue
            i = int(m[0])
            cells.append(f'y={float(s.positions[i,1]):7.5f} vy={float(s.velocities[i,1]):+7.4f}'
                         .rjust(26))
        print(f'{tt:7.3f} ' + ' '.join(cells))

else:
    # the right-wall event: local detail in the 3.3-3.6 window
    for f in files:
        step = int(os.path.basename(f)[6:-3])
        if not (33000 <= step <= 36000):
            continue
        st, t = load(f)
        fl = st.kinds == 0
        p, v = st.positions[fl], st.velocities[fl]
        vm = v.norm(dim=1)
        band = p[:, 0] > 0.43                     # within ~9 dx of the right wall
        if int(band.sum()) == 0:
            print(f't={t:.3f}  (no fluid in the right-wall band)'); continue
        vb = vm[band]
        bulk = float(torch.median(vm))
        k = int(torch.argmax(vb))
        pb = p[band]
        print(f't={t:.3f}  wall-band n={int(band.sum()):4d}  |v| med {float(torch.median(vb)):6.3f} '
              f'max {float(vb.max()):7.3f} at y={float(pb[k,1]):.4f}  '
              f'(bulk median {bulk:.3f}, ratio {float(vb.max())/max(bulk,1e-9):6.1f}x)  '
              f'rhoMin_band {float(st.densities[fl][band].min()):.5f}')
