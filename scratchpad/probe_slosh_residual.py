"""Scan the clean (`finalize`) sloshingTank checkpoints for the two residual
issues: a particle hovering along the ceiling late in the run, and a cluster
ejected off the right wall just after t = 3.4 s.
"""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=None)
ap.add_argument('--mode', default='scan', choices=('scan', 'ceiling', 'rightwall'))
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
print(f'{runDir}   {len(files)} checkpoints   dx={dx:.5g}')

# tank extent from the boundary band
h0 = h5py.File(files[0], 'r')
s0 = loadState(h0['state'], dev, WeaklyCompressibleState)
b = s0.positions[s0.kinds == 1]
xMin, xMax = float(b[:, 0].min()), float(b[:, 0].max())
yMin, yMax = float(b[:, 1].min()), float(b[:, 1].max())
print(f'tank band: x [{xMin:.4f},{xMax:.4f}]  y [{yMin:.4f},{yMax:.4f}]')
ceilY = yMax - 6 * dx      # within 6dx of the top band
rightX = xMax - 6 * dx

if args.mode == 'scan':
    print(f'\n{"step":>7} {"t":>7} {"|v|max":>9} {"fastest pos":>22} {"nCeil":>6} '
          f'{"nRight":>7} {"rhoMin":>8}')
    for f in files:
        step = int(os.path.basename(f)[6:-3])
        h = h5py.File(f, 'r')
        st = loadState(h['state'], dev, WeaklyCompressibleState)
        fl = st.kinds == 0
        v = st.velocities[fl].norm(dim=1)
        p = st.positions[fl]
        j = int(torch.argmax(v))
        # fluid sitting in the ceiling band / right-wall band, moving fast
        nCeil = int(((p[:, 1] > ceilY) & (v > 1.0)).sum())
        nRight = int(((p[:, 0] > rightX) & (v > 3.0)).sum())
        print(f'{step:7d} {float(h.attrs["time"]):7.3f} {float(v.max()):9.3f} '
              f'({float(p[j,0]):8.4f},{float(p[j,1]):7.4f}) {nCeil:6d} {nRight:7d} '
              f'{float(st.densities[fl].min()):8.5f}')
