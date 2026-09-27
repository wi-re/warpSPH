"""Interrogate the sloshingTank blow-up in situ, from checkpoints.

The run dies at step 45,726 (t = 4.5726). Checkpoints every 500 steps let us
open the state just before that and ask what the offending particles actually
see -- their own rho/p, how many fluid vs boundary neighbours they have, and
what the wall particles next to them are reading (rho_b, numNeighbors, and the
MLS/Shepard blend weight w) -- instead of inferring a mechanism and paying a
1.5 h run per guess.

    python scratchpad/probe_slosh_blowup.py [--dir export/16-sloshingTank-...]
"""
import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=None, help='run export dir (default: newest sloshing run)')
ap.add_argument('--steps', default='43500,44500,45000,45500',
                help='checkpoint steps to inspect, comma separated')
ap.add_argument('--top', type=int, default=6, help='how many outlier particles to detail')
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
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.modules.liu import interpolateLiuLiu, determinantThresholdFor
from warpSPH.configurations.region import RegionType
from warpSPHCore import (SupportScheme, buildVerletList, OperationDirection)

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'),
                            key=os.path.getmtime)[-1]
print(f'run dir: {runDir}')

# A fresh one-step setup supplies a valid config / schemeConfig / regions for
# this resolution; the checkpoint then replaces the state arrays. (Going
# through `importSimulationSystem` instead trips a scheme-name collision:
# `deltaSPH` resolves to `CompressibleSPHScheme` before the weakly-compressible
# enum, so its stage loader builds the wrong Update class.)
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
cfg, scheme = r.ctx.config, r.ctx.schemeConfig
rho0 = float(scheme.fluid.restDensity)
c0 = float(scheme.fluid.fixedSoundSpeed)
print(f'rho0={rho0}  c0={c0:.4g}  dx={float(cfg.dx):.5g}')


def loadStep(step):
    f = os.path.join(runDir, 'trajectory', f'state_{step:04d}.h5')
    if not os.path.exists(f):
        return None, None
    h = h5py.File(f, 'r')
    st = loadState(h['state'], cfg.domain.min.device, WeaklyCompressibleState)
    return st, float(h.attrs['time'])


def analyse(st, t, step):
    fluid = st.kinds == 0
    bnd = st.kinds == 1
    v = st.velocities.norm(dim=1)
    rho = st.densities

    print(f'\n{"="*74}\nstep {step}  t={t:.4f}')
    print(f'  fluid |v|: mean {float(v[fluid].mean()):.4f}  p99 '
          f'{float(torch.quantile(v[fluid], 0.99)):.4f}  max {float(v[fluid].max()):.4f}')
    print(f'  fluid rho: [{float(rho[fluid].min()):.5f}, {float(rho[fluid].max()):.5f}]  '
          f'p01 {float(torch.quantile(rho[fluid], 0.01)):.5f}')

    # recompute the mDBC wall reading and its conditioning at this state
    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    merged = computeMdbcDensity(st, cfg, scheme, adj)
    _, _, nNb, A_g, _, _ = interpolateLiuLiu(
        st.positions[st.kinds == 2], referenceParticles=st,
        referenceQuantities=st.densities, config=cfg, neighbor_threshold=4,
        direction=OperationDirection.FluidToGhost, supportScale=1.0,
        adjacency=adj.hashMap if hasattr(adj, 'hashMap') else None)
    det = torch.linalg.det(A_g).abs()
    detFloor = determinantThresholdFor(cfg.kernel)
    wDet = torch.clamp((det - detFloor) / (detFloor * 0.25), 0.0, 1.0)
    wN = torch.clamp((nNb.to(det.dtype) - 4.0) / 1.0, 0.0, 1.0)
    w = wDet * wN
    rho_b = merged[bnd]
    p_b = c0 ** 2 * (rho_b - rho0)
    print(f'  wall rho_b: [{float(rho_b.min()):.5f}, {float(rho_b.max()):.5f}]   '
          f'|p_b| max {float(p_b.abs().max()):.4g}   w>0 on {int((w > 0).sum())}/{w.numel()}')

    # the fluid outliers, by speed
    fi = torch.where(fluid)[0]
    order = torch.argsort(v[fi], descending=True)[:args.top]
    bpos = st.positions[bnd]
    print(f'  --- top {args.top} fastest fluid particles ---')
    for k in order.tolist():
        i = int(fi[k])
        xi = st.positions[i]
        d = (bpos - xi).norm(dim=1)
        near = torch.argsort(d)[:8]
        nFluidNb = int(((st.positions[fluid] - xi).norm(dim=1) < float(st.supports[i])).sum()) - 1
        nBndNb = int((d < float(st.supports[i])).sum())
        print(f'   UID {int(st.UIDs[i]):6d}  |v|={float(v[i]):8.3f}  rho={float(rho[i]):.5f}  '
              f'p={c0**2*(float(rho[i])-rho0):9.2f}  pos=({float(xi[0]):.4f},{float(xi[1]):.4f})')
        print(f'        neighbours: {nFluidNb} fluid, {nBndNb} boundary   '
              f'nearest wall {float(d[near[0]])/float(cfg.dx):.2f} dx')
        print(f'        nearest walls rho_b: '
              + ' '.join(f'{float(rho_b[j]):.4f}' for j in near[:5]))
    return st


states = {}
for s in [int(x) for x in args.steps.split(',')]:
    st, t = loadStep(s)
    if st is None:
        print(f'(step {s} has no checkpoint)')
        continue
    states[s] = analyse(st, t, s)

# trace the eventual culprit backwards by UID
if states:
    last = max(states)
    st = states[last]
    fluid = st.kinds == 0
    v = st.velocities.norm(dim=1)
    fi = torch.where(fluid)[0]
    worst = int(fi[int(torch.argmax(v[fi]))])
    uid = int(st.UIDs[worst])
    print(f'\n{"="*74}\nhistory of UID {uid} (fastest at step {last})')
    print(f'{"step":>7} {"t":>8} {"|v|":>10} {"rho":>9} {"pos":>22}')
    for s in sorted(states):
        sst = states[s]
        m = (sst.UIDs == uid).nonzero()
        if m.numel() == 0:
            print(f'{s:7d}  (UID absent)')
            continue
        i = int(m[0])
        p = sst.positions[i]
        print(f'{s:7d} {"":>8} {float(sst.velocities[i].norm()):10.3f} '
              f'{float(sst.densities[i]):9.5f} ({float(p[0]):9.4f},{float(p[1]):8.4f})')
