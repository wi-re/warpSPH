"""Instrument the sloshingTank ceiling-hover particle (UID 4104) with the
pressures / free-surface mask the Antuono switch actually sees, to check
whether DELTASPH_VALIDATION_PLAN.md 5.14's fix (dropping the `or P_j>=0.0`
branch from PressureForceScheme.Antuono) changes anything for this specific
reproduction. Same resume mechanics as probe_slosh_resume.py, plus P_i, the
nearest boundary/ghost neighbour's P_j, and mask_i printed each step.
"""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--from-step', type=int, default=60000)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--uid', type=int, default=4104)
ap.add_argument('--every', type=int, default=25)
ap.add_argument('--dir', default=None)
ap.add_argument('--noPenShift', default='finalize')
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

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'), key=os.path.getmtime)[-1]
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
ctx = r.ctx
cfg, scheme = ctx.config, ctx.schemeConfig
if args.noPenShift is not None:
    scheme.mdbcNoPenShiftMode = args.noPenShift
print('mdbcNoPenShiftMode =', getattr(scheme, 'mdbcNoPenShiftMode', '?'))
print('pressureForceTerm =', getattr(scheme, 'pressureForceTerm', '?'))
dev = cfg.domain.min.device

f = os.path.join(runDir, 'trajectory', f'state_{args.from_step:04d}.h5')
h = h5py.File(f, 'r')
ckpt = loadState(h['state'], dev, WeaklyCompressibleState)
t0 = float(h.attrs['time'])

import dataclasses
running = r.state
particles = getattr(running, 'state', running)
assert particles.positions.shape[0] == ckpt.positions.shape[0], 'particle count mismatch'
for fl in dataclasses.fields(ckpt):
    v = getattr(ckpt, fl.name, None)
    if torch.is_tensor(v):
        setattr(particles, fl.name, v.clone())
    elif v is not None and fl.name != 't':
        setattr(particles, fl.name, v)
particles.t = torch.tensor(t0, device=dev, dtype=particles.positions.dtype)
if hasattr(running, 't'):
    running.t = particles.t
print(f'resumed {os.path.basename(f)}  t={t0:.5f}  particles={particles.positions.shape[0]}')

nhat = np.array([0.0, -1.0])
that = np.array([1.0, 0.0])

print(f'{"step":>6} {"t":>8} {"y":>9} {"v_norm":>10} {"rho_i":>8} {"P_i":>9} '
      f'{"maskI":>5} {"nearestBnd_rho":>14} {"nearestBnd_P":>12} {"dvdt_n":>10}')

state = running
for k in range(args.steps):
    step = args.from_step + k
    parts = getattr(state, 'state', state)
    m = (parts.UIDs == args.uid).nonzero()
    i = int(m[0]) if m.numel() else -1

    # snapshot pressures/mask BEFORE the step overwrites them, plus nearest
    # boundary/ghost neighbour's density/pressure (kinds 1 or 2)
    Pi = float(parts.pressures[i]) if (i >= 0 and getattr(parts, 'pressures', None) is not None) else float('nan')
    maskI = int(parts.surfaceIndicators[i]) if (i >= 0 and getattr(parts, 'surfaceIndicators', None) is not None) else -1
    rhoI = float(parts.densities[i]) if i >= 0 else float('nan')
    nearRho = float('nan'); nearP = float('nan')
    if i >= 0:
        xi = parts.positions[i]
        bndMask = parts.kinds != 0
        if bndMask.any():
            d = (parts.positions[bndMask] - xi).norm(dim=1)
            j = int(torch.argmin(d))
            bIdx = bndMask.nonzero()[j]
            nearRho = float(parts.densities[bIdx])
            if getattr(parts, 'pressures', None) is not None:
                nearP = float(parts.pressures[bIdx])

    stepResult = ctx.integrator.function(
        state=state, f=ctx.stepFunction, dt=cfg.dt,
        config=cfg, verbose=False, schemeConfig=scheme)
    newState = stepResult.state
    if sloshingTankCase.postStep is not None:
        sloshingTankCase.postStep(ctx, newState, step)

    if i >= 0 and (k % args.every == 0 or k == args.steps - 1):
        upd = stepResult.stages[-1].update if stepResult.stages else None
        dvdt = upd.dvdt[i].detach().cpu().numpy() if (upd is not None and
                                                      getattr(upd, 'dvdt', None) is not None) \
            else np.zeros(2)
        xi = parts.positions[i]
        v = getattr(newState, 'state', newState).velocities[i].detach().cpu().numpy()
        tNow = float(getattr(getattr(newState, 'state', newState), 't', t0 + (k + 1) * float(cfg.dt)))
        print(f'{step:6d} {tNow:8.4f} {float(xi[1]):9.5f} {float(np.dot(v, nhat)):10.4f} '
              f'{rhoI:8.5f} {Pi:9.4f} {maskI:5d} {nearRho:14.5f} {nearP:12.4f} '
              f'{float(np.dot(dvdt, nhat)):10.4g}')
    state = newState

    if not torch.isfinite(getattr(state, 'state', state).densities).all():
        print(f'--- non-finite density at step {step} ---')
        break
