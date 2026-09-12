"""DELTASPH_VALIDATION_PLAN.md 5.13 open-item 2 / 5.11's open instrument: the
right-wall cluster event just after t=3.4s (UIDs 3792/4673/4006) takes a ~2x
velocity kick right as its fluid-neighbour count halves (17 -> 9 -> 18). 5.11
attributes this whole class ("SPH with too few neighbours") but nothing so far
has actually decomposed the acceleration into its terms at the moment of the
kick. `deltaSPH_step` sums
`dvdt = dvdt_pressure + dvdt_forcing + dvdt_gravity + dvdt_diss + dvdt_nopenshift`
(schemes/deltaSPH.py:249) -- this monkeypatches each of those five calls
in-process (no file edits) to capture their return values, then resumes the
checkpoint and steps through the event with a per-term breakdown per UID.
"""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--from-step', type=int, default=33000)
ap.add_argument('--steps', type=int, default=1200)
ap.add_argument('--every', type=int, default=20)
ap.add_argument('--uids', type=int, nargs='+', default=[3792, 4673, 4006])
ap.add_argument('--dir', default=None)
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
import warpSPH.schemes.deltaSPH as deltaSPHModule

# -- monkeypatch: wrap each term-producing call to stash its return value --
_captured = {}


def _wrap(name):
    orig = getattr(deltaSPHModule, name)

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        _captured[name] = out
        return out
    return wrapped


for fn in ('computeVelocityDiffusion', 'computePressureForceSurfaceAware',
           'computeForcing', 'computeGravity', 'computeMdbcNoPenShift'):
    setattr(deltaSPHModule, fn, _wrap(fn))

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'), key=os.path.getmtime)[-1]
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
ctx = r.ctx
cfg, scheme = ctx.config, ctx.schemeConfig
scheme.mdbcNoPenShiftMode = 'finalize'
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

# `running.adjacency` was built at t=0 (the fresh `run(..., nSteps=1)` init) and
# is never touched by the checkpoint graft above. deltaSPH_step's first call
# reads it as `priorNeighborhood` and only rebuilds if `_verlet_validity_metrics`
# says the buffer is exceeded -- computed via `_minimum_image_delta`, i.e.
# minimum-image (wrap-around) distance, and this domain is periodic=[True,True].
# If any particle's t=0-to-checkpoint displacement happens to look small under
# wrapping, the stale adjacency could be reused instead of rebuilt. Force a
# from-scratch build on the very first resumed step to rule this out.
if not args.__dict__.get('keepStaleAdjacency', False):
    running.adjacency = None
    print('cleared stale t=0 adjacency -- forcing a from-scratch rebuild on the first resumed step')
print(f'resumed state_{args.from_step:04d}  t={t0:.5f}  particles={particles.positions.shape[0]}')
print(f'mdbcNoPenShiftMode = {scheme.mdbcNoPenShiftMode}')

state = running
header = (f'{"step":>6} {"t":>7} {"uid":>5} {"|v|":>7} {"rho":>8} {"nFl":>4}  '
          f'{"pressure":>9} {"diss":>9} {"forcing":>9} {"gravity":>9} {"nopen":>9} {"TOTAL":>9}')
print(header)

for k in range(args.steps):
    step = args.from_step + k
    parts = getattr(state, 'state', state)

    stepResult = ctx.integrator.function(
        state=state, f=ctx.stepFunction, dt=cfg.dt,
        config=cfg, verbose=False, schemeConfig=scheme)
    newState = stepResult.state
    if sloshingTankCase.postStep is not None:
        sloshingTankCase.postStep(ctx, newState, step)

    if k % args.every == 0 or k == args.steps - 1:
        for uid in args.uids:
            m = (parts.UIDs == uid).nonzero()
            if m.numel() == 0:
                continue
            i = int(m[0])
            newParts = getattr(newState, 'state', newState)
            v = newParts.velocities[i].detach().cpu().numpy()
            xi = parts.positions[i]
            nFl = int(((parts.positions[parts.kinds == 0] - xi).norm(dim=1)
                       < float(parts.supports[i])).sum()) - 1
            tNow = float(getattr(newParts, 't', t0 + (k + 1) * float(cfg.dt)))

            def termNorm(name):
                arr = _captured.get(name)
                if arr is None:
                    return float('nan')
                return float(arr[i].norm())

            pT = termNorm('computePressureForceSurfaceAware')
            dT = termNorm('computeVelocityDiffusion')
            # computeForcing returns a *force*, not acceleration -- divide by mass to match dvdt units
            fT = (float(_captured['computeForcing'][i].norm() / parts.masses[i])
                  if _captured.get('computeForcing') is not None else float('nan'))
            gT = termNorm('computeGravity')
            # only populated under mdbcNoPenShiftMode == 'derivative'; under
            # 'finalize' the correction is applied outside this step function
            nT = (float(_captured['computeMdbcNoPenShift'][i].norm() / cfg.dt)
                  if _captured.get('computeMdbcNoPenShift') is not None else 0.0)
            upd = stepResult.stages[-1].update if stepResult.stages else None
            total = float(upd.dvdt[i].norm()) if (upd is not None and getattr(upd, 'dvdt', None) is not None) else float('nan')

            print(f'{step:6d} {tNow:7.4f} {uid:5d} {float(np.linalg.norm(v)):7.3f} '
                  f'{float(newParts.densities[i]):8.5f} {nFl:4d}  '
                  f'{pT:9.3g} {dT:9.3g} {fT:9.3g} {gT:9.3g} {nT:9.3g} {total:9.3g}')
    state = newState

    if not torch.isfinite(getattr(state, 'state', state).densities).all():
        print(f'--- non-finite density at step {step} ---')
        break
