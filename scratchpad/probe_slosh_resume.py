"""Resume from a checkpoint and step forward with per-step instrumentation on
the particle that blows up (UID 4804), to find which term supplies its
into-the-ceiling velocity.

Between steps 43500 and 44000 it goes from v = (+1.32, +0.012) to
(+1.20, +20.11) -- 20 m/s *into* the ceiling -- while its position never
moves. Gravity points away from the ceiling and the wall particles are static
(the roll is applied as a rotating gravity vector, not tank motion), so
something else supplies it. This records, every step: position, velocity
decomposed on the wall normal, rho, the no-penetration shift the term would
apply, and the scheme's own dvdt.
"""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--from-step', type=int, default=43500)
ap.add_argument('--steps', type=int, default=520)
ap.add_argument('--uid', type=int, default=4804)
ap.add_argument('--every', type=int, default=10)
ap.add_argument('--dir', default=None, help='checkpointed run dir')
ap.add_argument('--noPenShift', default=None,
                help="override mdbcNoPenShiftMode after resuming: derivative|finalize|off")
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
from warpSPH.modules.mdbc import computeMdbcNoPenShift
from warpSPHCore import SupportScheme, buildVerletList

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'), key=os.path.getmtime)[-1]
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
ctx = r.ctx
cfg, scheme = ctx.config, ctx.schemeConfig
if args.noPenShift is not None:
    scheme.mdbcNoPenShiftMode = args.noPenShift
print('mdbcNoPenShiftMode =', getattr(scheme, 'mdbcNoPenShiftMode', '?'))
dev = cfg.domain.min.device

f = os.path.join(runDir, 'trajectory', f'state_{args["from_step"] if False else args.from_step:04d}.h5')
h = h5py.File(f, 'r')
ckpt = loadState(h['state'], dev, WeaklyCompressibleState)
t0 = float(h.attrs['time'])

# The integrator takes the *system* wrapper (it calls `.initialize`), with the
# particle arrays at `.state`; graft the checkpoint into the live system from
# the fresh setup so all the scheme plumbing stays intact.
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
st = particles
print(f'resumed {os.path.basename(f)}  t={t0:.5f}  particles={st.positions.shape[0]}')

nhat = np.array([0.0, -1.0])   # ceiling inward normal (into the fluid)
that = np.array([1.0, 0.0])

print(f'{"step":>6} {"t":>8} {"y":>9} {"v_tan":>9} {"v_norm":>10} {"rho":>8} '
      f'{"nopen_n":>10} {"dvdt_n":>11} {"nFl":>4}')

state = running
for k in range(args.steps):
    step = args.from_step + k
    # what the no-penetration term would apply at this state
    parts = getattr(state, 'state', state)
    adj = buildVerletList(parts, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    try:
        nopen = computeMdbcNoPenShift(parts, cfg, scheme, adj)
    except Exception:
        nopen = torch.zeros_like(parts.velocities)

    m = (parts.UIDs == args.uid).nonzero()
    i = int(m[0]) if m.numel() else -1

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
        v = parts.velocities[i].detach().cpu().numpy()
        xi = parts.positions[i]
        nFl = int(((parts.positions[parts.kinds == 0] - xi).norm(dim=1)
                   < float(parts.supports[i])).sum()) - 1
        npn = float(np.dot(nopen[i].detach().cpu().numpy(), nhat))
        tNow = float(getattr(parts, 't', t0 + k * float(cfg.dt)))
        print(f'{step:6d} {tNow:8.4f} {float(xi[1]):9.5f} '
              f'{float(np.dot(v, that)):9.4f} {float(np.dot(v, nhat)):10.4f} '
              f'{float(parts.densities[i]):8.5f} {npn:10.3f} '
              f'{float(np.dot(dvdt, nhat)):11.4g} {nFl:4d}')
    state = newState

    if not torch.isfinite(getattr(state,'state',state).densities).all():
        print(f'--- non-finite density at step {step} ---')
        break
