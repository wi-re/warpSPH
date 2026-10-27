"""Probe (WALL_CONTACT_PLAN W3 / OPEN_PROBLEMS §9): does a uniform pressure level grow in
a **periodic** ACSPH box (no walls, no free surface)?

In the sealed box (`probe_contactLine.py --toy sealed --p0 10`) a uniform level of +10 grows
exponentially (25 -> 75 -> 201 -> 513 -> 1290 over t 0.06-0.38, doubling ~ every 0.05 s).
Growth in a periodic box points at the dual-time solve itself (nothing anchors the level);
no growth points at the wall closure (Eq. 61 + the `(p_i + p_j)` force against walls).

The box is the Taylor-Green vortex case with `uMag` scaled down (`--uMag 0` is a box at
rest; the case's own pressure field is `-(rho0 uMag^2 / 4)(cos 2kx + cos 2ky)`, tiny for a small
uMag, and is replaced by the uniform level `--p0`). Every step records mean / min / max fluid
pressure and |v|max; the level is the quantity of interest.

Usage:
  python scripts/probe_periodicLevel.py [--nx 32] [--p0 10] [--uMag 0.01] [--tLimit 0.5]
"""
import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=32)
ap.add_argument('--p0', type=float, default=10.0)
ap.add_argument('--uMag', type=float, default=0.01)
ap.add_argument('--tLimit', type=float, default=0.5)
ap.add_argument('--scheme', default='artificialCompressible')
ap.add_argument('--no-video', dest='video', action='store_false')
ap.add_argument('--no-cudaGraph', dest='cudaGraph', action='store_false')
ap.add_argument('--tag', default='')
ap.add_argument('--out', default=os.path.join(HERE, 'out_periodicLevel'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as base
from warpSPH.runner import run

rows = []


def initialConditions(ctx, system):
    base.initialConditions(ctx, system)
    system.state.pressures = torch.full_like(system.state.pressures, args.p0)


def diagnostics(ctx, system):
    out = dict(base.diagnostics(ctx, system)) if base.diagnostics else {}
    st = system.state
    f = st.kinds == 0
    p = st.pressures[f]
    out.update(pMean=float(p.mean()), pMin=float(p.min()), pMax=float(p.max()),
               pStd=float(p.std()), vMax=float(st.velocities[f].norm(dim=-1).max()))
    rows.append(dict(t=float(system.t), **{k: v for k, v in out.items() if isinstance(v, (int, float))}))
    return out


case = dataclasses.replace(base, initialConditions=initialConditions, diagnostics=diagnostics)
tag = f'periodic_{args.scheme}_nx{args.nx}_p0{args.p0:g}_u{args.uMag:g}' + (f'_{args.tag}' if args.tag else '')
root = os.path.join(args.out, tag)
os.makedirs(root, exist_ok=True)
kw = dict(scheme=args.scheme, nx=args.nx, tLimit=args.tLimit, quiet=False, progress=True,
          params=dict(uMag=args.uMag, nu=0.01), cudaGraph=args.cudaGraph)
if args.video:
    kw.update(plot=True, video=True, plotInterval=5, exportRoot=root)
r = run(case, **kw)
json.dump(rows, open(os.path.join(root, 'level.json'), 'w'))
print(f'[{tag}] diverged={r.diverged} steps={r.nSteps}', flush=True)
step = max(1, len(rows) // 12)
print(f"{'t':>8} {'pMean':>10} {'pMin':>10} {'pMax':>10} {'pStd':>10} {'vMax':>10}")
for row in rows[::step] + [rows[-1]]:
    print(f"{row['t']:8.4f} {row['pMean']:10.4g} {row['pMin']:10.4g} {row['pMax']:10.4g} {row['pStd']:10.3g} {row['vMax']:10.3g}")
