"""Per-radius energy budget of the CRKSPH Gresho vortex (CRKSPH_LIMITER_PLAN P3).

Wraps `warpSPH.schemes.builder.crkSPH_step` (the stage state carries the pair accelerations `ap_ij` = pressure, `av_ij` = viscous)
and accumulates, per radial bin, the kinetic energy each term injects: dKE_i = m_i v_i . sum_j a_ij * w dt (w = 1/2 per RK2 stage).
Reports cumulative pressure and viscous work per bin / KE(0) at the end (t = tLimit), the total vs the measured KE change, and
how the injection scales with nx. Video on by default, one run at a time.

    scripts/probe_greshoPower.py --nx 48 64 96 --tLimit 3
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

import warpSPH.schemes.builder as B
from warpSPH.cases.greshoVortex import greshoVortexCase
from warpSPH.math.scatter import segment_sum
from warpSPH.runner import run

EDGES = np.linspace(0.0, 0.6, 13)


class Budget:
    def __init__(self):
        self.press = np.zeros(len(EDGES))   # last bin = r >= 0.6
        self.visc = np.zeros(len(EDGES))
        self.t = []
        self.cumP, self.cumV = [], []
        self.calls = 0

    def accumulate(self, state, adjacency, dt):
        x = state.positions
        v = state.velocities
        m = state.masses
        nn = adjacency.numNeighbors
        r = torch.linalg.norm(x, dim=-1)
        b = torch.bucketize(r, torch.as_tensor(EDGES[1:-1], dtype=r.dtype, device=r.device)).cpu().numpy()
        w = 0.5 * float(dt)
        for name, arr in (('press', state.ap_ij), ('visc', state.av_ij)):
            a = segment_sum(arr, nn)
            p = (m * (v * a).sum(-1)).double().cpu().numpy() * w
            np.add.at(getattr(self, name), np.minimum(b, len(EDGES) - 1), p)
        self.calls += 1
        if self.calls % 2 == 0:
            self.cumP.append(self.press.sum()); self.cumV.append(self.visc.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='+', default=[48, 64, 96])
    ap.add_argument('--tLimit', type=float, default=3.0)
    ap.add_argument('--out', default='results/probes/gresho_power')
    ap.add_argument('--noVideo', dest='video', action='store_false')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    orig = B.crkSPH_step
    rows = []
    for nx in a.nx:
        bud = Budget()

        def wrapped(system, dt, *args, **kw):
            res = orig(system, dt, *args, **kw)
            update, adjacency, state = res
            bud.accumulate(state, adjacency, dt)
            return res
        B.crkSPH_step = wrapped
        kw = dict(plot=True, video=True, exportRoot=str(out / 'runs' / f'nx{nx}'), velocityAlarmPlotInterval=1) if a.video else {}
        try:
            res = run(greshoVortexCase, nx=nx, tLimit=a.tLimit, progress=True, stallProgress=1e-3, **kw)
        finally:
            B.crkSPH_step = orig
        ke = res.series('kineticEnergy')
        ke0 = float(ke[0])
        row = dict(nx=nx, steps=int(res.nSteps), calls=bud.calls, ke0=ke0, keFinalRel=float(ke[-1] / ke0 - 1),
                   pressureTotal=float(bud.press.sum() / ke0), viscousTotal=float(bud.visc.sum() / ke0),
                   closure=float((bud.press.sum() + bud.visc.sum()) / ke0 - (ke[-1] / ke0 - 1)),
                   edges=EDGES.tolist(), pressPerBin=(bud.press / ke0).tolist(), viscPerBin=(bud.visc / ke0).tolist(),
                   cumPress=(np.array(bud.cumP) / ke0).tolist(), cumVisc=(np.array(bud.cumV) / ke0).tolist())
        rows.append(row)
        print('\n[gresho-power]', json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                                              if k in ('nx', 'steps', 'keFinalRel', 'pressureTotal', 'viscousTotal', 'closure')}), flush=True)
        print('  bin r<   :', ' '.join(f'{e:7.2f}' for e in EDGES[1:]), '   >=0.6', flush=True)
        print('  pressure :', ' '.join(f'{v:7.3f}' for v in row['pressPerBin']), flush=True)
        print('  viscous  :', ' '.join(f'{v:7.3f}' for v in row['viscPerBin']), flush=True)
        json.dump(rows, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
