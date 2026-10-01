"""Gresho-Chan vortex with smoothed cusps (CRKSPH_LIMITER_PLAN P3).

The standard profile has slope discontinuities at r = 0.2 and 0.4 and the CRK
spin-up energy injection sits next to them. `--widths` runs the vortex with
`cuspWidth = w` (`greshoProfile`; 0 = standard) and reports, against the *same*
profile's exact steady solution: KE(t)/KE(0), `ke_rebound` (largest rise of KE
above its running minimum), L1(v_phi), peak speed, angular-momentum change and
total-energy drift. Video on by default (vispy), one run at a time.

    scripts/probe_greshoCusp.py --nx 64 --widths 0 0.02 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from warpSPH.cases.greshoVortex import greshoVortexCase
from warpSPH.caseUtils.compressible.greshoVortex import greshoProfile
from warpSPH.runner import run


def keRebound(ke):
    ke = np.asarray(ke, dtype=float)
    return float((ke - np.minimum.accumulate(ke)).max() / ke[0])


def metrics(res, width):
    st = res.state.state
    x = st.positions.detach().double()
    v = st.velocities.detach().double()
    r = torch.linalg.norm(x, dim=-1)
    vphi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / r.clamp(min=1e-30)
    vex, _ = greshoProfile(r, width)
    m = r < 0.5
    ke = res.series('kineticEnergy')
    E = res.series('totalEnergy')
    mass = st.masses.detach().double()
    Lz = float((mass * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])).sum())
    return dict(width=width, t=float(res.state.t), steps=int(res.nSteps), diverged=bool(res.diverged),
                keFinalRel=float(ke[-1] / ke[0] - 1), keRebound=keRebound(ke),
                L1vphi=float((vphi - vex).abs()[m].mean()), peakSpeed=float(v.norm(dim=-1).max()),
                peakExact=float(vex.max()), Lz=Lz, energyDrift=float((E[-1] - E[0]) / E[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='+', default=[64])
    ap.add_argument('--widths', type=float, nargs='+', default=[0.0, 0.02, 0.05])
    ap.add_argument('--tLimit', type=float, default=3.0)
    ap.add_argument('--switch', default='NoneSwitch')
    ap.add_argument('--out', default='results/probes/gresho_cusp')
    ap.add_argument('--noVideo', dest='video', action='store_false')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for nx in a.nx:
        for w in a.widths:
            tag = f'nx{nx}_w{w:g}'
            kw = dict(plot=True, video=True, exportRoot=str(out / 'runs' / tag),
                      velocityAlarmPlotInterval=1) if a.video else {}
            res = run(greshoVortexCase, nx=nx, tLimit=a.tLimit, progress=True, stallProgress=1e-3,
                      params=dict(cuspWidth=w, viscositySwitch=a.switch), **kw)
            row = dict(nx=nx, **metrics(res, w))
            rows.append(row)
            print('\n[gresho-cusp]', json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}),
                  flush=True)
            json.dump(rows, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
