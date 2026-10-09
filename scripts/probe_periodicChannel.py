#!/usr/bin/env python
"""Plane Poiseuille between two periodic analytic plates (`periodicChannel`), delta+ on analytic walls, against the boundaries repo's rung 3 (`scripts/studies/periodic_channel.py`, DeltaSPH2D, alpha form,
n = 48, W = 0.5, alpha 0.5, c0 10, H = 4, f 0.05, 12 s).

    python scripts/probe_periodicChannel.py [--n 48] [--time 12] [--wall noSlip] [--out DIR] [--no-video]

1. the viscosity of the discretisation, `nu_shear`, from the decay of a sinusoidal shear wave in the fully periodic box of the same lattice (`shearWave=True`: `exp(-nu k^2 t)`);
2. the channel run to the end time, video on: the fitted profile amplitude over the parabola `f y (W - y) / (2 nu_shear)` (1 = no-slip at both plates with the bulk viscosity), ten-bin profile, the plate
   loads against the body force on the fluid.
"""
import argparse
import math
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--W', type=float, default=0.5)
    ap.add_argument('--time', type=float, default=12.0)
    ap.add_argument('--wall', default='noSlip')
    ap.add_argument('--visc', default='alpha', choices=('alpha', 'morris'))
    ap.add_argument('--closure', default='mirror', choices=('mirror', 'noslipMoment'))
    ap.add_argument('--no-complement', dest='complement', action='store_false', default=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--f', type=float, default=0.05)
    ap.add_argument('--no-shift', dest='shift', action='store_false', default=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    a = ap.parse_args()
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    case = getCase('periodicChannel')

    def go(params, tLimit, video):
        spec = CaseSpec(caseName='periodicChannel', scheme='deltaSPH', params={**case.params, 'alpha': a.alpha, 'W': a.W, 'f': a.f, 'wallBC': a.wall, 'shifting': a.shift, 'fluidViscosity': a.visc, 'wallViscosityClosure': a.closure, 'complementMoments': a.complement, **params}).merged(**case.defaults).merged(
            nx=a.n, tLimit=tLimit, plot=video, video=video, show=False, store=False, progress=True, quiet=True, plotInterval=100, velocityAlarmPlotInterval=1, stallProgress=1e-3,
            **({'exportRoot': a.out} if (a.out and video) else {}))
        return run(case, spec)

    r = go({'shearWave': True}, 1.0, False)
    t = np.asarray(r.series('t'))
    A = np.asarray(r.series('modeAmplitude'))
    Wsh = r.ctx.scratch['W']
    k2 = (2.0 * math.pi / Wsh) ** 2
    m = (t > 0.1) & (A > 1e-3)
    nu = float(-np.polyfit(t[m], np.log(A[m]), 1)[0] / k2)
    nuNominal = r.ctx.scratch['nuNominal']
    print(f'SHEAR n={a.n} nu_shear {nu:.5f} (nominal alpha c0 h / (8 xi) {nuNominal:.5f}, ratio {nu / nuNominal:.4f})', flush=True)

    r = go({'nuReference': nu}, a.time, a.video)
    W = r.ctx.scratch['W']
    st = r.state.state
    f = st.kinds == 0
    y = st.positions[f, 1].double().cpu().numpy()
    u = st.velocities[f, 0].double().cpu().numpy()
    exact = a.f * y * (W - y) / (2 * nu)
    amp = float((u * exact).sum() / (exact * exact).sum())
    umax = a.f * W * W / (8 * nu)
    bins = np.linspace(0, W, 11)
    prof = [u[(y >= b0) & (y < b1)].mean() / umax for b0, b1 in zip(bins[:-1], bins[1:])]
    N = int(f.sum())
    Fbal = a.f * float(st.masses[f].sum())
    pl = np.asarray(r.series('plateLoad'))
    plate = float(pl[-max(len(pl) // 10, 1):].mean())
    kc = np.asarray(r.series('profileAmplitude'))
    print(f'RESULT n={a.n} W={W:.4f} N={N} visc={a.visc} closure={a.closure} nu_shear={nu:.5f} t={float(r.state.t):.2f} diverged={r.diverged}')
    if a.visc == 'morris':
        print(f'RESULT Morris: profile amplitude / parabola(nu long-wave = nominal {nuNominal:.5f}) = {amp * nu / nuNominal:.4f}')
    print(f'RESULT profile amplitude / parabola(nu_shear) = {amp:.4f}   (history at t/4, t/2, t: {kc[len(kc)//4]:.3f} {kc[len(kc)//2]:.3f} {kc[-1]:.3f})')
    print('RESULT u/umax over 10 bins:', np.round(prof, 3).tolist(), ' parabola:', np.round([4 * ((i + .5) / 10) * (1 - (i + .5) / 10) for i in range(10)], 3).tolist())
    print(f'RESULT plate load (pressure + viscous, mean over the last tenth of the run) {plate:.5f}  body force on the fluid {Fbal:.5f}  ratio {plate / Fbal:.4f}')


if __name__ == '__main__':
    main()
