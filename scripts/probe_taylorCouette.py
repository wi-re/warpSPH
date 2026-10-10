#!/usr/bin/env python
"""Taylor-Couette flow (`taylorCouette`): analytic rotating inner cylinder + fixed outer cavity, against the exact solution (the boundaries repo's `scripts/studies/taylor_couette.py`: DFSPH closed preset
n = 48 profile 1.0002 / inner torque 0.974; delta+ Morris + noslipMoment + complement n = 32 profile 1.0001 / torques 0.938 inner, 0.77 outer on the cut lattice).

    python scripts/probe_taylorCouette.py --scheme omniIncompressible --closedPreset [--n 48] [--time 25] [--out DIR] [--no-video]
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
    ap.add_argument('--scheme', default='omniIncompressible', choices=('deltaSPH', 'omniIncompressible', 'divergenceFree'))
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--nh', type=float, default=4.0, help='support radius in dx (the oracle DFSPH2D runs Taylor-Couette at 1 / PACKING = 2.505)')
    ap.add_argument('--time', type=float, default=25.0)
    ap.add_argument('--dt', type=float, default=2e-3)
    ap.add_argument('--projection', default='jacobi', choices=('jacobi', 'compact'))
    ap.add_argument('--noDensitySolve', action='store_true')
    ap.add_argument('--particleShift', default='none')
    ap.add_argument('--closedPreset', action='store_true')
    ap.add_argument('--pressureConsistent', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    a = ap.parse_args()
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    inc = a.scheme != 'deltaSPH'
    common = dict(kernel='Wendland2', **({'integrationScheme': 'semiImplicitEuler', 'supportMode': 'SuperSymmetric', 'dt': a.dt, 'adaptiveDt': False} if inc else {}))

    def go(caseName, params, tLimit, video):
        case = getCase(caseName)
        spec = CaseSpec(caseName=caseName, scheme=a.scheme, params={**case.params, 'wallViscosityClosure': 'noslipMoment', 'fluidViscosity': 'morris', 'projection': a.projection, 'densitySolve': not a.noDensitySolve,
                                                              'particleShift': a.particleShift, 'closedPreset': a.closedPreset, 'pressureConsistent': a.pressureConsistent, **params}).merged(**case.defaults).merged(
            scheme=a.scheme, nx=a.n, n_h=a.nh, tLimit=tLimit, plot=video, video=video, show=False, store=False, progress=True, quiet=True, plotInterval=200, velocityAlarmPlotInterval=1, stallProgress=1e-3, **common,
            **({'exportRoot': a.out} if (a.out and video) else {}))
        return run(case, spec)

    r = go('periodicChannel', {'shearWave': True}, 1.0, False)
    t, A = np.asarray(r.series('t')), np.asarray(r.series('modeAmplitude'))
    W = r.ctx.scratch['W']
    m = (t > 0.1) & (A > 1e-3)
    nu = float(-np.polyfit(t[m], np.log(A[m]), 1)[0] / (2 * math.pi / W) ** 2)
    print(f'SHEAR n={a.n} nu_shear {nu:.5f}', flush=True)

    r = go('taylorCouette', {'nuReference': nu}, a.time, a.video)
    st = r.state.state
    f = st.kinds == 0
    amp = np.asarray(r.series('profileAmplitude'))
    Ti, To = np.asarray(r.series('torqueInner')), np.asarray(r.series('torqueOuter'))
    Tex = np.asarray(r.series('torqueExact'))
    tail = slice(-max(len(amp) // 10, 1), None)
    x = st.positions[f].double().cpu().numpy()
    v = st.velocities[f].double().cpu().numpy()
    rr = np.linalg.norm(x, axis=1)
    ut = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rr
    r1, r2, om = 0.2, 0.5, 1.0
    ex = om * r1 ** 2 * (r2 ** 2 / rr - rr) / (r2 ** 2 - r1 ** 2)
    bins = np.linspace(r1, r2, 9)
    prof = [ut[(rr >= b0) & (rr < b1)].mean() / ex[(rr >= b0) & (rr < b1)].mean() for b0, b1 in zip(bins[:-1], bins[1:])]
    print(f'RESULT {a.scheme} n={a.n} N={int(f.sum())} nu_shear={nu:.5f} t={float(r.state.t):.1f} diverged={r.diverged}')
    print(f'RESULT profile amplitude / exact = {amp[tail].mean():.4f}   by radial bin (inner -> outer): {np.round(prof, 3).tolist()}')
    print(f'RESULT torque inner / exact = {Ti[tail].mean() / Tex[tail].mean():.4f}   outer / (-exact) = {To[tail].mean() / -Tex[tail].mean():.4f}')


if __name__ == '__main__':
    main()
