#!/usr/bin/env python
"""Stokes flow through a periodic square array of cylinders / squares (`stokesArray`), analytic bodies, against Sangani-Acrivos (disk) / the Fourier Stokes solve (square), the boundaries repo's rungs 2 and 4
(`scripts/studies/periodic_cylinder_array.py`).

    python scripts/probe_stokesArray.py --scheme omniIncompressible --closedPreset [--shape disk|square] [--n 48] [--R 0.2] [--time 20] [--out DIR] [--no-video]

1. the viscosity of the discretisation `nu_shear` from the decay of a shear wave in the fluid-only periodic box of the same lattice (`periodicChannel`, `shearWave=True`);
2. the array run (video on): `K = F / (nu U)`, `F` the load of the fluid on the body (pressure + viscous), `U` the superficial velocity, against `K_ref = (1 - c) K_SA(c)` (disk) / 25.91 (square a = 1/3);
   `F / (rho f N dx^2)`: the momentum balance.
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
    ap.add_argument('--shape', default='disk', choices=('disk', 'square'))
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--R', type=float, default=0.2)
    ap.add_argument('--a', type=float, default=1.0 / 3.0)
    ap.add_argument('--f', type=float, default=0.03)
    ap.add_argument('--time', type=float, default=20.0)
    ap.add_argument('--dt', type=float, default=2e-3)
    ap.add_argument('--closure', default='noslipMoment')
    ap.add_argument('--visc', default='morris', choices=('alpha', 'morris'))
    ap.add_argument('--projection', default='jacobi', choices=('jacobi', 'compact'))
    ap.add_argument('--noDensitySolve', action='store_true')
    ap.add_argument('--particleShift', default='none')
    ap.add_argument('--closedPreset', action='store_true')
    ap.add_argument('--pressureConsistent', action='store_true')
    ap.add_argument('--wallPressureViscous', action='store_true')
    ap.add_argument('--no-shift', dest='shift', action='store_false', default=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    ap.add_argument('--no-cudaGraph', dest='cudaGraph', action='store_false', default=True, help='delta+: run the step eagerly instead of replaying it from a CUDA graph (bitwise the same, ~4x slower)')
    ap.add_argument('--no-compile', dest='compileWalls', action='store_false', default=True, help='do not torch.compile the no-slip wall closure (WARPSPH_COMPILE_WALLS; ~1.6x faster, not bitwise)')
    a = ap.parse_args()
    if a.compileWalls:
        os.environ.setdefault('WARPSPH_COMPILE_WALLS', '1')
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
        spec = CaseSpec(caseName=caseName, scheme=a.scheme, params={**case.params, 'wallViscosityClosure': a.closure, 'fluidViscosity': a.visc, 'projection': a.projection, 'densitySolve': not a.noDensitySolve,
                                                              'particleShift': a.particleShift, 'closedPreset': a.closedPreset, 'shifting': a.shift, 'pressureConsistent': a.pressureConsistent, 'wallPressureViscous': a.wallPressureViscous, **params}).merged(**case.defaults).merged(
            cudaGraph=(a.cudaGraph and a.scheme == 'deltaSPH'), scheme=a.scheme, nx=a.n, tLimit=tLimit, plot=video, video=video, show=False, store=False, progress=True, quiet=True, plotInterval=200, velocityAlarmPlotInterval=1, stallProgress=1e-3, **common,
            **({'exportRoot': a.out} if (a.out and video) else {}))
        return run(case, spec)

    r = go('periodicChannel', {'shearWave': True}, 1.0, False)
    t, A = np.asarray(r.series('t')), np.asarray(r.series('modeAmplitude'))
    W = r.ctx.scratch['W']
    m = (t > 0.1) & (A > 1e-3)
    nu = float(-np.polyfit(t[m], np.log(A[m]), 1)[0] / (2 * math.pi / W) ** 2)
    print(f'SHEAR n={a.n} nu_shear {nu:.5f} (nominal {r.ctx.scratch["nuNominal"]:.5f})', flush=True)

    r = go('stokesArray', {'shape': a.shape, 'R': a.R, 'a': a.a, 'f': a.f, 'nuReference': nu}, a.time, a.video)
    st = r.state.state
    f = st.kinds == 0
    c = float(r.ctx.scratch['solidFraction'])
    if a.shape == 'disk':
        from warpSPH.cases.stokesArray import sanganiAcrivos
        Kref = (1.0 - c) * sanganiAcrivos(c)
    else:
        Kref = 25.91
    K = np.asarray(r.series('dragCoefficient'))
    F = np.asarray(r.series('bodyLoad'))
    U = np.asarray(r.series('superficialVelocity'))
    tail = slice(-max(len(K) // 10, 1), None)
    Fbal = a.f * float(st.masses[f].sum())
    print(f'RESULT {a.scheme} {a.shape} n={a.n} N={int(f.sum())} c={c:.4f} nu_shear={nu:.5f} t={float(r.state.t):.1f} diverged={r.diverged}')
    print(f'RESULT K = {K[tail].mean():.3f}  K_ref = {Kref:.3f}  ratio = {K[tail].mean() / Kref:.4f}   (history of K / K_ref at t/4, t/2, t: {K[len(K) // 4] / Kref:.3f} {K[len(K) // 2] / Kref:.3f} {K[-1] / Kref:.3f})')
    print(f'RESULT load / body force on the fluid = {F[tail].mean() / Fbal:.4f}   U = {U[tail].mean():.5f}')


if __name__ == '__main__':
    main()
