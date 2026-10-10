#!/usr/bin/env python
"""Flow past a disk / square in a periodic box with a pinned-velocity frame (`cylinderWake`), analytic body, against the unconfined literature and the boundaries repo's `scripts/studies/cylinder_wake.py`
(stage 3 E5/E6: the `pinned` band).

    python scripts/probe_cylinderWake.py --Re 100 --res 10 --time 150 [--shape disk|square] [--scheme deltaSPH] [--out DIR] [--no-video]

Reports C_D (mean over the second half), C_L (rms and amplitude), the Strouhal number of the lift and the recirculation length behind the body from the end state, against the literature of the
UNCONFINED circular cylinder (the frame and the periodic images give a blockage D / (Ly - 2 b), which raises C_D and St by a few percent):
  Re 20: C_D 2.0-2.09, L_w / D 0.91-0.94;  Re 40: C_D 1.50-1.55, L_w / D 2.2-2.35;  Re 100: C_D 1.33-1.35, C_L amplitude 0.32-0.34, St 0.164-0.166 (Dennis & Chang 1970, Fornberg 1980, Williamson 1996).
"""
import argparse
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

REF = {20: dict(CD=(2.0, 2.09), Lw=(0.91, 0.94)), 40: dict(CD=(1.50, 1.55), Lw=(2.2, 2.35)), 100: dict(CD=(1.33, 1.35), CLamp=(0.32, 0.34), St=(0.164, 0.166))}


def recirculationLength(x, v, xc, yc, half, dx):
    """length behind the body over which the centreline velocity is negative: particles within one spacing of the centreline, binned by dx from the rear of the body to the first bin with u > 0."""
    m = (np.abs(x[:, 1] - yc) < dx) & (x[:, 0] > xc + half)
    if not m.any():
        return 0.0
    k = ((x[m, 0] - (xc + half)) / dx).astype(int)
    nb = int(k.max()) + 1
    cnt = np.bincount(k, minlength=nb)
    ub = np.bincount(k, weights=v[m, 0], minlength=nb) / np.maximum(cnt, 1)
    neg = (ub < 0) & (cnt > 0)
    if not neg[:3].any():
        return 0.0
    first = int(np.argmax(~neg & (cnt > 0)))
    if first == 0 or not (~neg[first:]).any():
        return 0.0
    u0, u1 = ub[first - 1], ub[first]
    return ((first - 0.5) + (-u0) / (u1 - u0)) * dx


def strouhal(t, y, t0):
    """D f / U (D = U = 1) with f the dominant frequency of y(t) for t >= t0, resampled uniformly: mean removed, Hann window, FFT peak with parabolic interpolation."""
    m = t >= t0
    tu = np.linspace(t[m][0], t[m][-1], int(m.sum()))
    y = np.interp(tu, t[m], y[m])
    dt = float(tu[1] - tu[0])
    y = (y - y.mean()) * np.hanning(len(y))
    n = 8 * len(y)
    sp = np.abs(np.fft.rfft(y, n))
    k = int(np.argmax(sp[1:])) + 1
    if 1 <= k < len(sp) - 1:
        a, b, c = np.log(sp[k - 1] + 1e-300), np.log(sp[k] + 1e-300), np.log(sp[k + 1] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k / (n * dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', default='deltaSPH', choices=('deltaSPH', 'omniIncompressible', 'divergenceFree'))
    ap.add_argument('--shape', default='disk', choices=('disk', 'square'))
    ap.add_argument('--Re', type=float, default=100.0)
    ap.add_argument('--res', type=int, default=10, help='D / dx')
    ap.add_argument('--Lx', type=float, default=30.0)
    ap.add_argument('--Ly', type=float, default=15.0)
    ap.add_argument('--band', type=float, default=1.5)
    ap.add_argument('--xb', type=float, default=10.0)
    ap.add_argument('--c0', type=float, default=10.0, help='speed of sound in units of U (deltaSPH)')
    ap.add_argument('--time', type=float, default=150.0, help='in D / U')
    ap.add_argument('--closure', default='noslipMoment')
    ap.add_argument('--visc', default='morris', choices=('alpha', 'morris'))
    ap.add_argument('--pressureConsistent', action='store_true')
    ap.add_argument('--no-shift', dest='shift', action='store_false', default=True)
    ap.add_argument('--dt', type=float, default=2e-2, help='incompressible loops: the fixed step')
    ap.add_argument('--cfl', type=float, default=0.25, help='deltaSPH: the fixed step is cfl * h / (c0 + U)')
    ap.add_argument('--closedPreset', action='store_true')
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
    case = getCase('cylinderWake')
    params = {**case.params, 'shape': a.shape, 'Re': a.Re, 'Ly': a.Ly, 'band': a.band, 'xb': a.xb, 'soundSpeed': a.c0, 'wallViscosityClosure': a.closure, 'fluidViscosity': a.visc,
              'pressureConsistent': a.pressureConsistent, 'shifting': a.shift, 'closedPreset': a.closedPreset,
              'targetDt': a.cfl * 4.0 / a.res / (a.c0 + 1.0)}
    common = dict(kernel='Wendland2', **({'integrationScheme': 'semiImplicitEuler', 'supportMode': 'SuperSymmetric', 'dt': a.dt, 'adaptiveDt': False} if inc else {}))
    spec = CaseSpec(caseName='cylinderWake', scheme=a.scheme, params=params).merged(**case.defaults).merged(
        cudaGraph=(a.cudaGraph and a.scheme == 'deltaSPH'), scheme=a.scheme, nx=int(round(a.Lx * a.res)), L=a.Lx, tLimit=a.time, plot=a.video, video=a.video, show=False, store=False, progress=True, quiet=True, plotInterval=400,
        velocityAlarmPlotInterval=1, stallProgress=1e-3, **common, **({'exportRoot': a.out} if (a.out and a.video) else {}))
    r = run(case, spec)
    cd, cl = np.asarray(r.series('dragCoefficient')), np.asarray(r.series('liftCoefficient'))
    t = np.asarray(r.series('t'))[-len(cd):]
    half = t >= 0.5 * t[-1]
    st = r.state.state
    f = st.kinds == 0
    dx = a.Lx / int(round(a.Lx * a.res))
    xc, yc = r.ctx.scratch['bodyCentre']
    x = st.positions[f].cpu().numpy()
    v = st.velocities[f].cpu().numpy()
    Lx, Ly = np.array(r.ctx.config.domain.max.cpu())
    xw = np.stack([np.mod(x[:, 0], Lx), np.mod(x[:, 1], Ly)], 1)
    Lw = recirculationLength(xw, v, xc, yc, 0.5, dx)
    clr = cl[half]
    St = strouhal(t, cl, 0.5 * t[-1]) if clr.std() > 1e-2 else float('nan')
    amp = 0.5 * (np.percentile(clr, 99) - np.percentile(clr, 1))
    print(f'RESULT {a.scheme} {a.shape} Re={a.Re:g} D/dx={a.res} N={int(f.sum())} t={float(r.state.t):.1f} blockage={r.ctx.scratch["blockage"]:.3f} diverged={r.diverged}')
    print(f'RESULT C_D={cd[half].mean():.4f}  C_L rms={clr.std():.4f} amp={amp:.4f}  St={St:.4f}  L_w/D={Lw:.3f}')
    ref = REF.get(int(a.Re)) if a.shape == 'disk' else None
    if ref:
        print('RESULT unconfined literature:', '  '.join(f'{n} {lo:g}-{hi:g}' for n, (lo, hi) in ref.items()))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        np.savez(os.path.join(a.out, f'wake_{a.shape}_Re{a.Re:g}_r{a.res}_{a.scheme}.npz'), t=t, CD=cd, CL=cl, x=x, v=v)


if __name__ == '__main__':
    main()
