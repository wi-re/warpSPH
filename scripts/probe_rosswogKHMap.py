#!/usr/bin/env python3
"""AV_PLAN Phase 2 open item (a): where does the Rosswog (2020) entropy trigger fire in
Kelvin-Helmholtz? The AV report shows A(1.5) = 0.012 vs C&D 0.098. This runs KH (2D, Monaghan, C_q = 0,
the AV report's `rosswog2020` column) and snapshots the trigger epsdot (Eq. 16), alpha, density and the
entropy s = P / rho^gamma at several times, then maps them over the box.

    scripts/probe_rosswogKHMap.py [--nx 128] [--tLimit 1.5] [--times 0.05 0.2 0.5 1.5] [--out results/av_M2_khmap]

Writes `<out>/kh_map.png` and `<out>/kh_map.npz`, and prints, per snapshot, the fraction of particles
with epsdot above eps_0 inside the shear layers (|y - 0.25| or |y - 0.75| < band) vs elsewhere.
One run, video on, progress streamed.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warpSPHBootstrap  # noqa: F401,E402
from warpSPH.runner import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=128)
    ap.add_argument('--tLimit', type=float, default=1.5)
    ap.add_argument('--times', type=float, nargs='+', default=[0.05, 0.2, 0.5, 1.5])
    ap.add_argument('--noShear', action='store_true',
                    help='control: v1 = v2 = 0 and no seed perturbation (w0 = 0): density contact only')
    ap.add_argument('--smooth', action='store_true',
                    help='smooth density ramp across the interface (Frontiere 2017 Eq. 100), case param smoothDensity=1')
    ap.add_argument('--switch', default='Rosswog2020', choices=('Rosswog2020', 'CullenDehnen2010', 'NoneSwitch'),
                    help='viscosity switch (Rosswog at the AV report alpha range 0..1; the others at their defaults)')
    ap.add_argument('--stride', type=int, default=25, help='record the amplitude every N steps')
    ap.add_argument('--scheme', default='Monaghan', choices=('Monaghan', 'CompSPH', 'CRKSPH'),
                    help='compressible scheme (default Monaghan, the AV laboratory host)')
    ap.add_argument('--band', type=float, default=0.05)
    ap.add_argument('--out', default='results/av_M2_khmap')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from warpSPH.cases.kelvinHelmholtz import kelvinHelmholtzCase as base
    todo = sorted(a.times)
    snaps = []

    series = []        # (t, McNally mode amplitude, mean alpha), every `stride` steps

    def amplitude(ctx, st):
        kk = float(ctx.param('freq')) * np.pi
        xx, yy_, vy_ = st.positions[:, 0], st.positions[:, 1], st.velocities[:, 1]
        yy_ = torch.where(yy_ < 0.5, yy_, 1.0 - yy_)
        dd = torch.exp(-kk * (yy_ - 0.25).abs())     # McNally et al. (2012) mode amplitude, as av_report
        return (2.0 * torch.sqrt((vy_ * torch.sin(kk * xx) * dd).sum() ** 2
                                 + (vy_ * torch.cos(kk * xx) * dd).sum() ** 2) / dd.sum()).item()

    counter = [0]

    def diagnostics(ctx, state):
        row = base.diagnostics(ctx, state)
        t = float(state.t)
        if counter[0] % a.stride == 0:
            series.append((t, amplitude(ctx, state.state), float(state.state.alphas.mean())))
        counter[0] += 1
        if todo and t >= todo[0]:
            todo.pop(0)
            st = state.state
            kk = float(ctx.param('freq')) * np.pi
            xx, yy_, vy_ = st.positions[:, 0], st.positions[:, 1], st.velocities[:, 1]
            yy_ = torch.where(yy_ < 0.5, yy_, 1.0 - yy_)
            dd = torch.exp(-kk * (yy_ - 0.25).abs())     # McNally et al. (2012) mode amplitude, as av_report
            amp = (2.0 * torch.sqrt((vy_ * torch.sin(kk * xx) * dd).sum() ** 2
                                    + (vy_ * torch.cos(kk * xx) * dd).sum() ** 2) / dd.sum()).item()
            snaps.append(dict(
                t=t, amplitude=amp,
                pos=st.positions.detach().cpu().numpy(),
                alpha=st.alphas.detach().cpu().numpy(),
                epsdot=(st.entropyRates.detach().cpu().numpy() if st.entropyRates is not None
                        else np.zeros(len(st.alphas))),
                rho=st.densities.detach().cpu().numpy(),
                s=st.entropies.detach().cpu().numpy(),
                eps0=float(ctx.schemeConfig.viscositySwitchParams.entropy_eps0),
                eps1=float(ctx.schemeConfig.viscositySwitchParams.entropy_eps1)))
            print(f'[khmap] snapshot at t = {t:.4f}', flush=True)
        return row

    configure = base.configureScheme

    def configureScheme(ctx):
        configure(ctx)
        sp = ctx.schemeConfig.viscositySwitchParams
        if a.switch == 'Rosswog2020':
            sp.alpha_min, sp.alpha_max = 0.0, 1.0       # the AV report's rosswog2020 column

    case = dataclasses.replace(base, diagnostics=diagnostics, configureScheme=configureScheme)
    res = run(case, scheme=a.scheme, nx=a.nx, tLimit=a.tLimit, plot=True, video=True,
              exportRoot=str(out / 'runs'), progress=True, stallProgress=1e-3,
              velocityAlarmPlotInterval=1, params=dict(viscositySwitch=a.switch, **(dict(v1=0.0, v2=0.0, w0=0.0) if a.noShear else {}),
                          **(dict(smoothDensity=1) if a.smooth else {})))
    if todo:     # last requested time past the final step
        print(f'[khmap] note: times never reached: {todo}', flush=True)

    for sn in snaps:
        y = sn['pos'][:, 1]
        layer = (np.abs(y - 0.25) < a.band) | (np.abs(y - 0.75) < a.band)
        hot = sn['epsdot'] > sn['eps0']
        print(f'[khmap] t {sn["t"]:.3f}: layer frac {layer.mean():.3f} | epsdot>eps0: in layer {hot[layer].mean():.3f}, '
              f'outside {hot[~layer].mean():.3f} | alpha mean in layer {sn["alpha"][layer].mean():.3f}, '
              f'outside {sn["alpha"][~layer].mean():.3f} | epsdot median in layer {np.median(sn["epsdot"][layer]):.2e}, '
              f'outside {np.median(sn["epsdot"][~layer]):.2e} | A {sn["amplitude"]:.4f}', flush=True)
    np.savez(out / 'kh_map.npz', series=np.array(series), **{f'{i}_{k}': v for i, sn in enumerate(snaps) for k, v in sn.items()})
    ser = np.array(series)
    iA = int(np.argmin(np.abs(ser[:, 0] - 1.5)))
    print(f'[khmap] SERIES {a.scheme}/{a.switch}{"/smooth" if a.smooth else "/sharp"}: A(t~1.5 = {ser[iA, 0]:.3f}) {ser[iA, 1]:.4f}  A max {ser[:, 1].max():.4f} at t {ser[ser[:, 1].argmax(), 0]:.3f}  '
          f'A(final t {ser[-1, 0]:.3f}) {ser[-1, 1]:.4f}  alpha mean (final) {ser[-1, 2]:.4f}', flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    n = len(snaps)
    hasEps = any(np.any(sn['epsdot'] > 0) for sn in snaps)
    nRows = 3 if hasEps else 2
    fig, ax = plt.subplots(nRows, n, figsize=(4.2 * n, 3.7 * nRows), squeeze=False)
    for i, sn in enumerate(snaps):
        p = sn['pos'].copy()
        p[:, 0] = np.mod(p[:, 0], 1.0)      # periodic box: particles drift by v t
        panels = [(np.maximum(sn['epsdot'], 1e-8), 'epsdot (Eq. 16)', dict(norm=LogNorm(1e-6, 1.0), cmap='magma')),
                  (sn['alpha'], 'alpha', dict(vmin=0, vmax=1, cmap='viridis')),
                  (sn['rho'], 'density', dict(vmin=0.9, vmax=2.1, cmap='RdBu_r'))]
        if not hasEps:
            panels = panels[1:]
        for j, (c, name, kw) in enumerate(panels):
            im = ax[j, i].scatter(p[:, 0], p[:, 1], c=c, s=2, **kw)
            ax[j, i].set(aspect='equal', title=f'{name}, t = {sn["t"]:.2f}', xlim=(0, 1), ylim=(0, 1))
            fig.colorbar(im, ax=ax[j, i], shrink=0.7)
    fig.suptitle(f'Rosswog (2020) trigger on Kelvin-Helmholtz{" (NO SHEAR control)" if a.noShear else ""}{" (smooth density IC)" if a.smooth else ""}, {a.scheme}, {a.switch}, nx {a.nx}')
    fig.tight_layout()
    fig.savefig(out / 'kh_map.png', dpi=110)
    print(f'[khmap] wrote {out / "kh_map.png"}', flush=True)


if __name__ == '__main__':
    main()
