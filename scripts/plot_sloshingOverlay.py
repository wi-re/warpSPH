#!/usr/bin/env python
"""Overlay the Sensor-1 pressure (and kinetic energy) of several sloshingTank runs on the measured SPHERIC record.

    python scripts/plot_sloshingOverlay.py --out DIR \
        "δ⁺ particle walls=DIR/wcsph_series.npz" "δ⁺ analytic walls=DIR/wcsph_analytic_series.npz" "DFSPH=DIR/dfsph_series.npz"

Series files are the `<tag>_series.npz` written by `examples/sloshingTank/run_sloshingTank.py`. Top row: the full 7 s, each run smoothed
(Gaussian, `--smoothSigma`, default 10 ms) against the measurement; middle row: the three impacts; bottom row: kinetic energy.
Writes `sloshing_overlay.png` into `--out` and prints the smoothed impact peaks.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLL = os.path.join(REPO, 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']          # categorical slots 1-3 (+ 7), dataviz reference palette
WINDOWS = ((2.2, 2.9), (3.9, 4.6), (5.4, 6.1))                 # the three impacts


def smooth(t, p, sigma, dt=1e-3):
    grid = np.arange(0.0, float(np.nanmax(t)), dt)
    ok = np.isfinite(p)
    y = np.interp(grid, t[ok], p[ok])
    n = int(4 * sigma / dt)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * dt / sigma) ** 2)
    k /= k.sum()
    return grid, np.convolve(np.pad(y, n, mode='edge'), k, mode='valid')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='"label=path/to/series.npz"')
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoothSigma', type=float, default=0.01)
    ap.add_argument('--field', default='sensorPressure', help='series key of the pressure (every step)')
    a = ap.parse_args()

    raw = np.genfromtxt(ROLL, delimiter='\t', skip_header=1)
    expT, expP = raw[:, 0], raw[:, 1] * 100.0
    runs = []
    for spec in a.runs:
        label, path = spec.split('=', 1)
        runs.append((label, np.load(path)))

    fig = plt.figure(figsize=(12, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.0, 0.8])
    axFull = fig.add_subplot(gs[0, :])
    axW = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axKE = fig.add_subplot(gs[2, :])

    # the measured record smoothed the same way as the runs (a sharp raw peak and a 10 ms-smoothed one are not comparable: compare like with like)
    gm, sm_ = smooth(expT, expP, a.smoothSigma)
    for ax in [axFull] + axW:
        ax.plot(expT, expP, color='0.7', lw=0.8, ls='--', label='measured (Sensor 1, raw)')
        ax.plot(gm, sm_, color='0.25', lw=1.5, ls='--', label=f'measured, same {a.smoothSigma * 1e3:.0f} ms smoothing')
    peaks = []
    peaks.append(('measured, smoothed', [(float(sm_[(gm >= lo) & (gm < hi)].max()), float(gm[(gm >= lo) & (gm < hi)][sm_[(gm >= lo) & (gm < hi)].argmax()])) for lo, hi in WINDOWS]))
    for (label, d), c in zip(runs, COLORS):
        t, p = d['t'], d[a.field]
        g, s = smooth(t, p, a.smoothSigma)
        axFull.plot(g, s, color=c, lw=1.5, label=label)
        for ax in axW:
            ax.plot(g, s, color=c, lw=1.5)
        axKE.plot(t, d['kineticEnergy'], color=c, lw=1.2, label=label)
        row = []
        for lo, hi in WINDOWS:
            m = (g >= lo) & (g < hi)
            row.append((float(s[m].max()), float(g[m][s[m].argmax()])))
        peaks.append((label, row))

    axFull.set_ylim(-1000, 7500)
    axFull.set_xlim(0, 7)
    axFull.set_ylabel('Sensor 1 pressure [Pa]')
    axFull.set_title(f'SPHERIC TC10 sloshing, nx = 200: Sensor 1 ({a.smoothSigma * 1e3:.0f} ms Gaussian smoothing) against the measurement')
    axFull.legend(loc='upper left', fontsize=9, frameon=False, ncol=3)
    for ax, (lo, hi) in zip(axW, WINDOWS):
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1000, 7500)
        ax.set_xlabel('time [s]')
    axW[0].set_ylabel('pressure [Pa]')
    for ax in axW[1:]:
        ax.tick_params(labelleft=False)
    axKE.set_xlim(0, 7)
    axKE.set_ylabel('kinetic energy [J/m]')
    axKE.set_xlabel('time [s]')
    for ax in [axFull, axKE] + axW:
        ax.grid(alpha=0.25)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    png = os.path.join(a.out, 'sloshing_overlay.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('wrote', png)
    for label, row in peaks:
        print(f'{label:28s}' + '  '.join(f'{p:6.0f} Pa @ {tp:.3f} s' for p, tp in row))


if __name__ == '__main__':
    main()
