#!/usr/bin/env python
"""The gate for retiring the boundaries repo's `DeltaSPH2D`: warpSPH's own delta+ on analytic walls against `DeltaSPH2D`'s stored 7 s SPHERIC-10 series (`results/deltasph/slosh_B_nx200_series.npz` of the
boundaries repo), with the same processing, plus the boundary-particle warpSPH run and the measurement.

    python scripts/compare_deltaSloshing.py [--out DIR] [--sigma 0.01]

Per run: the Sensor-1 pressure smoothed with the NaN-aware Gaussian of `plot_sloshingOverlay.smooth` (the fluid probe is undefined where fewer than three particles are near it, the stored record
is 43 % NaN), peaks and times in the three impact windows, the timing lag of each impact against the measurement, and the kinetic energy of the run against `DeltaSPH2D`'s on a common 1 ms grid
(correlation, rms difference over the mean, lag of the cross-correlation maximum). Prints a markdown table.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sloshingOverlay import ROLL, WINDOWS, smooth  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = os.path.expanduser('~/dev/curvatureBoundaries/results/deltasph/slosh_B_nx200_series.npz')
A = os.path.join(REPO, 'examples', 'sloshingTank', 'output', 'analytic_ab_2026-10-09')
RUNS = [
    ('DeltaSPH2D, analytic walls (C4, Sun shift)', DELTA, 'sensorPressureProbe'),
    ('warpSPH delta+, analytic walls (C2, Michel)', os.path.join(A, 'wcsph_analytic_series.npz'), 'sensorPressure'),
    ('warpSPH delta+, boundary particles (C4, Michel)', os.path.join(A, 'wcsph_series.npz'), 'sensorPressure'),
]


def peaks(t, p, sigma):
    g, s = smooth(t, p, sigma)
    out = []
    for lo, hi in WINDOWS:
        m = (g >= lo) & (g < hi)
        out.append((float(np.nanmax(s[m])), float(g[m][np.nanargmax(s[m])])))
    return out, g, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, default=0.01)
    a = ap.parse_args()
    raw = np.genfromtxt(ROLL, delimiter='\t', skip_header=1)
    meas, gm, sm = peaks(raw[:, 0], raw[:, 1] * 100.0, a.sigma)
    print('| run | impact 1 [Pa] @ s | impact 2 | impact 3 | lag vs measurement [ms] | rms err of the smoothed record in the windows [Pa] |')
    print('|---|---|---|---|---|---|')
    print('| measured, same smoothing | ' + ' | '.join(f'{p:.0f} @ {t:.3f}' for p, t in meas) + ' | | |')
    data = {}
    for label, path, key in RUNS:
        d = np.load(path)
        pk, g, s = peaks(d['t'], d[key], a.sigma)
        data[label] = d
        lag = [1e3 * (t - tm) for (_, t), (_, tm) in zip(pk, meas)]
        err = []
        for lo, hi in WINDOWS:
            m = (g >= lo) & (g < hi) & np.isfinite(s)
            err.append(np.sqrt(np.mean((s[m] - np.interp(g[m], gm, np.nan_to_num(sm))) ** 2)))
        print(f'| {label} | ' + ' | '.join(f'{p:.0f} @ {t:.3f}' for p, t in pk) + ' | ' + ' / '.join(f'{x:+.0f}' for x in lag) + ' | ' + ' / '.join(f'{e:.0f}' for e in err) + ' |')
    print()
    ref = data[RUNS[0][0]]
    g = np.arange(0.0, 7.0, 1e-3)
    keRef = np.interp(g, ref['t'], ref['kineticEnergy'])
    print('kinetic energy against DeltaSPH2D on a 1 ms grid (mean KE of DeltaSPH2D {:.3e}):'.format(keRef.mean()))
    print('| run | correlation | rms difference / mean | KE peak ratio | cross-correlation lag [ms] | density range | steps |')
    print('|---|---|---|---|---|---|---|')
    for label, path, key in RUNS[1:]:
        d = data[label]
        ke = np.interp(g, d['t'], d['kineticEnergy'])
        corr = float(np.corrcoef(ke, keRef)[0, 1])
        rms = float(np.sqrt(np.mean((ke - keRef) ** 2)) / keRef.mean())
        x = np.correlate(ke - ke.mean(), keRef - keRef.mean(), mode='full')
        lag = (np.argmax(x) - (len(g) - 1)) * 1.0
        print(f'| {label} | {corr:.3f} | {rms:.3f} | {ke.max() / keRef.max():.2f} | {lag:+.0f} | [{float(np.nanmin(d["minDensity"])):.2f}, {float(np.nanmax(d["maxDensity"])):.2f}] | {int(d["nSteps"])} |')
    print(f'DeltaSPH2D density range [{float(np.nanmin(ref["minDensity"])):.2f}, {float(np.nanmax(ref["maxDensity"])):.2f}], {int(ref["steps"])} steps, {float(ref["wall"]):.0f} s')


if __name__ == '__main__':
    main()
