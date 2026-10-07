#!/usr/bin/env python3
"""KH mode amplitude A(t) of the nx 256 runs written by `probe_rosswogKHMap.py` (series in each `kh_map.npz`), side by side.

    scripts/plot_kh256_series.py [--out docs/av/kh256_2026-10-07]
"""
import argparse
from pathlib import Path
import numpy as np

RUNS = [('Monaghan + fixed alpha = 1 (smooth IC)', 'results/kh256_NoneSwitch', '#999999', '-'),
        ('Monaghan + Cullen & Dehnen (smooth IC)', 'results/kh256_CullenDehnen2010', '#d62728', '-'),
        ('Monaghan + Rosswog 2020 (smooth IC)', 'results/kh256_Rosswog2020', '#1f77b4', '-'),
        ('CRKSPH default, alpha = 1 (sharp IC)', 'results/kh256crk_sharp', '#2ca02c', '-'),
        ('CRKSPH default, alpha = 1 (smooth IC)', 'results/kh256crk_smooth', '#2ca02c', '--')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='docs/av/kh256_2026-10-07')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    rows = []
    for label, d, color, ls in RUNS:
        f = Path(d) / 'kh_map.npz'
        if not f.exists():
            continue
        s = np.load(f)['series']
        ax[0].plot(s[:, 0], s[:, 1], color=color, ls=ls, label=label)
        ax[1].plot(s[:, 0], s[:, 2], color=color, ls=ls)
        i = int(np.argmin(np.abs(s[:, 0] - 1.5)))
        rows.append((label, s[i, 1], s[:, 1].max(), s[s[:, 1].argmax(), 0], s[-1, 1], s[-1, 2]))
    ax[0].scatter([1.5], [0.1479], marker='*', s=140, color='k', zorder=5, label='McNally et al. (2012), t = 1.5')
    ax[0].set(xlabel='t', ylabel='transverse-velocity mode amplitude A(t)', title='KH mode growth, nx 256')
    ax[1].set(xlabel='t', ylabel='mean alpha', yscale='log', title='mean dissipation parameter')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'kh256_amplitude.png', dpi=130)
    lines = ['| run | A(1.5) | A max | t at max | A(t = 3) | alpha mean (final) |', '|---|---|---|---|---|---|']
    lines += [f'| {l} | {a15:.4f} | {am:.4f} | {tm:.2f} | {af:.4f} | {al:.4f} |' for l, a15, am, tm, af, al in rows]
    (out / 'kh256_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
