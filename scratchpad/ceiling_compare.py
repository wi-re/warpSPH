"""Compare forensics resumes: the ceiling cluster's |P| / |v| per t* window, run extremes."""
import os
import sys
import numpy as np

# --uids a,b,c picks the watched cluster (default: the seed-3 §3.1 cluster;
# seed 2's lone rider of §7 is --uids 80)
argv = sys.argv[1:]
C = [889, 402, 808, 321, 726]
if '--uids' in argv:
    k = argv.index('--uids')
    C = [int(u) for u in argv[k + 1].split(',')]
    argv = argv[:k] + argv[k + 2:]
tags = argv or ['ev11000', 'ev11000_cfl015', 'ev11000_rk4', 'ev11000_tcc']
base = 'scripts/out_ceiling/forensics'
for tag in tags:
    F = np.load(f'{base}/{tag}_rows.npz')
    ts = F['t'] * np.sqrt(9.81 / 0.6)
    m = np.isin(F['uid'], C)
    rd = f'{base}/{tag}'
    r = np.load(os.path.join(rd, [f for f in os.listdir(rd) if f.endswith('.npz')][0]))
    print(f"{tag}: run maxV {np.nanmax(r['maxVelocity']):.2f} rho [{np.nanmin(r['minDensity']):.3f},{np.nanmax(r['maxDensity']):.3f}] t* end {np.nanmax(r['tStar']):.2f}")
    for lo, hi in [(4.32, 4.4), (4.4, 4.5), (4.5, 4.55), (4.55, 4.6), (4.6, 4.65), (4.65, 4.8), (4.8, 5.0), (5.0, 5.2), (5.2, 5.4)]:
        w = m & (ts >= lo) & (ts < hi)
        if w.any():
            print(f"    [{lo},{hi}) cluster |P|max {np.abs(F['P'][w]).max():7.1f} |v|max {np.linalg.norm(F['v'][w], axis=1).max():5.2f}"
                  f"  s/dx mean {np.mean((F['yCeil'] - F['y'][w]) / F['dx']):.2f}")
