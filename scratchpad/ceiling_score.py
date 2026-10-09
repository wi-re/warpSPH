"""Score Marrone 3.1 npz runs with the probe's own acceptance checks and add
the kick metrics: max|v| and when, rho extremes, and max|v| restricted to
t* in the ceiling-kick window. Usage: ceiling_score.py <npz> ..."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import probe_deltaSPHMarrone as P

print(f'{"run":58s} {"checks":>6} {"P1pl":>5} {"P2pk":>9} {"max|v|@t*":>12} {"rho min,max":>13} {"v>8 steps":>9}  failed')
for f in sys.argv[1:]:
    d = np.load(f, allow_pickle=True)
    col = {k: d[k] for k in d.files if k != 'meta'}
    checks, m = P._score(col)
    n = sum(ok for _, ok, _ in checks)
    v = col['maxVelocity']; ts = col['tStar']
    i = int(np.nanargmax(v))
    fails = ', '.join(nm for nm, ok, _ in checks if not ok) or '-'
    print(f'{os.path.basename(f)[:58]:58s} {n}/{len(checks)} {m.get("p1_plateau", np.nan):5.2f} '
          f'{m.get("p2_peak", np.nan):4.2f}@{m.get("p2_tPeak", np.nan):4.2f} {v[i]:5.1f}@{ts[i]:5.2f} '
          f'[{m["rhoMin"]:.2f},{m["rhoMax"]:.2f}] {int(np.sum(v > 8.0)):9d}  {fails}')
