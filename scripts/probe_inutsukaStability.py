#!/usr/bin/env python3
"""GODUNOV_SPH_PLAN layer 3: which variant of Inutsuka's scheme survives the strong-shock cases (cold 1D Noh, Sedov)?

Runs the variants (the full run unless `--steps`) with `av_report`'s machinery (no video) and prints the case's headline metric -- Noh: the post-shock density
error (a few 1e-3 when right, -1 when the run has collapsed), Sedov: the peak density ratio (exact 4, the AV rows 2.1-2.7) -- or where it diverged.

    python scripts/probe_inutsukaStability.py [--steps 600] [--cases noh sedov]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
import av_report as A   # noqa: E402

VARIANTS = {
    'base': {},
    'clamp s*': dict(interfaceClamp=True),
    'linear V': dict(cubicVolume=False),
    'clamp + linear': dict(interfaceClamp=True, cubicVolume=False),
    'shock switch 3': dict(shockSwitchC=3.0),
    'clamp + shock switch': dict(interfaceClamp=True, shockSwitchC=3.0),
    'first order': dict(velocityPairPolicy=0),
    'first order + clamp': dict(velocityPairPolicy=0, interfaceClamp=True),
    'monotonised': dict(stateLimiter=2),
    'eta 0.75': dict(gaussianEta=0.75),
    'eta 1.0': dict(gaussianEta=1.0),
    'eta 1.33': dict(gaussianEta=1.33),
    'eta 1.0 + shock switch': dict(gaussianEta=1.0, shockSwitchC=3.0),
    'eta 1.0 linear V': dict(gaussianEta=1.0, cubicVolume=False),
    'eta 0.75 linear V': dict(gaussianEta=0.75, cubicVolume=False),
    'eta 0.75 + shock switch': dict(gaussianEta=0.75, shockSwitchC=3.0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--steps', type=int, default=0, help='0: the full run of the case')
    ap.add_argument('--cases', nargs='*', default=['noh', 'sedov'])
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--scheme', default='inutsuka', help='av_report config to start from')
    args = ap.parse_args()
    base = A.CONFIGS[args.scheme]
    out = Path(tempfile.mkdtemp(prefix='inutStab_'))
    print(f'{"variant":24s} ' + ' '.join(f'{c:>22s}' for c in args.cases))
    for name in args.variants:
        cells = []
        for case in args.cases:
            cfg = dataclasses.replace(base, name=name.replace(' ', '_'), diffusion={**base.diffusion, **VARIANTS[name]})
            rec = A.runOne(cfg, A.CASES[case], 'full', out, False, f'{name}_{case}'.replace(' ', '_'), dict(nSteps=args.steps) if args.steps else None)
            m = rec['metrics']
            key = {'noh': 'postShockRhoErr', 'sedov': 'peakRhoRatio'}.get(case)
            val = m.get(key) if key else None
            cells.append(('DIVERGED @%s' % m.get('nSteps', '?')) if m.get('diverged') else f'{key}={val:.4g}' if isinstance(val, float) else 'ok')
        print(f'{name:24s} ' + ' '.join(f'{c:>22s}' for c in cells), flush=True)


if __name__ == '__main__':
    main()
