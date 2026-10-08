#!/usr/bin/env python3
"""AV_PLAN sweep 2026-10-08 follow-up: why do Sphenix and Wadsley under-dissipate at a Sod shock?

The sweep found Sod L1(v_x) 2-3x C&D's for both, with a noisy post-shock plateau (velocity scatter 0.75-0.97
where C&D is flat). This probe runs variants of an av_report config with switch-parameter overrides and reports,
per variant, L1(v_x), the post-shock plateau noise (std of v_x over the plateau between the contact and the shock,
relative to the exact u* = 0.927), the peak alpha and the alpha at the shock; video on (av_report --profile full).

    python scripts/probe_shockUnderdissipation.py sphenix sphenix_ell=0.05 sphenix_ell=0.25 sphenix_ell=1.0
    python scripts/probe_shockUnderdissipation.py wadsley2017 wadsley_prefactor=0.5 wadsley_prefactor=2.0

Each extra argument is one variant (`KEY=VALUE[,KEY=VALUE]`, ViscositySwitchConfig fields); `-` is the config as is.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import av_report as A   # noqa: E402

U_STAR = 0.9274526   # Sod (1, 1) | (0.125, 0.1), gamma 1.4: post-shock velocity


def plateauNoise(res) -> dict:
    st = res.state.state
    x = st.positions[:, 0].detach().cpu().numpy()
    v = st.velocities[:, 0].detach().cpu().numpy()
    alpha = st.alphas.detach().cpu().numpy() if getattr(st, 'alphas', None) is not None else np.ones_like(x)
    t = float(res.state.t)
    # contact at x0 + u* t, shock at x0 + 1.752 t (gamma 1.4); x0 = 0.5 in the sod case's [0, 1] tube
    x0 = 0.5
    lo, hi = x0 + U_STAR * t, x0 + 1.7522 * t
    pad = 0.15 * (hi - lo)
    sel = (x > lo + pad) & (x < hi - pad)
    shock = (x > hi - 0.03) & (x < hi + 0.03)
    return dict(plateauStd=float(v[sel].std() / U_STAR) if sel.any() else float('nan'),
                plateauMean=float(v[sel].mean() / U_STAR) if sel.any() else float('nan'),
                alphaAtShock=float(alpha[shock].max()) if shock.any() else float('nan'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('variants', nargs='*', default=['-'])
    ap.add_argument('--cases', nargs='*', default=['sod'])
    ap.add_argument('--out', default='results/probe_shockUnderdissipation')
    args = ap.parse_args()
    base = A.CONFIGS[args.config]
    out = Path(args.out)
    rows = []
    for var in args.variants:
        sp = {}
        if var != '-':
            for kv in var.split(','):
                k, v = kv.split('=', 1)
                sp[k] = ast.literal_eval(v)
        cfg = dataclasses.replace(base, name=f'{args.config}[{var}]', switchParams={**base.switchParams, **sp})
        for case in args.cases:
            print(f'[probe] {cfg.name} / {case}', flush=True)
            cdef = A.CASES[case]
            res = A._execute(cdef.spec('full'), cfg, out, f'{args.config}_{var.replace("=", "").replace(",", "_")}_{case}',
                             True, cdef.avStride('full'), cdef.extra)
            m = cdef.metrics(res, cfg)
            row = dict(variant=var, case=case, diverged=m.get('diverged'), alphaMax=m.get('alphaMax'))
            if case == 'sod':
                row.update(L1_vx=m.get('L1_vx'), **plateauNoise(res))
            if case == 'sedov':
                row.update(shockRadiusErr=m.get('shockRadiusErr'), peakRhoRatio=m.get('peakRhoRatio'))
            if case == 'gresho':
                row.update(L1_vphi=m.get('L1_vphi'), alphaMean=m.get('alphaMean'))
            if case == 'kelvinHelmholtz':
                row.update(khAmplitudeAt1p5=m.get('khAmplitudeAt1p5'), khAmplitudeMax=m.get('khAmplitudeMax'))
            if case == 'noh2d':
                row.update(preShockAlphaMean=m.get('preShockAlphaMean'), postShockRhoErr=m.get('postShockRhoErr'))
            rows.append(row)
            print('[probe] RESULT ' + '  '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                                                for k, v in row.items()), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
