#!/usr/bin/env python
"""Inviscid Taylor-Green vortex (periodic, no walls) with the divergence stage as omniSPH's relaxed Jacobi and as the converged compact projection (`modules/incompressible/compactProjection.py`):
the kinetic energy of a flow that should not lose any (`nu = 0`). Closed preset of DFSPH2D for the compact run: `densitySolve=False`, `divergenceGauge='min'`, no free-surface Dirichlet.

    python scripts/probe_tgvCompactProjection.py [--nx 48] [--tLimit 2] [--nu 0] [--variants jacobi compact compact_nods]
"""
import argparse
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

VARIANTS = {
    'jacobi': dict(projection='jacobi', densitySolve=True),                          # the scheme as shipped
    'jacobi_nods': dict(projection='jacobi', densitySolve=False),
    'compact': dict(projection='compact', densitySolve=True),
    'compact_nods': dict(projection='compact', densitySolve=False, divergenceGauge='min'),
    'nods_noshift': dict(projection='compact', densitySolve=False, divergenceGauge='min', shifting='none'),
    'nods_fixed': dict(projection='compact', densitySolve=False, divergenceGauge='min', shifting='fixed'),
    'nods_fickian': dict(projection='compact', densitySolve=False, divergenceGauge='min', shifting='fickian'),
    'preset_fickian': dict(closedPreset=True, shifting='fickian'),
    'preset': dict(closedPreset=True),                                                  # DFSPH2D CLOSED_PRESET: compact, no density solve, fixed shift A 0.5, min gauge
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=48)
    ap.add_argument('--tLimit', type=float, default=2.0)
    ap.add_argument('--nu', type=float, default=0.0)
    ap.add_argument('--scheme', choices=('omniIncompressible', 'divergenceFree'), default='divergenceFree')
    ap.add_argument('--variants', nargs='+', default=['jacobi', 'compact', 'compact_nods'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-video', dest='video', action='store_false', default=True)
    a = ap.parse_args()
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    for name in a.variants:
        case = getCase('tgv')
        _cs = case.configureScheme

        def _cs2(ctx, _cs=_cs, name=name):
            _cs(ctx)
            for k, v in VARIANTS[name].items():
                setattr(ctx.schemeConfig, k, v)
            ctx.schemeConfig.freeSurface = False
        case.configureScheme = _cs2
        spec = CaseSpec(caseName=f'tgv-{name}', scheme=a.scheme, params={**case.params, 'nu': a.nu}).merged(**case.defaults).merged(
            scheme=a.scheme, nx=a.nx, n_h=4.0, tLimit=a.tLimit, plot=a.video, video=a.video, show=False, store=False, progress=True, quiet=True, plotInterval=20,
            velocityAlarmPlotInterval=1, stallProgress=1e-3, **({'exportRoot': os.path.join(a.out, name)} if a.out else {}))
        res = run(case, spec)
        t = np.asarray(res.series('t'))
        ke = np.asarray(res.series('kineticEnergy'))
        pick = [int(np.argmin(abs(t - x))) for x in (0.25, 0.5, 1.0, 1.5, 2.0) if x <= t[-1] + 1e-9]
        print(f'RESULT {name:14s} nx {a.nx} nu {a.nu}: KE/KE0 at t=' + '  '.join(f'{t[k]:.2f}: {ke[k] / ke[0]:.4f}' for k in pick) + f'   max|v| {float(np.nanmax(res.series("maxVelocity"))):.3f}  diverged {res.diverged}', flush=True)


if __name__ == '__main__':
    main()
