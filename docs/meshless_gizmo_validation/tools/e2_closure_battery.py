#!/usr/bin/env python
"""E2 closure A/B battery (COMPRESSIBLE_SPH_PLAN Track E): MFM / MFV, closure 'project' (the shipped
default) vs 'none' (GIZMO / Hopkins 2015: no closure projection), over the wave-2 sweep plus the Gresho
vortex, with accuracy metrics where an exact reference is cheap:

  sod / linearWave : L1 density error against the exact Riemann / analytic wave
  gresho           : L1 |v - v_exact| over the vortex profile, KE ratio
  everything       : finished, steps, ms/step, energy drift, max |v| / initial max |v|, KE ratio

    python .tmp/e2_closure_battery.py [--cases sod,gresho64,...] [--video]

Many tiny runs: video is OFF by default (CLAUDE.md allows it for a tight loop of tiny A/B runs); the
arms that come out worse are re-run with video afterwards. Progress rows stream per run; one flushed
`[ab]` line per run.
"""
import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

torch.set_grad_enabled(False)

from warpSPH.cases.compressible.sod import sodCase
from warpSPH.cases.compressible.greshoVortex import greshoVortexCase
from warpSPH.cases.compressible.noh import nohCase
from warpSPH.cases.compressible.shearingNoh import shearingNohCase
from warpSPH.cases.compressible.sedov import sedovCase
from warpSPH.cases.compressible.sodND import sod2dCase
from warpSPH.cases.compressible.linearWave import linearWaveCase
from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase
from warpSPH.cases.compressible.triplePoint import triplePointCase
from warpSPH.cases.compressible.shearBox import shearBoxCase
from warpSPH.cases.compressible.yeeVortex import yeeVortexCase
from warpSPH.caseUtils.compressible.greshoVortex.sample import greshoProfile
from warpSPH.caseUtils.compressible.sod.sodSolution import solve as sodSolve
import dataclasses
from warpSPH.runner import run

CASES = {
    'sod': (sodCase, dict(nx=200, tLimit=0.12)),
    'noh': (nohCase, dict(nx=100, tLimit=0.05)),
    'shearingNoh': (shearingNohCase, dict(nx=64, tLimit=0.1)),
    'sedov': (sedovCase, dict(dim=2, nx=45, tLimit=0.2)),
    'sod2d': (sod2dCase, dict(nx=64, tLimit=0.15)),
    'linearWave': (linearWaveCase, dict(nx=100, tLimit=1.0)),
    'kh64': (kelvinHelmholtzCase, dict(nx=64, tLimit=1.0)),
    'kh128': (kelvinHelmholtzCase, dict(nx=128, tLimit=3.0)),
    'triplePoint': (triplePointCase, dict(nx=32, tLimit=0.2)),
    'shearBox': (shearBoxCase, dict(nx=64, tLimit=1.0)),
    'yee': (yeeVortexCase, dict(params=dict(nr=16), nx=64, tLimit=2.0)),
    'gresho48': (greshoVortexCase, dict(nx=48, tLimit=3.0)),
    'gresho64': (greshoVortexCase, dict(nx=64, tLimit=3.0)),
    'gresho96': (greshoVortexCase, dict(nx=96, tLimit=3.0)),
}


def accuracy(name, result):
    """Case-specific L1 error against the exact solution (None where there is none here)."""
    ctx, p = result.ctx, result.state.state
    t = float(result.trajectory[-1]['t'])
    pos = p.positions.double().cpu()
    rho = p.densities.double().cpu()
    if name == 'sod':
        gamma = ctx.param('gamma')
        left = (ctx.param('left_pressure'), ctx.param('left_rho'), ctx.param('left_velocity'))
        right = (ctx.param('right_pressure'), ctx.param('right_rho'), ctx.param('right_velocity'))
        _, _, v = sodSolve(left_state=left, right_state=right, geometry=(0., 1., 0.5), t=t,
                           gamma=gamma, npts=2000, dustFrac=0.0)
        x = pos[:, 0].numpy()
        m = (x > 0.0) & (x < 1.0)
        ref = np.interp(x[m], v['x'], v['rho'])
        return dict(L1rho=float(np.mean(np.abs(rho.numpy()[m] - ref))))
    if name == 'linearWave':
        rho0, A, lam, cs = (ctx.param(k) for k in ('rho0', 'A', 'lamda', 'c_s'))
        ref = rho0 + A * torch.sin(2 * math.pi * pos[:, 0] / 1.0 - t * cs / lam * 2 * math.pi)
        return dict(L1rho_over_A=float((rho - ref).abs().mean() / abs(A)))
    if name.startswith('gresho'):
        dom = ctx.config.domain
        c = 0.5 * (dom.min + dom.max).double().cpu()
        d = pos - c
        r = d.norm(dim=-1).clamp_min(1e-12)
        vphi, _ = greshoProfile(r, 0.0)
        phi = torch.stack([-d[:, 1], d[:, 0]], dim=-1) / r[:, None]
        vex = vphi[:, None] * phi
        v = p.velocities.double().cpu()
        return dict(L1v=float((v - vex).norm(dim=-1).mean()))
    return {}


def one(name, scheme, closure, video, outRoot, cfl=None):
    case, kw = CASES[name]
    caseKw = dict(kw)
    cfgOver = dict(closure=closure)
    if cfl is not None:
        cfgOver['cfl'] = cfl

    def configureScheme(ctx):
        case.configureScheme(ctx)
        for k, v in cfgOver.items():
            setattr(ctx.schemeConfig, k, v)

    c = dataclasses.replace(case, configureScheme=configureScheme)
    runRoot = os.path.join(outRoot, f'{name}_{scheme}_{closure}')
    kw2 = dict(caseKw, scheme=scheme, precision='float32', quiet=True, progress=True,
               velocityAlarmPlotInterval=1, stallProgress=1e-3, exportRoot=runRoot)
    if video:
        kw2.update(plot=True, video=True, plotInterval=max(1, 20), plotBackend='vispy')
    t0 = time.perf_counter()
    r = run(c, **kw2)
    wall = time.perf_counter() - t0
    rows = r.trajectory
    p = r.state.state
    n = len(rows) - 1
    te = [x['totalEnergy'] for x in rows]
    ke = [x['kineticEnergy'] for x in rows]
    finite = all(bool(torch.isfinite(getattr(p, f).double()).all().item())
                 for f in ('positions', 'Q', 'supports') if getattr(p, f, None) is not None)
    mv = [x.get('maxVelocity', float('nan')) for x in rows]
    out = dict(t_end=float(rows[-1]['t']), steps=n, msPerStep=1e3 * wall / max(n, 1), finite=finite,
               drift=abs(te[-1] - te[0]) / abs(te[0]) if te[0] else float('nan'),
               keRatio=ke[-1] / ke[0] if ke[0] else float('nan'),
               vmaxRatio=max(mv) / mv[0] if mv and mv[0] else float('nan'),
               diverged=bool(r.diverged), stop=r.stopReason)
    out.update(accuracy(name, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--schemes', default='MFM,MFV')
    ap.add_argument('--closures', default='project,none')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--cfl', type=float, default=None)
    ap.add_argument('--out', default=str(REPO / 'results' / 'e2_closure_ab_2026-10-12'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.cases.split(','):
        for scheme in a.schemes.split(','):
            for closure in a.closures.split(','):
                tag = f'{name} {scheme} {closure}' + (f' cfl={a.cfl}' if a.cfl else '')
                print(f'=== {tag} ===', flush=True)
                try:
                    o = one(name, scheme, closure, a.video, a.out, a.cfl)
                    print('[ab] ' + tag + ' | ' + ' '.join(
                        f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in o.items()), flush=True)
                except Exception as e:  # keep going: a failed arm is a result
                    traceback.print_exc()
                    print(f'[ab] {tag} | FAILED {type(e).__name__}: {e}', flush=True)
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
