#!/usr/bin/env python
"""Closure A/B (project vs none) on original vs scheme-consistent ICs, for the cases where `none` looked worse:
valid Yee (jittered lattice, steady), Gresho (jittered lattice), Sod.  Accuracy vs the exact solution; cfl 0.1;
video on.   python .tmp/ic_ab.py --case yee --nx 64 --jitter 0.1"""
import argparse, dataclasses, math, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / ".tmp"))
import numpy as np, torch
torch.set_grad_enabled(False)
import yee_valid, ic_audit
from warpSPH.runner import run
from warpSPH.cases.compressible.yeeVortex import yeeVortexCase
from warpSPH.cases.compressible.greshoVortex import greshoVortexCase
from warpSPH.cases.compressible.sod import sodCase
from warpSPH.caseUtils.compressible.greshoVortex.sample import greshoProfile
from warpSPH.caseUtils.compressible.sod.sodSolution import solve as sodSolve

PREC = os.environ.get('warpSPHCore_PRECISION', 'float32')

def wrap(case, consistent, closure, cfl):
    orig = case.buildSystem
    def buildSystem(ctx):
        system = orig(ctx)
        if consistent:
            st = system.state
            ic_audit.consistentIC(system, ctx.config, ctx.schemeConfig, st.densities.clone(), st.velocities.clone(), st.pressures.clone())
        return system
    def configureScheme(ctx):
        case.configureScheme(ctx)
        ctx.schemeConfig.closure = closure; ctx.schemeConfig.cfl = cfl
    return dataclasses.replace(case, buildSystem=buildSystem, configureScheme=configureScheme)

def metrics(name, r):
    ctx, p = r.ctx, r.state.state
    t = float(r.trajectory[-1]['t']); pos = p.positions.double().cpu(); vel = p.velocities.double().cpu()
    rho = p.densities.double().cpu(); m = p.Q[:, 0].double().cpu()
    if name == 'yee':
        rho_e, v_e, _ = yee_valid.exact(p.positions, t, ctx.config.domain, 0.0); rho_e, v_e = rho_e.cpu(), v_e.cpu()
        ke = lambda v: float((0.5 * m * (v ** 2).sum(-1)).sum())
        return dict(keR=ke(vel) / ke(v_e.double()), L1rho=float((rho - rho_e.double()).abs().mean()), L1v=float((vel - v_e.double()).norm(dim=-1).mean()))
    if name == 'gresho':
        dom = ctx.config.domain; c = (0.5 * (dom.min + dom.max)).double().cpu(); d = pos - c
        rr = d.norm(dim=-1).clamp_min(1e-12); vphi, _ = greshoProfile(rr, 0.0)
        vex = vphi[:, None] * torch.stack([-d[:, 1], d[:, 0]], -1) / rr[:, None]
        ke = lambda v: float((0.5 * m * (v ** 2).sum(-1)).sum())
        return dict(keR=ke(vel) / ke(vex), L1v=float((vel - vex).norm(dim=-1).mean()))
    if name == 'sod':
        left = (ctx.param('left_pressure'), ctx.param('left_rho'), ctx.param('left_velocity'))
        right = (ctx.param('right_pressure'), ctx.param('right_rho'), ctx.param('right_velocity'))
        _, _, v = sodSolve(left_state=left, right_state=right, geometry=(0., 1., 0.5), t=t, gamma=ctx.param('gamma'), npts=2000, dustFrac=0.0)
        x = pos[:, 0].numpy(); mk = (x > 0) & (x < 1)
        return dict(L1rho=float(np.mean(np.abs(rho.numpy()[mk] - np.interp(x[mk], v['x'], v['rho'])))))
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=['yee', 'gresho', 'sod'])
    ap.add_argument('--nx', type=int, default=64); ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--cfl', type=float, default=0.1); ap.add_argument('--schemes', default='MFM,MFV')
    ap.add_argument('--no-video', dest='video', action='store_false')
    ap.add_argument('--out', default=str(REPO / 'results' / 'e2_ic_ab_2026-10-12'))
    a = ap.parse_args()
    if a.case == 'yee':
        yee_valid.OPT.update(jitter=a.jitter, bg=0.0)
        base = dataclasses.replace(yeeVortexCase, buildSystem=yee_valid.buildSystem); kw = dict(tLimit=2.0)
    elif a.case == 'gresho':
        base = greshoVortexCase; kw = dict(tLimit=3.0, params=dict(greshoVortexCase.params, jitter=a.jitter))
    else:
        base = sodCase; kw = dict(tLimit=0.12, params=dict(sodCase.params, jitter=a.jitter))
    for scheme in a.schemes.split(','):
        for ic in ('orig', 'cons'):
            for closure in ('project', 'none'):
                case = wrap(base, ic == 'cons', closure, a.cfl)
                tag = f'{a.case}_nx{a.nx}_j{a.jitter}_{scheme}_{ic}_{closure}'
                rk = dict(kw, scheme=scheme, nx=a.nx, precision=PREC, quiet=True, progress=True, velocityAlarmPlotInterval=1,
                          stallProgress=1e-3, exportRoot=os.path.join(a.out, tag))
                if a.video: rk.update(plot=True, video=True, plotInterval=10, plotBackend='vispy')
                t0 = time.perf_counter()
                try:
                    r = run(case, **rk)
                except Exception as e:
                    print(f'[icab] {tag} | FAILED {type(e).__name__}: {e}', flush=True); continue
                n = len(r.trajectory) - 1
                met = metrics(a.case, r)
                print(f'[icab] {tag} | steps={n} ' + ' '.join(f'{k}={v:.4e}' if k != 'keR' else f'{k}={v:.4f}' for k, v in met.items()), flush=True)
main()
