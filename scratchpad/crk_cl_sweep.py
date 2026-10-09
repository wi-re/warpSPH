"""CRKSPH artificial-viscosity linear coefficient A/B (AV follow-up of OPEN_PROBLEMS §15): C_l in {1 (ours), 2 (Frontiere Table D.1, B7)},
C_q = 1, on Gresho (nx 64, 96), Sod (nx 800), reusing the limiter-plan harness's run/metrics. Metric A/B: no video (many small runs)."""
import dataclasses, json, os, sys, time
os.environ.setdefault('warpSPHCore_PRECISION', 'float64')
H = '/home/lu26029/dev/warpSPHCore/higherOrderSPH/harness/pde'
sys.path.insert(0, H); sys.path.insert(0, os.path.dirname(H))
import torch
import run_crk_limiter_sweep as S
import derive_crk_limiter as D
from warpSPH.runner import run
from conservation import ke_rebound

def run_cl(name, Cl, Cq, **over):
    case = D.load_case(name); inner = case.configureScheme
    def cfg(ctx):
        if inner is not None: inner(ctx)
        ctx.schemeConfig.diffusionParams.C_l = float(Cl); ctx.schemeConfig.diffusionParams.C_q = float(Cq)
    case2 = dataclasses.replace(case, configureScheme=cfg)
    ov = dict(S.SWEEP[name][0]); ov.update(over)
    spec = D.build_spec(name, case2, nSteps=None, plot=False, show=False, store=False, video=False, progress=False, quiet=True,
                        device='cuda:0', **ov)
    t0 = time.perf_counter(); res = run(case2, spec); wall = time.perf_counter() - t0
    st = res.state.state if hasattr(res.state, 'state') else res.state
    t = float(res.state.t); ke = res.series('kineticEnergy'); E = res.series('totalEnergy')
    out = dict(case=name, nx=ov.get('nx'), C_l=Cl, C_q=Cq, t=t, steps=int(res.nSteps), wall_s=round(wall, 1),
               diverged=bool(getattr(res, 'diverged', False)), e_drift=float((E[-1] - E[0]) / E[0]),
               ke_final_rel=float(ke[-1] / ke[0] - 1) if len(ke) and ke[0] else float('nan'),
               ke_rebound=ke_rebound(ke) if len(ke) else float('nan'))
    params = {**dict(case.params), **dict(getattr(getattr(res, 'ctx', None), 'spec', spec).params or {})}
    out.update(S.accuracy(name, st, t, params, res))
    return out

rows = []
JOBS = [('gresho', dict(nx=64), Cl, 1.0) for Cl in (1.0, 2.0)] + [('gresho', dict(nx=96), Cl, 1.0) for Cl in (1.0, 2.0)] + [('sod', dict(nx=800), Cl, 1.0) for Cl in (1.0, 2.0)]
if os.environ.get('EXTRA'):
    JOBS = [('gresho', dict(nx=64), 0.0, 0.0), ('gresho', dict(nx=64), 0.0, 1.0), ('gresho', dict(nx=64), 0.5, 1.0)]
for name, over, Cl, Cq in JOBS:
    if True:
        r = run_cl(name, Cl, Cq, **over); rows.append(r)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(rows, open('scripts/out_crk2d/cl/cl_sweep%s.json' % ('_extra' if os.environ.get('EXTRA') else ''), 'w'))
