"""CRKSPH_LIMITER_PLAN P1: (eta_crit, eta_fold) parameter-stability map.

For each (case, eta_crit, eta_fold) runs the case under CRKSPH (no switch, `C_l`
= 1, float64) with the limiter constants overridden and records the harness
accuracy metrics plus KE(t_end)/KE0 - 1 and `ke_rebound`. Results are appended
to `<out>/p1.jsonl`, one JSON line per run; finished points are skipped, so an
interrupted sweep resumes. Runs are deterministic (CRK reproducibility fix,
2026-10-01), so no repeats. Metric A/B over many runs -> no video here; pick the
finalists and re-run them with `scripts/probe_greshoCusp.py`-style video.

    scripts/crk_limiter_p1.py --cases gresho:64 sod:800 --crit 0.24 0.25 0.28 0.3333 --fold 0.05 0.2
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('warpSPHCore_PRECISION', 'float64')
H = '/home/lu26029/dev/warpSPHCore/higherOrderSPH/harness/pde'
sys.path.insert(0, H)
sys.path.insert(0, os.path.dirname(H))

import run_crk_limiter_sweep as S      # noqa: E402
import derive_crk_limiter as D         # noqa: E402
from conservation import ke_rebound    # noqa: E402
from warpSPH.runner import run         # noqa: E402


def runPoint(name, over, crit, fold):
    case = D.load_case(name)
    inner = case.configureScheme

    def cfg(ctx):
        if inner is not None:
            inner(ctx)
        ctx.schemeConfig.crkViscosityParams.eta_crit = float(crit)
        ctx.schemeConfig.crkViscosityParams.eta_fold = float(fold)
    case2 = dataclasses.replace(case, configureScheme=cfg)
    ov = dict(S.SWEEP[name][0])
    ov.update(over)
    spec = D.build_spec(name, case2, nSteps=None, plot=False, show=False, store=False, video=False,
                        progress=False, quiet=True, device='cuda:0', **ov)
    t0 = time.perf_counter()
    res = run(case2, spec)
    wall = time.perf_counter() - t0
    st = res.state.state if hasattr(res.state, 'state') else res.state
    t = float(res.state.t)
    ke, E = res.series('kineticEnergy'), res.series('totalEnergy')
    out = dict(case=name, nx=ov.get('nx'), eta_crit=crit, eta_fold=fold, t=t, steps=int(res.nSteps),
               wall_s=round(wall, 1), diverged=bool(getattr(res, 'diverged', False)),
               e_drift=float((E[-1] - E[0]) / E[0]),
               ke_final_rel=float(ke[-1] / ke[0] - 1) if len(ke) and ke[0] else float('nan'),
               ke_rebound=ke_rebound(ke) if len(ke) else float('nan'))
    params = {**dict(case.params), **dict(getattr(getattr(res, 'ctx', None), 'spec', spec).params or {})}
    out.update(S.accuracy(name, st, t, params, res))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='+', default=['gresho:64', 'sod:800'], help='name[:nx]')
    ap.add_argument('--crit', type=float, nargs='+', required=True)
    ap.add_argument('--fold', type=float, nargs='+', required=True)
    ap.add_argument('--out', default='results/crk_limiter_p1')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'p1.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r['case'], r['nx'], round(r['eta_crit'], 6), round(r['eta_fold'], 6)))
    for spec in a.cases:
        name, _, nx = spec.partition(':')
        over = dict(nx=int(nx)) if nx else {}
        for fold in a.fold:
            for crit in a.crit:
                key = (name, over.get('nx', S.SWEEP[name][0].get('nx')), round(crit, 6), round(fold, 6))
                if key in done:
                    continue
                r = runPoint(name, over, crit, fold)
                print('[p1]', json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                with open(path, 'a') as f:
                    f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
