#!/usr/bin/env python3
"""How much does the adaptive-support solve over-grow h in a pure-fluid run (OPEN_PROBLEMS §20)?

Runs a case with `supportVolumeClamp` off and reports, every `--every` steps, the ratio of each
fluid row's h to the clamp's bound `n_h (m/rho)^(1/d)`: the fraction of rows the clamp would cap
(ratio > 1) and the ratio's p50/p99/max. Owen's relaxation is a one-way ratchet (COMPRESSIBLE_WALLS_PLAN
2026-10-05), so any ratio > 1 is h left over-large by an earlier state.

    scripts/probe_supportClampBinding.py --case gresho --switch CullenDehnen2010
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warpSPHBootstrap  # noqa: F401,E402
from warpSPH.runner import run  # noqa: E402
from warpSPH.utils.support import nH_to_n_h  # noqa: E402

CASES = {
    'gresho': ('warpSPH.cases.greshoVortex', 'greshoVortexCase', dict(nx=100, tLimit=3.0), {}),
    'kelvinHelmholtz': ('warpSPH.cases.kelvinHelmholtz', 'kelvinHelmholtzCase', dict(nx=128, tLimit=1.5), {}),
    'sod': ('warpSPH.cases.sod', 'sodCase', dict(nx=400, nSteps=400), dict(right_rho=0.125, right_pressure=0.1)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=CASES, default='gresho')
    ap.add_argument('--switch', default='CullenDehnen2010')
    ap.add_argument('--clamp', choices=('always', 'walls', 'off'), default='off')
    ap.add_argument('--every', type=int, default=100)
    ap.add_argument('--out', default='results/probes/supportClampBinding')
    a = ap.parse_args()
    mod, name, kw, params = CASES[a.case]
    case = getattr(__import__(mod, fromlist=[name]), name)
    configure, diagnostics = case.configureScheme, case.diagnostics

    def configureScheme(ctx):
        configure(ctx)
        ctx.config.supportVolumeClamp = a.clamp

    step = [0]

    def diag(ctx, state):
        row = diagnostics(ctx, state)
        if step[0] % a.every == 0:
            st = state.state
            dim = ctx.config.domain.dim
            bound = nH_to_n_h(ctx.config.targetNeighbors, dim) * (st.masses / st.densities) ** (1.0 / dim)
            r = (st.supports / bound)[st.kinds == 0].float()
            q = torch.quantile(r, torch.tensor([0.5, 0.99], device=r.device))
            print(f'[clamp] step {step[0]:5d} t {float(state.t):.4f}  h/bound > 1: {(r > 1 + 1e-6).float().mean():6.1%}  '
                  f'p50 {q[0]:.4f}  p99 {q[1]:.4f}  max {r.max():.4f}', flush=True)
        step[0] += 1
        return row

    case = dataclasses.replace(case, configureScheme=configureScheme, diagnostics=diag)
    run(case, scheme='Monaghan', plot=True, video=True, progress=True, stallProgress=1e-3,
        velocityAlarmPlotInterval=1, exportRoot=f'{a.out}/{a.case}_{a.switch}_{a.clamp}',
        params=dict(params, viscositySwitch=a.switch), **kw)


if __name__ == '__main__':
    main()
