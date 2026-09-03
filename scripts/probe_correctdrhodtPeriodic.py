"""`correctdrhodt` where it is *supposed* to matter: a periodic
weakly-compressible box with no free surface (Sun et al. 2019 TC1 / Fig. 9).

The δu-terms of Eq. (9)-(10) exist to keep `Σ_i V_i` (V_i = m_i/ρ_i) from
drifting when particles are advected by `u + δu` instead of `u`. On the square
patch that drift is tiny anyway (§3.3) and the term only got a chance to
misbehave because the arms fragment. Here there is no surface and no
fragmentation, so this is the clean test: does `correctdrhodt` reduce the
volume drift (as the paper shows) without destabilising ρ?

Runs `kolmogorov --scheme deltaSPH` with `correctdrhodt` off then on, and
reports the volume drift `ε_V = |Σ V_i / Σ V_i(0) − 1|` and the ρ bounds over
time.

Usage:
  python scripts/probe_correctdrhodtPeriodic.py [--nx 64] [--tLimit 3.0]
      [--every 200]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--tLimit', type=float, default=3.0)
parser.add_argument('--every', type=int, default=200)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases.kolmogorov import kolmogorovCase as case
from warpSPH.runner import run


def make_diag():
    ref = {}

    def diag(ctx, system):
        p = system.state
        fluid = p.kinds == 0
        V = (p.masses[fluid] / p.densities[fluid]).sum().item()
        ref.setdefault('V0', V)
        return {
            'epsV': abs(V / ref['V0'] - 1.0),
            'minRho': p.densities[fluid].min().item(),
            'maxRho': p.densities[fluid].max().item(),
            'sumV': V,
        }

    return diag


print(f"{'correctdrhodt':>13} {'step':>6} {'t':>7} {'epsV %':>10} {'minRho':>9} {'maxRho':>9}")
for correct in (False, True):
    _baseCfg, _baseDiag = case.configureScheme, case.diagnostics

    def _cfg(ctx, correct=correct):
        _baseCfg(ctx)
        ctx.schemeConfig.shiftProperties.correctdrhodt = correct

    case.configureScheme = _cfg
    case.diagnostics = make_diag()
    try:
        r = run(case, nx=args.nx, tLimit=args.tLimit, nSteps=None,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        case.configureScheme, case.diagnostics = _baseCfg, _baseDiag

    rows = [row for row in r.trajectory if 'epsV' in row and row.get('step', -1) >= 0]
    show = [row for row in rows if row['step'] % args.every == 0]
    if rows and rows[-1] not in show:
        show.append(rows[-1])
    for row in show:
        print(f"{str(correct):>13} {row['step']:>6} {row['t']:>7.4f} "
              f"{100 * row['epsV']:>10.4f} {row['minRho']:>9.5f} {row['maxRho']:>9.5f}")
    print(f"{'':>13}  diverged={r.diverged}")
