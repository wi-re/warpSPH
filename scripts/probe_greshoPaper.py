"""Gresho vortex set up as in Frontiere et al. 2017, Sec. 4.4.1 / Fig. 12 (CRKSPH_LIMITER_PLAN P3, item 1).

Paper: gamma = 5/3, rho = 1, periodic unit box centred on the origin, 64^2 lattice, seventh-order kernel (eta_max = 4, ~50 neighbours
in 2D = our B7 with n_h = 4, H ~ 4.07 dx), CRKSPH with C_l = 2, C_q = 1, eps^2 = 1e-2, limiter (eta_crit, eta_fold) = (1/n_h, 0.2) in
units of dx = (0.25, 0.05) in our r/H units (CRKSPH_LIMITER_PLAN O2), no viscosity switch; v_phi(r) compared at t = 3 and t = 5 (Fig. 12:
"maintaining a near theoretical peak rotational velocity even as late as t = 5"). The paper reports no KE; this probe adds KE(t),
ke_rebound and L1 so the vortex can be judged on more than its peak.

Configs run one at a time with video: `paper` (above), `current` (our defaults: C_l = 1, (1/3, 0.2)), `candidate` (C_l = 1, (0.25, 0.03)).

    scripts/probe_greshoPaper.py --nx 64 --tLimit 5
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

from warpSPH.cases.greshoVortex import greshoVortexCase
from warpSPH.caseUtils.compressible.greshoVortex import greshoProfile
from warpSPH.runner import run

CONFIGS = {
    'paper': dict(C_l=2.0, C_q=1.0, crit=0.25, fold=0.05),
    'current': dict(C_l=1.0, C_q=1.0, crit=1 / 3, fold=0.2),
    'candidate': dict(C_l=1.0, C_q=1.0, crit=0.25, fold=0.03),
    # Spheral's own Gresho regression test (~/dev/spheral/tests/functional/Hydro/GreshoVortex): NBSpline order 5 (extent 3),
    # nPerh 1.51 -> support 4.53 dx, Cl = Cq = 1, limiter (1/nPerh, 0.2/nPerh) in r/h = (1/n_h, 0.2/n_h) in r/H with n_h = 4.53
    'spheral': dict(C_l=1.0, C_q=1.0, crit=1 / 4.53, fold=0.2 / 4.53, kernel='QuinticSpline', n_h=4.53),
    'b7n453': dict(C_l=1.0, C_q=1.0, crit=1 / 3, fold=0.2, kernel='B7', n_h=4.53),
    'quintic4': dict(C_l=1.0, C_q=1.0, crit=1 / 3, fold=0.2, kernel='QuinticSpline', n_h=4.0),
    'spheralCtrl': dict(C_l=1.0, C_q=1.0, crit=1 / 3, fold=0.2, kernel='QuinticSpline', n_h=4.53),
}
SNAP_TIMES = (3.0, 5.0)


def keRebound(ke):
    ke = np.asarray(ke, dtype=float)
    return float((ke - np.minimum.accumulate(ke)).max() / ke[0])


def profile(x, v, nb=40):
    """Radial bins of v_phi: (r, mean, std, max) for r < 0.6."""
    r = np.linalg.norm(x, axis=1)
    vphi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / np.maximum(r, 1e-30)
    edges = np.linspace(0, 0.6, nb + 1)
    idx = np.digitize(r, edges) - 1
    rc, mu, sd = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 2:
            rc.append(r[m].mean()); mu.append(vphi[m].mean()); sd.append(vphi[m].std())
    return np.array(rc), np.array(mu), np.array(sd), float(np.abs(vphi).max()), float(np.abs(vphi)[r < 0.5].mean())


def runConfig(name, c, nx, tLimit, out, video):
    inner = greshoVortexCase.configureScheme
    snaps = {}
    base = greshoVortexCase.diagnostics

    def cfg(ctx):
        inner(ctx)
        ctx.schemeConfig.diffusionParams.C_l = c['C_l']
        ctx.schemeConfig.diffusionParams.C_q = c['C_q']
        ctx.schemeConfig.crkViscosityParams.eta_crit = c['crit']
        ctx.schemeConfig.crkViscosityParams.meanVolumeWeights = bool(c.get('meanW', False))
        ctx.schemeConfig.crkViscosityParams.eta_fold = c['fold']

    def diag(ctx, state):
        row = base(ctx, state)
        t = float(state.t)
        for ts in SNAP_TIMES:
            if ts not in snaps and t >= ts - 5e-3:   # the last step lands just short of tLimit
                snaps[ts] = (state.state.positions.detach().cpu().numpy().copy(),
                             state.state.velocities.detach().cpu().numpy().copy())
        return row

    case = dataclasses.replace(greshoVortexCase, configureScheme=cfg, diagnostics=diag)
    over = {k: c[k] for k in ('kernel', 'n_h') if k in c}
    kw = dict(plot=True, video=True, exportRoot=str(out / 'runs' / name), velocityAlarmPlotInterval=1) if video else {}
    res = run(case, nx=nx, tLimit=tLimit, progress=True, stallProgress=1e-3, **over, **kw)
    ke = res.series('kineticEnergy')
    tKey = next((k for k in ('t', 'time') if k in res.trajectory[0]), None)
    tSeries = res.series(tKey) if tKey else np.linspace(0, float(res.state.t), len(ke))
    row = dict(config=name, **c, nx=nx, t=float(res.state.t), steps=int(res.nSteps), diverged=bool(res.diverged),
               keFinalRel=float(ke[-1] / ke[0] - 1), keMinRel=float(ke.min() / ke[0] - 1), keRebound=keRebound(ke))
    row['tSeries'] = tSeries
    prof = {}
    for ts, (x, v) in snaps.items():
        rc, mu, sd, peak, _ = profile(x, v)
        ex, _ = greshoProfile(torch.as_tensor(rc))
        r = np.linalg.norm(x, axis=1)
        vphi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / np.maximum(r, 1e-30)
        vex = greshoProfile(torch.as_tensor(r))[0].numpy()
        m = r < 0.5
        row[f'peakBinMean_t{ts:g}'] = float(mu.max())
        row[f'peakParticle_t{ts:g}'] = peak
        row[f'L1_t{ts:g}'] = float(np.abs(vphi - vex)[m].mean())
        row[f'scatter_t{ts:g}'] = float(sd[(rc > 0.1) & (rc < 0.45)].mean())
        prof[ts] = (rc, mu, sd)
    np.savez(out / f'{name}.npz', ke=ke, **{f'rc{ts:g}': p[0] for ts, p in prof.items()},
             **{f'mu{ts:g}': p[1] for ts, p in prof.items()}, **{f'sd{ts:g}': p[2] for ts, p in prof.items()})
    return row, ke, prof


def plot(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'paper': '#0072B2', 'current': '#D55E00', 'candidate': '#009E73', 'spheral': '#CC79A7', 'spheralCtrl': '#E69F00', 'b7n453': '#56B4E9', 'quintic4': '#999999'}
    colors = {**colors, **{k: '#444444' for k in results if k not in colors}}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rr = torch.linspace(1e-4, 0.6, 600, dtype=torch.float64)
    ex = greshoProfile(rr)[0].numpy()
    for k, ts in enumerate(SNAP_TIMES):
        ax[k].plot(rr, ex, color='black', lw=1.2, label='exact')
        for name, (row, ke, prof) in results.items():
            if ts in prof:
                rc, mu, sd = prof[ts]
                ax[k].errorbar(rc, mu, yerr=sd, fmt='o-', ms=3, lw=1, color=colors[name], elinewidth=0.6, alpha=0.9, label=name)
        ax[k].set_title(f'v_phi(r), t = {ts:g} (bin mean, std)'); ax[k].set_xlabel('r')
    for name, (row, ke, prof) in results.items():
        ax[2].plot(row['tSeries'], ke / ke[0], color=colors[name], lw=1.5, label=name)
    ax[2].axhline(1, color='black', lw=0.8); ax[2].set_title('KE(t)/KE(0)'); ax[2].set_xlabel('t')
    ax[0].legend(frameon=False)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'gresho_paper.png', dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--tLimit', type=float, default=5.0)
    ap.add_argument('--configs', nargs='+', default=None)
    ap.add_argument('--scaledNh', type=float, nargs='+', default=[],
                    help='extra configs nh<v>: B7, C_l = C_q = 1, limiter (1/v, 0.2/v) (Spheral scaling), n_h = v')
    ap.add_argument('--meanW', action='store_true', help="pair weights ((V_i + V_j)/2)^2 (Spheral) instead of V_i V_j")
    ap.add_argument('--out', default='results/probes/gresho_paper')
    ap.add_argument('--noVideo', dest='video', action='store_false')
    a = ap.parse_args()
    for v in a.scaledNh:
        CONFIGS[f'nh{v:g}'] = dict(C_l=1.0, C_q=1.0, crit=1 / v, fold=0.2 / v, kernel='B7', n_h=v)
    names = a.configs if a.configs else ([] if a.scaledNh else ['paper', 'current', 'candidate'])
    names = names + [f'nh{v:g}' for v in a.scaledNh]
    if a.meanW:
        for k in CONFIGS:
            CONFIGS[k]['meanW'] = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for name in names:
        row, ke, prof = runConfig(name, CONFIGS[name], a.nx, a.tLimit, out, a.video)
        results[name] = (row, ke, prof)
        rows.append(row)
        print('\n[gresho-paper]', json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items() if k != 'tSeries'}), flush=True)
        json.dump([{k: v for k, v in r.items() if k != 'tSeries'} for r in rows], open(out / 'summary.json', 'w'), indent=1)
        plot(out, results)


if __name__ == '__main__':
    sys.exit(main())
