#!/usr/bin/env python3
"""AV_PLAN Phase 2 detector map (§0.3): Sedov alpha(r) under the Rosswog (2020) entropy trigger vs
Cullen & Dehnen (2010), the comparison of rosswog2020entropy Fig. 4. Monaghan host, C_q = 2 (the
strong-shock family of the AV report), Sedov 3D at the AV report's full resolution by default.

    scripts/probe_rosswogSedovMap.py [--dim 3] [--nx 40] [--out results/av_M2_maps]

Writes `<out>/sedov_alpha_map.png` (alpha, the trigger scalar and rho against r, exact rho overlaid)
and `<out>/sedov_alpha_map.npz`. One run at a time, video on, progress streamed.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warpSPHBootstrap  # noqa: F401,E402
from warpSPH.runner import run  # noqa: E402

CONFIGS = {
    'Rosswog2020': dict(alpha_min=0.0, alpha_max=1.0),
    'CullenDehnen2010': {},          # its defaults, as in the AV report's cullenDehnen2010Q column
}


def _case(switchParams):
    from warpSPH.cases.sedov import sedovCase
    configure = sedovCase.configureScheme

    def configureScheme(ctx):
        configure(ctx)
        for k, v in switchParams.items():
            setattr(ctx.schemeConfig.viscositySwitchParams, k, v)
        ctx.schemeConfig.diffusionParams.C_q = 2.0
    return dataclasses.replace(sedovCase, configureScheme=configureScheme)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--nx', type=int, default=40)
    ap.add_argument('--out', default='results/av_M2_maps')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for switch, sp in CONFIGS.items():
        res = run(_case(sp), scheme='Monaghan', dim=a.dim, nx=a.nx, plot=True, video=True,
                  exportRoot=str(out / f'runs_{switch}'), progress=True, stallProgress=1e-3,
                  velocityAlarmPlotInterval=1, params=dict(viscositySwitch=switch))
        st = res.state.state
        r = np.linalg.norm(st.positions.detach().cpu().numpy(), axis=1)
        data[switch] = dict(r=r, alpha=st.alphas.detach().cpu().numpy(), rho=st.densities.detach().cpu().numpy(),
                            t=float(res.state.t))
        if st.entropyRates is not None:
            data[switch]['epsdot'] = st.entropyRates.detach().cpu().numpy()
        sol = res.ctx.scratch['solution']
        rr, _, _, rhoEx, *_ = sol.solution(float(res.state.t))
        data[switch]['rExact'], data[switch]['rhoExact'] = np.array(rr), np.array(rhoEx)
        print(f'[map] {switch}: t {res.state.t:.4f} alpha mean {data[switch]["alpha"].mean():.4f} '
              f'max {data[switch]["alpha"].max():.4f} peak rho {data[switch]["rho"].max():.3f}', flush=True)

    np.savez(out / 'sedov_alpha_map.npz', **{f'{s}_{k}': v for s, d in data.items() for k, v in d.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {'Rosswog2020': '#1f77b4', 'CullenDehnen2010': '#d62728'}
    for s, d in data.items():
        ax[0].scatter(d['r'], d['alpha'], s=1, alpha=0.4, color=colors[s], label=s)
        ax[2].scatter(d['r'], d['rho'], s=1, alpha=0.4, color=colors[s], label=s)
    d = data['Rosswog2020']
    ax[1].scatter(d['r'], np.maximum(d['epsdot'], 1e-10), s=1, alpha=0.4, color=colors['Rosswog2020'])
    for y in (1e-4, 5e-2):
        ax[1].axhline(y, color='k', lw=0.7, ls='--')
    ax[2].plot(d['rExact'], d['rhoExact'], 'k-', lw=1, label='exact')
    ax[0].set(xlabel='r', ylabel='alpha', title='dissipation parameter')
    ax[1].set(xlabel='r', ylabel='epsdot (Eq. 16)', yscale='log', ylim=(1e-8, 10),
              title='entropy trigger (eps_0, eps_1 dashed)')
    ax[2].set(xlabel='r', ylabel='rho', title=f'density, t = {d["t"]:.3f}')
    for x in ax:
        x.set_xlim(0, d['rExact'].max() * 1.4)
    ax[0].legend(markerscale=8)
    ax[2].legend(markerscale=8)
    fig.suptitle(f'Sedov {a.dim}D nx {a.nx}, Monaghan, C_q = 2: Rosswog (2020) vs Cullen & Dehnen (2010)')
    fig.tight_layout()
    fig.savefig(out / 'sedov_alpha_map.png', dpi=130)
    print(f'[map] wrote {out / "sedov_alpha_map.png"}', flush=True)


if __name__ == '__main__':
    main()
