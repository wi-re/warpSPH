#!/usr/bin/env python
"""Direct project-vs-none comparison on the valid Yee vortex with a scheme-consistent IC (MFM, nx 64, cfl 0.1,
t = 2): per-particle error maps against the exact solution, same colour scales, plus |v| against radius."""
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / ".tmp"))
import numpy as np
import torch

torch.set_grad_enabled(False)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yee_valid
import ic_audit
from warpSPH.runner import run
from warpSPH.cases.compressible.yeeVortex import yeeVortexCase

JIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
SCHEME = sys.argv[2] if len(sys.argv) > 2 else 'MFM'
yee_valid.OPT.update(jitter=JIT, bg=0.0)
res = {}
for closure in ('project', 'none'):
    def build(ctx):
        system = yee_valid.buildSystem(ctx)
        st = system.state
        ic_audit.consistentIC(system, ctx.config, ctx.schemeConfig, st.densities.clone(), st.velocities.clone(),
                              st.pressures.clone())
        return system

    def conf(ctx, closure=closure):
        yeeVortexCase.configureScheme(ctx)
        ctx.schemeConfig.closure = closure
        ctx.schemeConfig.cfl = 0.1

    case = dataclasses.replace(yeeVortexCase, buildSystem=build, configureScheme=conf)
    r = run(case, scheme=SCHEME, nx=64, tLimit=2.0, precision='float32', quiet=True, progress=False)
    p = r.state.state
    t = float(r.trajectory[-1]['t'])
    rho_e, v_e, _ = yee_valid.exact(p.positions, t, r.ctx.config.domain, 0.0)
    res[closure] = dict(pos=p.positions.cpu().numpy(), dv=(p.velocities - v_e).norm(dim=-1).cpu().numpy(),
                        drho=(p.densities - rho_e).abs().cpu().numpy(), rr=p.positions.norm(dim=-1).cpu().numpy(),
                        vmag=p.velocities.norm(dim=-1).cpu().numpy(), vmag_e=v_e.norm(dim=-1).cpu().numpy())

ink, sec = '#0b0b0b', '#52514e'
fig = plt.figure(figsize=(15, 8.4), facecolor='#fcfcfb')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], left=0.03, right=0.97, top=0.88, bottom=0.08, wspace=0.12,
                      hspace=0.22)
vm = {'dv': np.percentile(np.concatenate([res[c]['dv'] for c in res]), 99.5),
      'drho': np.percentile(np.concatenate([res[c]['drho'] for c in res]), 99.5)}
names = {'dv': '|v - v_exact|', 'drho': '|rho - rho_exact|'}
for k, (c, lab) in enumerate((('project', 'closure = project (shipped)'), ('none', 'closure = none'))):
    for row, key in enumerate(('dv', 'drho')):
        a = fig.add_subplot(gs[row, k])
        sc = a.scatter(res[c]['pos'][:, 0], res[c]['pos'][:, 1], c=res[c][key], s=9, cmap='magma_r', vmin=0, vmax=vm[key])
        a.set_aspect('equal')
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f'{lab}\n{names[key]}  L1 = {res[c][key].mean():.2e}', color=ink, fontsize=10, loc='left')
        if k == 1:
            fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02, label=names[key] + '  (same scale in both columns)')
b = fig.add_subplot(gs[:, 2])
b.set_facecolor('#fcfcfb')
order = np.argsort(res['project']['rr'])
b.plot(res['project']['rr'][order], res['project']['vmag_e'][order], color=ink, lw=1.6, label='exact')
for c, col in (('project', '#2a78d6'), ('none', '#eb6834')):
    b.scatter(res[c]['rr'], res[c]['vmag'], s=5, color=col, alpha=0.5, label=c)
b.set_xlim(0, 4)
b.set_xlabel('r', color=sec)
b.set_ylabel('|v|', color=sec)
b.set_title('|v| of every particle vs radius, t = 2', color=ink, loc='left', fontsize=10)
for s_ in ('top', 'right'):
    b.spines[s_].set_visible(False)
b.legend(frameon=False, markerscale=3)
fig.suptitle(f'Yee vortex (periodic box filled, scheme-consistent IC), {SCHEME}, jitter {JIT}, nx 64, t = 2',
             color=ink, fontsize=12, x=0.03, ha='left')
out = REPO / 'results' / 'e2_ic_ab_2026-10-12' / f'yee_direct_{SCHEME}_j{JIT}.png'
fig.savefig(out, dpi=115, facecolor=fig.get_facecolor())
print(out, {c: (float(res[c]['dv'].mean()), float(res[c]['drho'].mean())) for c in res})
