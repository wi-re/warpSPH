"""Visualize the actual merged boundary SDF (value + central-difference
gradient, same eps as `_mergedSurface`) used by mDBC ghost placement for
Marrone 3.4, plus its zero-level isoline, over the full domain and zoomed on
the wedge toe and the fillet. Diagnostic only -- no time stepping.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import probe_deltaSPHMarrone34 as m34
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run
from warpSPH.configurations.region import RegionType

OUT = os.path.join('scratchpad', 'openchecks', 'm34_sdffield')
os.makedirs(OUT, exist_ok=True)

H = 1.0
params = m34._params('deltaSPH', cornerOnly=False)
params['machTarget'] = 1.95 / 28.3
r = run(dambreakCase, scheme='deltaSPH', L=m34.TANK_L, nx=256, nSteps=1,
        tLimit=1e9, quiet=True, store=False, progress=False, params=params)

dx = float(r.ctx.config.dx)
regions = r.ctx.schemeConfig.regions
solidSdfs = [reg.sdf for reg in regions if reg.type == RegionType.Boundary]
assert len(solidSdfs) == 1, solidSdfs
sdf = solidSdfs[0]

wall = m34.TANK_W / 2.0
bed = -m34.TANK_L / 2.0
apex = (-wall + 5.0 * H, bed + H)
toe = (-wall + 6.0 * H, bed)
backTop = (-wall + 7.0 * H, bed + H)

eps = 0.1 * dx  # same step _mergedSurface uses


_DEVICE = r.ctx.config.domain.min.device


def field(cx, cy, rad, n=400):
    gx = np.linspace(cx - rad, cx + rad, n)
    gy = np.linspace(cy - rad, cy + rad, n)
    GX, GY = np.meshgrid(gx, gy)
    P = torch.tensor(np.stack([GX.ravel(), GY.ravel()], axis=1), dtype=torch.float32, device=_DEVICE)

    def mval(p):
        return sdf(p)[0].reshape(-1).detach().cpu().numpy()

    D = mval(P).reshape(GX.shape)

    # central-difference gradient, same eps as _mergedSurface
    ox = torch.zeros_like(P); ox[:, 0] = eps
    oy = torch.zeros_like(P); oy[:, 1] = eps
    gxv = (mval(P + ox) - mval(P - ox)) / (2 * eps)
    gyv = (mval(P + oy) - mval(P - oy)) / (2 * eps)
    GXg = gxv.reshape(GX.shape)
    GYg = gyv.reshape(GX.shape)
    gmag = np.sqrt(GXg**2 + GYg**2)
    return GX, GY, D, GXg, GYg, gmag


def plotZone(name, cx, cy, rad, n=400, quiverStride=14):
    GX, GY, D, GXg, GYg, gmag = field(cx, cy, rad, n)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    ax = axes[0]
    im = ax.pcolormesh(GX, GY, D, shading='auto', cmap='RdBu', vmin=-rad*0.3, vmax=rad*0.3)
    ax.contour(GX, GY, D, levels=[0], colors='k', linewidths=1.4)
    fig.colorbar(im, ax=ax, label='sdf value')
    ax.set_title(f'{name}: merged SDF value + zero isoline')
    ax.set_aspect('equal')

    ax = axes[1]
    im = ax.pcolormesh(GX, GY, gmag, shading='auto', cmap='viridis', vmin=0, vmax=2)
    ax.contour(GX, GY, D, levels=[0], colors='w', linewidths=1.0)
    fig.colorbar(im, ax=ax, label='|grad sdf|  (should be ~1)')
    ax.set_title(f'{name}: gradient magnitude (eps={eps:.4g})')
    ax.set_aspect('equal')

    ax = axes[2]
    ax.contour(GX, GY, D, levels=[0], colors='k', linewidths=1.4)
    s = slice(None, None, quiverStride)
    ax.quiver(GX[s, s], GY[s, s], GXg[s, s], GYg[s, s], gmag[s, s],
              cmap='plasma', scale=30, width=0.003)
    ax.set_title(f'{name}: gradient direction (colour = magnitude)')
    ax.set_aspect('equal')

    for ax in axes:
        ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad)
        ax.grid(alpha=0.2)
    fig.suptitle(f'Marrone 3.4 merged boundary SDF -- {name}  (dx={dx:.4g})', fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, f'sdffield_{name.replace(" ", "_")}.png')
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print('->', p)
    # report gradient magnitude stats away from the singular vertex-ish region
    print(f'   {name}: |grad| min={gmag.min():.3f} max={gmag.max():.3f} '
          f'median={np.median(gmag):.3f}')


plotZone('overall domain', 0.0, bed + m34.TANK_L * 0.35, max(wall, m34.TANK_L * 0.5) + 0.5, n=500, quiverStride=22)
plotZone('wedge toe', toe[0], toe[1] + 0.3, 1.4 * H, n=450, quiverStride=16)
plotZone('fillet', wall - 0.6 * H, bed + 0.6 * H, 1.6 * H, n=450, quiverStride=16)
