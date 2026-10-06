#!/usr/bin/env python3
"""SUPPORT_SOLVER_PLAN step 1: static toy problems for the adaptive support solve (no time stepping).

    scripts/probe_supportSolver.py --toy lattice     # uniform lattice 1D/2D/3D, h started at 0.5x / 1x / 2x
    scripts/probe_supportSolver.py --toy step        # 1D density step 1:8 (equal masses), h profile across the jump
    scripts/probe_supportSolver.py --toy surface     # 2D half-space (free surface), outer-row h

Every toy runs the Owen solve and the Monaghan (Newton) solve from the same start for `--iters` iterations
(threshold 0, so all iterations run) and reports h against
  * h_lat   = the lattice support n_h dx (what a uniform lattice of the local spacing should get), and
  * h_vol   = n_h (m / rho)^(1/d) with rho the SPH density summed at that h (the clamp's bound, Newton's target),
plus rho / rho_lattice. Static solves are not simulations, so no video; figures go to `--out`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warpSPHBootstrap  # noqa: F401,E402
from warpSPH.configurations import buildConfig, CompressibleSPHConfig  # noqa: E402
from warpSPH.enumTypes import AdaptiveSupportScheme  # noqa: E402
from warpSPH.modules.adaptiveSupport.optimalSupport import evaluateOptimalSupport  # noqa: E402
from warpSPH.systems import CompressibleState  # noqa: E402
from warpSPH.utils.domain import buildDomainDescription  # noqa: E402
from warpSPHCore import KernelFunctions, SupportScheme  # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def makeConfig(dim, periodic=True, kernel=KernelFunctions.B7, n_h=4.0, l=2.0):
    domain = buildDomainDescription(l, dim, periodic=periodic, device=DEV, dtype=torch.float32)
    cfg, _ = buildConfig(dim=dim, domain=domain, kernel=kernel, n_h=n_h, device=DEV, dtype=torch.float32)
    cfg.supportVolumeClamp = 'off'          # look at the solvers themselves
    return cfg


def makeState(x, m, h, kinds=None):
    n = x.shape[0]
    z = torch.zeros(n, device=DEV)
    return CompressibleState(
        positions=x.to(DEV), velocities=torch.zeros_like(x, device=DEV), supports=h.to(DEV).clone(),
        masses=m.to(DEV), densities=torch.ones(n, device=DEV),
        kinds=(torch.zeros(n, dtype=torch.int32, device=DEV) if kinds is None else kinds.to(DEV)),
        materials=torch.zeros(n, dtype=torch.int32, device=DEV),
        UIDs=torch.arange(n, dtype=torch.int32, device=DEV), UIDcounter=n,
        internalEnergies=z.clone(), totalEnergies=z.clone(), entropies=z.clone(), pressures=z.clone(),
        soundspeeds=z.clone(), divergence=z.clone(), alpha0s=z.clone(), alphas=z.clone())


TABLE = ['shell']


def solve(state, cfg, scheme, iters):
    comp = CompressibleSPHConfig(adaptiveSupportScheme=scheme, adaptiveSupportIterations=iters,
                                 adaptiveSupportThreshold=0.0, adaptiveSupportCorrections=False)
    comp.owenTable = TABLE[0]
    rho, h, *_ = evaluateOptimalSupport(state, cfg, comp, SupportScheme.Gather, None)
    return rho, h


def hVol(cfg, m, rho):
    return cfg.n_h * (m / rho) ** (1.0 / cfg.domain.dim)


def lattice(dim, nx, l=2.0):
    dx = l / nx
    ax = torch.arange(nx, dtype=torch.float32) * dx - l / 2 + dx / 2
    grids = torch.meshgrid(*([ax] * dim), indexing='ij')
    return torch.stack([g.reshape(-1) for g in grids], dim=1), dx


def toyLattice(a):
    print('== uniform periodic lattice: h / h_lat, h / h_vol, rho / rho_lat (median over particles)')
    for dim, nx in ((1, 200), (2, 64), (3, 24)):
        cfg = makeConfig(dim)
        x, dx = lattice(dim, nx)
        m = torch.full((x.shape[0],), dx ** dim)
        hLat = cfg.n_h * dx
        for start in (0.5, 1.0, 2.0):
            for scheme in (AdaptiveSupportScheme.Owen, AdaptiveSupportScheme.Monaghan):
                st = makeState(x, m, torch.full((x.shape[0],), start * hLat))
                rho, h = solve(st, cfg, scheme, a.iters)
                hv = hVol(cfg, st.masses, rho)
                print(f'  {dim}D start {start:3.1f}x {scheme.name:8s}  h/h_lat {(h / hLat).median():.5f}  '
                      f'h/h_vol {(h / hv).median():.5f}  rho/rho_lat {rho.median():.5f}  '
                      f'spread(h) {(h.max() - h.min()) / h.mean():.1e}', flush=True)


def toyStep(a):
    print('== 1D density step 1:8, equal masses, periodic (two jumps); profile near x = 0')
    cfg = makeConfig(1)
    dxL, nL = 0.005, 200            # dense half [-1, 0)
    dxR = 8 * dxL                   # sparse half [0, 1)
    xL = -1 + dxL / 2 + torch.arange(nL) * dxL
    xR = dxR / 2 + torch.arange(int(1 / dxR)) * dxR
    x = torch.cat([xL, xR]).unsqueeze(1)
    m = torch.full((x.shape[0],), dxL)
    hLat = cfg.n_h * torch.where(x[:, 0] < 0, torch.tensor(dxL), torch.tensor(dxR))
    out = {}
    for scheme in (AdaptiveSupportScheme.Owen, AdaptiveSupportScheme.Monaghan):
        st = makeState(x, m, hLat.clone())
        rho, h = solve(st, cfg, scheme, a.iters)
        hv = hVol(cfg, st.masses, rho)
        out[scheme.name] = (h.cpu(), hv.cpu(), rho.cpu())
    xs = x[:, 0]
    sel = torch.nonzero((xs > -0.06) & (xs < 0.2)).squeeze(1)
    print(f'  {"x":>8} {"h_lat":>7} | ' + ' | '.join(f'{s:^30s}' for s in out))
    print(f'  {"":>8} {"":>7} | ' + ' | '.join(f'{"h/h_lat":>9} {"h/h_vol":>9} {"rho":>9}' for _ in out))
    for i in sel.tolist():
        cols = ' | '.join(f'{(h[i] / hLat[i]).item():9.4f} {(h[i] / hv[i]).item():9.4f} {rho[i].item():9.4f}'
                          for h, hv, rho in out.values())
        print(f'  {xs[i].item():8.4f} {hLat[i].item():7.4f} | {cols}')
    _plot(a, 'step', xs, hLat, out)


def toySurface(a):
    print('== 2D half-space: lattice in x < 0 (non-periodic, empty x > 0), periodic in y; rows from the surface')
    dim, dx = 2, 2.0 / 64
    domain = buildDomainDescription(2.0, dim, periodic=False, device=DEV, dtype=torch.float32)
    domain.periodic[1] = True
    cfg, _ = buildConfig(dim=dim, domain=domain, kernel=KernelFunctions.B7, n_h=4.0, device=DEV, dtype=torch.float32)
    cfg.supportVolumeClamp = 'off'
    xs = -dx / 2 - torch.arange(24) * dx          # 24 rows below the surface at x = 0
    ys = -1 + dx / 2 + torch.arange(64) * dx
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    m = torch.full((x.shape[0],), dx ** dim)
    hLat = cfg.n_h * dx
    row = torch.round((-x[:, 0] - dx / 2) / dx).long()
    for scheme in (AdaptiveSupportScheme.Owen, AdaptiveSupportScheme.Monaghan):
        st = makeState(x, m, torch.full((x.shape[0],), hLat))
        rho, h = solve(st, cfg, scheme, a.iters)
        hv = hVol(cfg, st.masses, rho)
        print(f'  {scheme.name}: row  h/h_lat  h/h_vol  rho')
        for r in range(0, 8):
            k = row.to(h.device) == r
            print(f'      {r:3d}  {(h[k] / hLat).mean():7.4f}  {(h[k] / hv[k]).mean():7.4f}  {rho[k].mean():7.4f}', flush=True)


def _plot(a, name, xs, hLat, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for s, (h, hv, rho) in out.items():
        ax[0].plot(xs, h / hLat.cpu(), '.', ms=3, label=f'{s}: h / h_lat')
        ax[1].plot(xs, h / hv, '.', ms=3, label=f'{s}: h / h_vol')
    for x_ in ax:
        x_.axhline(1, color='k', lw=0.6)
        x_.legend()
        x_.set_xlabel('x')
    fig.tight_layout()
    fig.savefig(Path(a.out) / f'{name}.png', dpi=120)
    print(f'  figure -> {Path(a.out) / f"{name}.png"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--toy', choices=('lattice', 'step', 'surface'), required=True)
    ap.add_argument('--iters', type=int, default=64)
    ap.add_argument('--out', default='results/support_solver/toys')
    ap.add_argument('--owenTable', choices=('shell', 'lattice'), default='shell')
    a = ap.parse_args()
    TABLE[0] = a.owenTable
    {'lattice': toyLattice, 'step': toyStep, 'surface': toySurface}[a.toy](a)


if __name__ == '__main__':
    main()
