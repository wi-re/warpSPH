"""Calibrate the mDBC determinant-gate threshold for Wendland C4 against the
DualSPHysics-derived Wendland C2 value (`DELTASPH_VALIDATION_PLAN.md` Part 3).

Confirmed by an A/B on the Marrone H/Δx = 40 dam break (`probe_deltaSPHMarrone.py
--kernel Wendland2` vs the case default): `interpolateLiuLiu`'s borrowed
`determinantThreshold = 1e-3` (DualSPHysics' own `determlimit`, tuned to their
Wendland C2 kernel) regresses that case under this repo's default Wendland C4
kernel (vmax 38 vs a 5.86 baseline) but works cleanly under Wendland C2 at the
*same* support radius (n_h = 4 both ways -- verified support = 4·Δx in both
codebases, so this is a kernel-shape effect, not a search-radius one). This
probe computes `det(A_g)` for the same synthetic particle geometries under
both kernels to derive that shape-driven scale factor, so a C4 case can use a
properly-scaled threshold instead of DualSPHysics' literal constant.

Two reference geometries, both centred on a query "ghost" point at the origin,
support = n_h * dx for both kernels (n_h = 4.0, matching `cases/dambreak.py`):
  - 'healthy'    -- a half-disc of fluid neighbours: a normal near-wall mDBC
                    point with full one-sided support.
  - 'degenerate' -- a thin, nearly coplanar band of neighbours: the
                    dam-break-front-over-dry-bed pathology that the regression
                    showed Wendland C4 handles differently from C2.

`det(A_g)` is computed for each (geometry, kernel) combination and the C4/C2
ratio reported. If the ratio is similar for both geometries, a single scalar
correction (`determinantThreshold_C4 = 1e-3 * ratio`) is a faithful stand-in
for DualSPHysics' constant; if the two ratios diverge, a single scalar cannot
serve both directions (admit-healthy / reject-degenerate) and the degenerate
ratio should be used since under-rejecting is what caused the regression
(over-rejecting only pushes a point to the safe Shepard/rho0 fallback, not an
explosion).

Usage:  python scripts/probe_mdbcKernelDeterminantScale.py [--nh 4.0] [--dx 0.015]
"""
from __future__ import annotations
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--nh', type=float, default=4.0, help='support / dx, matching cases/dambreak.py n_h')
ap.add_argument('--dx', type=float, default=0.015, help='particle spacing (Marrone H/dx=40: H=0.6, dx=0.015)')
ap.add_argument('--jitter', type=float, default=0.1, help='fractional lattice jitter (of dx)')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')

import torch
import numpy as np
from warpSPHCore import (KernelFunctions, DomainDescription, OperationDirection,
                         OperationProperties, SupportScheme)
from warpSPH.modules.liu.wp_mat import computeLiuMatricesWarp

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dt = torch.float64
dx = args.dx
support = args.nh * dx
torch.manual_seed(args.seed)


class _S:                                                # minimal ParticleState
    pass


def _lattice(xmin, xmax, ymin, ymax, jitterFrac):
    xs = torch.arange(xmin, xmax, dx, device=dev, dtype=dt)
    ys = torch.arange(ymin, ymax, dx, device=dev, dtype=dt)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    pos = pos + (torch.rand_like(pos) - 0.5) * jitterFrac * dx
    return pos


def _makeState(pos):
    st = _S()
    st.positions = pos
    st.supports = torch.full((pos.shape[0],), support, device=dev, dtype=dt)
    st.masses = torch.full((pos.shape[0],), dx * dx, device=dev, dtype=dt)  # 2D areal mass, rho0=1
    st.densities = torch.ones(pos.shape[0], device=dev, dtype=dt)
    st.kinds = torch.zeros(pos.shape[0], device=dev, dtype=torch.int32)
    return st


def _domain():
    pad = support * 3
    return DomainDescription(
        torch.tensor([-pad, -pad], device=dev, dtype=dt),
        torch.tensor([pad, pad], device=dev, dtype=dt),
        torch.zeros(2, dtype=torch.bool, device=dev), 2)


class _C:
    pass


def _detA(pos, kernel):
    cfg = _C()
    cfg.kernel = kernel
    cfg.domain = _domain()
    cfg.dim = 2
    st = _makeState(pos)
    q = torch.zeros((1, 2), device=dev, dtype=dt)
    field = torch.ones(pos.shape[0], device=dev, dtype=dt)
    shep, b, A_g, nnbr = computeLiuMatricesWarp(
        queryPositions=q, referenceParticles=st, referenceQuantities=field,
        operationProperties=OperationProperties(
            kernel=cfg.kernel, supportMode=SupportScheme.Scatter,
            operationMode=OperationDirection.AllToAll),
        domain=cfg.domain)
    det = torch.linalg.det(A_g)
    return float(det[0]), int(nnbr[0])


# ------------------------------------------------------------------ geometries
# 'healthy': a half-disc of fluid on the y <= 0 side out to ~1.3 supports, so
# the query point (at the origin, on the interface) sees a full one-sided
# neighbourhood -- the normal, accepted mDBC case.
padded = 1.3 * support
lattice = _lattice(-padded, padded, -padded, 0.0, args.jitter)
r = torch.linalg.norm(lattice, dim=1)
healthy = lattice[r <= support]

# 'degenerate': fluid confined to a band only ~0.15 dx thick (a fraction of a
# particle spacing) spanning several supports in x -- the thin, nearly
# coplanar sheet sliding over the dry bed that broke the C4 case. All
# neighbours are within the support radius (band half-width << support).
bandHalfWidth = 0.15 * dx
xs = torch.arange(-padded, padded, dx, device=dev, dtype=dt)
ys = torch.arange(-bandHalfWidth, bandHalfWidth, dx * 0.3, device=dev, dtype=dt)
if ys.numel() == 0:
    ys = torch.zeros(1, device=dev, dtype=dt)
gx, gy = torch.meshgrid(xs, ys, indexing='ij')
band = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
band = band + (torch.rand_like(band) - 0.5) * args.jitter * dx
r = torch.linalg.norm(band, dim=1)
degenerate = band[r <= support]

print(f"support = {support:.4g}  (n_h={args.nh}, dx={dx:.4g})")
print(f"healthy geometry:    {healthy.shape[0]} neighbours within support")
print(f"degenerate geometry: {degenerate.shape[0]} neighbours within support")
print()

results = {}
for name, geo in [('healthy', healthy), ('degenerate', degenerate)]:
    for kname, kernel in [('Wendland2', KernelFunctions.Wendland2), ('Wendland4', KernelFunctions.Wendland4)]:
        det, nnbr = _detA(geo, kernel)
        results[(name, kname)] = det
        print(f"  {name:10s} {kname:10s}  det(A_g) = {det: .6e}   nnbr={nnbr}")
    print()

ratio_healthy = results[('healthy', 'Wendland4')] / results[('healthy', 'Wendland2')]
ratio_degenerate = results[('degenerate', 'Wendland4')] / results[('degenerate', 'Wendland2')]
print(f"C4/C2 determinant ratio -- healthy:    {ratio_healthy: .4e}")
print(f"C4/C2 determinant ratio -- degenerate: {ratio_degenerate: .4e}")
print()
print("Both fixed geometries sit ~15-35x BELOW 1e-3 for both kernels -- that")
print("band is too thin to be diagnostic; DualSPHysics' cutoff must be crossed")
print("by a marginal, not maximally-degenerate, geometry. Sweeping instead.")
print()

# ------------------------------------------------------------------ sweep
# Hold a fixed-density lattice (spacing dx) and grow the vertical half-width
# `w` of the neighbour band from ~coplanar to a full half-disc, i.e. directly
# parametrise "how thin is the sheet" -- the actual failure axis in the dam
# break (a front that thins as it slides over the dry bed). For each `w`,
# find where det(A_g) for Wendland2 crosses DualSPHysics' own 1e-3 cutoff --
# that crossing geometrically DEFINES what "marginal" means to `determlimit`
# -- then read off det(A_g) for Wendland4 at that same geometry. That value,
# not a ratio taken at an arbitrary reference point, is the correctly
# calibrated Wendland4 threshold: it is C4's determinant at the exact
# geometry C2's own accepted cutoff sits at.
print("=" * 68)
print("Sweep: band half-width w  ->  det(A_g), both kernels")
print("=" * 68)
ws = np.geomspace(0.02 * dx, support, 40)
detsC2, detsC4, nnbrs = [], [], []
for w in ws:
    xs = torch.arange(-padded, padded, dx, device=dev, dtype=dt)
    ys = torch.arange(-w, w, max(dx * 0.25, w * 0.3), device=dev, dtype=dt)
    if ys.numel() == 0:
        ys = torch.zeros(1, device=dev, dtype=dt)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    geo = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    geo = geo + (torch.rand_like(geo) - 0.5) * args.jitter * dx
    r = torch.linalg.norm(geo, dim=1)
    geo = geo[r <= support]
    d2, n2 = _detA(geo, KernelFunctions.Wendland2)
    d4, n4 = _detA(geo, KernelFunctions.Wendland4)
    detsC2.append(d2); detsC4.append(d4); nnbrs.append(n2)

detsC2 = np.array(detsC2); detsC4 = np.array(detsC4); nnbrs = np.array(nnbrs)
DUALSPHYSICS_DETERMLIMIT = 1e-3
print(f"{'w/dx':>8} {'nnbr':>5} {'det C2':>12} {'det C4':>12} {'C4/C2':>10}")
for w, n, d2, d4 in zip(ws, nnbrs, detsC2, detsC4):
    print(f"{w/dx:8.3f} {n:5d} {d2:12.4e} {d4:12.4e} {d4/max(d2,1e-300):10.3f}")

# Locate the crossing w* where det_C2(w) == 1e-3 (log-linear interpolation).
crossed = np.where(detsC2 >= DUALSPHYSICS_DETERMLIMIT)[0]
print()
if crossed.size == 0 or crossed[0] == 0:
    print("No clean crossing found in this sweep range -- widen --nh/--dx or the w range.")
else:
    i1 = crossed[0]; i0 = i1 - 1
    logD2 = np.log(detsC2[[i0, i1]]); logTarget = np.log(DUALSPHYSICS_DETERMLIMIT)
    t = (logTarget - logD2[0]) / (logD2[1] - logD2[0])
    wStar = ws[i0] + t * (ws[i1] - ws[i0])
    logD4 = np.log(detsC4[[i0, i1]])
    d4Star = float(np.exp(logD4[0] + t * (logD4[1] - logD4[0])))
    nStar = int(round(nnbrs[i0] + t * (nnbrs[i1] - nnbrs[i0])))
    print(f"C2 crosses det=1e-3 at w* = {wStar:.4g} = {wStar/dx:.3f} dx  (~{nStar} neighbours)")
    print(f"C4's det(A_g) at that SAME geometry = {d4Star:.4e}")
    print()
    print(f"=> RECOMMENDED determinantThreshold for Wendland4 (n_h={args.nh}): {d4Star:.4e}")
    print(f"   (vs. the naive fixed-reference-point ratios: "
          f"{DUALSPHYSICS_DETERMLIMIT * ratio_healthy:.4e} [healthy-ratio], "
          f"{DUALSPHYSICS_DETERMLIMIT * ratio_degenerate:.4e} [degenerate-ratio])")
