#!/usr/bin/env python
"""Prototype of Band et al. 2018 ("MLS pressure boundaries for divergence-free
and viscous SPH fluids", literature/band2018_mls-pressure-boundaries.pdf) on
the exact toy geometries that
`scripts/probe_boundaryExtrapolationSwitchDiscontinuity.py` used to
demonstrate a real, physically-sized discontinuity in this codebase's
MLS-vs-Shepard wall extrapolation for few-particle boundary stencils.

Recap of why that discontinuity happens (see the sibling probe's docstring
and the DELTASPH_VALIDATION_PLAN.md 5.2.3 history it cites): this codebase's
`interpolateLiuLiu` builds one (dim+1)x(dim+1) moment matrix `A_g` mixing
kernel-VALUE sums (row 0, the Shepard/value equation) with kernel-GRADIENT
sums (rows 1..d, the SPH-consistent-gradient equations), evaluated around a
ghost node offset into the fluid. Its 0th- and 1st-order terms are coupled
through that matrix, so a single scalar gate on the WHOLE matrix's
determinant decides between two SEPARATELY-COMPUTED formulas (the full fit,
or a plain Shepard fallback) -- and because the gate is on a determinant of a
coupled matrix, one new neighbour entering a sparse stencil's support radius
can jump it by orders of magnitude in a single infinitesimal geometric step
(measured directly: `line-10` in the sibling probe, det(A_g) 9.5e-4 -> 3.3e-3
between two samples 0.006 dx apart -- 3.5x past the *existing* smooth-ramp's
entire blend width, so the "already fixed" ramp gives zero protection there).

Band et al.'s system is structurally different in the two ways that matter:

  1. It fits directly AT the boundary particle b (Eq. 9, Fig. 2) from its own
     fluid neighbours -- no ghost node, no separate Taylor-correction
     transport step that could amplify gradient noise over an offset.
  2. It translates the moment system by the neighbourhood's own KERNEL-
     WEIGHTED CENTROID `d_b` (Eqs. 13-15). This makes Sum_f x~_f V_f W_f = 0
     EXACTLY (Eq. 17), which decouples the 4x4 system into a 1x1 block (the
     0th-order Shepard term `alpha_b`, Eq. 19 -- exact and well-posed
     whenever there is at least one neighbour, full stop, not a separately-
     computed fallback formula) and a 3x3 block (here, in 2D, 2x2) for the
     gradient `(beta_b, gamma_b)` alone (Eq. 20). That reduced block is
     `Sum_f x~_f x~_f^T V_f W_f` -- a literal weighted-least-squares normal-
     equations matrix, symmetric PSD by construction, degenerate ONLY when
     the neighbourhood is truly collinear/coplanar, and Band et al.
     themselves reach for SVD-based "safe inversion" exactly there (their
     Section 5) rather than a hard determinant threshold on the full system.

This probe reimplements that: `wendland2` (matching warpSPHCore's own
constant, `W(r,h) = 7/(pi h^2) (1-q)^4(1+4q)`, cutoff at q=r/h=1 -- verified
against `warpSPHCore/kernels/kernelFunctions/wendland2.py`), the centroid
transform, and the reduced-block solve via a Tikhonov-damped eigendecomposition
(smooth per-mode degradation, one further step beyond the paper's own plain
SVD truncation) as well as plain `torch.linalg.pinv` (closer to the paper's
literal "safe inversion via SVD"). Both are compared against this codebase's
CURRENT production `computeMdbcDensity` (already-ramped) and
`wallPressureExtrapolation('mls')` (still hard-switched) on the identical
sheet-scatter sweep and the identical discrete-neighbour-entry crossing
(`line-10`) that broke the existing ramp.

    python scripts/probe_bandMlsPressureBoundary.py
"""
from __future__ import annotations

import argparse
import os
import zlib

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--c0', type=float, default=40.0)
ap.add_argument('--supportRatio', type=float, default=4.0)
ap.add_argument('--layers', type=int, default=5)
ap.add_argument('--rhoFluid', type=float, default=1.03)
ap.add_argument('--rhoGrad', type=float, default=0.02)
ap.add_argument('--fieldNoise', type=float, default=0.003,
                help='IID per-particle density noise (fraction of rho0), fixed per '
                     'particle across the sweep -- a noise-free exactly-linear field '
                     'lets a near-singular inversion recover the true gradient EXACTLY '
                     'regardless of conditioning (checked directly: it does), which is '
                     'not how real SPH data behaves and hides the actual failure mode '
                     '(amplifying noise, not reconstructing signal). 0 = disable.')
ap.add_argument('--lineCounts', type=int, nargs='+', default=[6, 7, 8, 10, 13, 18])
ap.add_argument('--nDist', type=int, default=240)
ap.add_argument('--jitterAmp', type=float, default=1.5)
ap.add_argument('--tikhonovLambda', type=float, default=1e-3,
                help='Tikhonov damping (in units of the reduced-block eigenvalue '
                     'scale) for the smooth per-mode gradient-block inverse')
ap.add_argument('--out', default=os.path.join('scratchpad', 'band_mls_prototype'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')

import numpy as np
import torch

from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList, OperationDirection)
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.modules.liu import determinantThresholdFor
from warpSPH.modules.incompressible.wallPressure import wallPressureExtrapolation

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
dx = args.dx
RHO0 = 1.0
support = args.supportRatio * dx
NHAT = np.array([0.0, 1.0])
THAT = np.array([1.0, 0.0])


class _Cfg:
    pass


def shapeOffsets(name):
    if name.startswith('line'):
        n = int(name.split('-')[1])
        c = (n - 1) / 2.0
        return [(i - c, 0.0) for i in range(n)]
    raise ValueError(name)


def baseCluster(shape, distDx):
    offs = shapeOffsets(shape)
    return np.array([(distDx + n) * dx * NHAT + t * dx * THAT for (t, n) in offs])


def buildState(fluidPositions, fieldNoise=None):
    """Identical construction to probe_boundaryExtrapolationSwitchDiscontinuity.py
    (ghost-mirror sign corrected relative to probe_mdbcSparseFluid.py)."""
    pos, vel, kinds, goff = [], [], [], []
    nSpan = int(np.ceil((6 * support) / dx)) + 4
    for k in range(args.layers):
        base = -(k + 0.5) * dx * NHAT
        for s in range(-nSpan, nSpan + 1):
            p = base + s * dx * THAT
            pos.append(p.tolist()); vel.append([0.0, 0.0]); kinds.append(1)
            goff.append((-(2 * k + 1) * dx * NHAT).tolist())
    nBnd = len(pos)

    ghostStart = nBnd
    for bi in range(nBnd):
        b = np.array(pos[bi]); o = np.array(goff[bi])
        pos.append((b - o).tolist()); vel.append([0.0, 0.0]); kinds.append(2)
        goff.append((-o).tolist())

    fluidStart = len(pos)
    for p in fluidPositions:
        pos.append(np.asarray(p).tolist()); vel.append([0.0, 0.0]); kinds.append(0)
        goff.append([0.0, 0.0])

    nTot = len(pos)
    ghostIndices = torch.full((nTot,), -1, dtype=torch.int64, device=dev)
    ghostIndices[ghostStart:ghostStart + nBnd] = torch.arange(nBnd, device=dev)

    class S:
        pass
    st = S()
    st.positions = torch.tensor(pos, device=dev, dtype=DT)
    st.velocities = torch.tensor(vel, device=dev, dtype=DT)
    st.kinds = torch.tensor(kinds, device=dev, dtype=torch.int32)
    st.ghostOffsets = torch.tensor(goff, device=dev, dtype=DT)
    st.ghostIndices = ghostIndices
    st.supports = torch.full((nTot,), support, device=dev, dtype=DT)
    st.masses = torch.full((nTot,), dx * dx, device=dev, dtype=DT)
    st.densities = torch.full((nTot,), RHO0, device=dev, dtype=DT)
    yCoord = torch.tensor(np.asarray(fluidPositions), device=dev, dtype=DT) @ \
        torch.tensor(NHAT, device=dev, dtype=DT)
    fluidRho = args.rhoFluid + args.rhoGrad * (yCoord / dx)
    if fieldNoise is not None:
        fluidRho = fluidRho + torch.tensor(fieldNoise, device=dev, dtype=DT)
    st.densities[fluidStart:] = fluidRho
    st.pressures = args.c0 ** 2 * (st.densities - RHO0)
    return st, fluidStart


def cfgScheme():
    cfg = _Cfg()
    cfg.kernel = KernelFunctions.Wendland2
    cfg.dim = 2
    cfg.verletScale = 1.0
    scheme = _Cfg()
    scheme.fluid = _Cfg()
    scheme.fluid.restDensity = RHO0
    scheme.fluid.fixedSoundSpeed = args.c0
    scheme.fluid.kappa = 1.3
    scheme.fluid.polytropicExponent = 7.0
    return cfg, scheme


# --------------------------------------------------------------- Band's fit
def wendland2(r, h):
    """warpSPHCore's own Wendland C2, 2D: W(r,h) = 7/(pi h^2) (1-q)^4(1+4q),
    q=r/h, exactly 0 for q>=1 -- verified against
    warpSPHCore/kernels/kernelFunctions/wendland2.py."""
    q = r / h
    w = np.where(q < 1.0, (7.0 / np.pi) / h ** 2 * (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)
    return w


def bandMlsFit(xb, fluidPos, fluidVal, h, volume, lam=0.0, mode='tikhonov'):
    """Band et al. 2018 Eqs. (9)-(21): a hyperplane fit AT the boundary
    particle `xb` (dim,), from its fluid neighbours `fluidPos` (N,dim) /
    `fluidVal` (N,), each carrying volume `volume` (scalar or (N,)).

    `mode='tikhonov'`: smooth per-eigenvalue damped inverse of the reduced
    (decoupled) gradient block -- `lam` sets the damping scale, in units of
    the block's own trace/dim (so it is comparable across geometries).
    `mode='pinv'`: plain `torch.linalg.pinv` on that block -- closer to the
    paper's literal "safe inversion via SVD" (a hard per-eigenvalue cutoff at
    machine-precision-scale rtol, but still per-mode, not per-determinant).

    Returns (p_b, alpha_b, neighbourCount) -- `alpha_b` is the Shepard value
    alone, always defined once neighbourCount > 0, algebraically IDENTICAL
    regardless of how the gradient block conditions -- there is no separate
    fallback formula to snap to.
    """
    r = np.linalg.norm(fluidPos - xb[None, :], axis=1)
    w = volume * wendland2(r, h)
    active = w > 0
    n = int(active.sum())
    if n == 0:
        return 0.0, 0.0, 0
    w = w[active]; fp = fluidPos[active]; fv = fluidVal[active]
    M = w.sum()
    d_b = (w[:, None] * fp).sum(axis=0) / M
    alpha_b = float((w * fv).sum() / M)
    if n < 3:      # a single point or a 2-point line can't determine a 2D gradient at all
        return alpha_b, alpha_b, n

    xt = fp - d_b[None, :]                                   # (n, 2), Sum w*xt == 0 exactly
    G = (w[:, None, None] * (xt[:, :, None] * xt[:, None, :])).sum(axis=0)   # (2,2), PSD
    rhs = (w[:, None] * xt * fv[:, None]).sum(axis=0)         # (2,)

    G_t = torch.tensor(G, dtype=DT); rhs_t = torch.tensor(rhs, dtype=DT)
    if mode == 'pinv':
        c = (torch.linalg.pinv(G_t) @ rhs_t).numpy()
    else:
        evals, evecs = torch.linalg.eigh(G_t)
        scale = float(torch.clamp(evals.abs().mean(), min=1e-300))
        damped = evals / (evals ** 2 + (lam * scale) ** 2)
        Ginv = (evecs * damped.unsqueeze(0)) @ evecs.T
        c = (Ginv @ rhs_t).numpy()

    xt_b = xb - d_b
    p_b = alpha_b + float(c @ xt_b)
    return p_b, alpha_b, n


def evaluateAll(base_np, s, direction, shape, fieldNoise=None):
    fluidPos = base_np + s * dx * direction
    st, fluidStart = buildState(fluidPos, fieldNoise=fieldNoise)
    cfg, scheme = cfgScheme()
    lo = st.positions.min(dim=0).values - 8 * dx
    hi = st.positions.max(dim=0).values + 8 * dx
    cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
    cfg.dx = dx
    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)

    fluidMask = st.kinds == 0
    rho_ramped_full = computeMdbcDensity(st, cfg, scheme, adj)
    p_mls_full = wallPressureExtrapolation(st, cfg, adj, st.pressures, fluidMask,
                                           mode='mls', clampNonNeg=False)

    # the boundary row nearest the cluster -- its own position IS the query
    # point for Band's fit (no ghost node involved)
    bndMask = (st.kinds == 1).cpu().numpy()
    bndPos = st.positions[st.kinds == 1].cpu().numpy()
    fp = fluidPos
    dists = np.linalg.norm(bndPos[:, None, :] - fp[None, :, :], axis=2)
    bi_local = int(np.argmin(dists.min(axis=1)))
    xb = bndPos[bi_local]
    bIdxGlobal = np.where(bndMask)[0][bi_local]

    volume = dx * dx
    rhoVals = st.densities[fluidMask].cpu().numpy()
    pVals = st.pressures[fluidMask].cpu().numpy()

    rho_band_tik, alpha_rho, nNb = bandMlsFit(xb, fp, rhoVals, support, volume,
                                              lam=args.tikhonovLambda, mode='tikhonov')
    rho_band_pinv, _, _ = bandMlsFit(xb, fp, rhoVals, support, volume, mode='pinv')
    p_band_tik, alpha_p, _ = bandMlsFit(xb, fp, pVals, support, volume,
                                        lam=args.tikhonovLambda, mode='tikhonov')
    p_band_pinv, _, _ = bandMlsFit(xb, fp, pVals, support, volume, mode='pinv')

    return dict(
        nNb=nNb,
        rhoRamped=float(rho_ramped_full[bIdxGlobal]),
        pMls=float(p_mls_full[bIdxGlobal]),
        rhoBandTik=rho_band_tik, rhoBandPinv=rho_band_pinv, rhoShepard=alpha_rho,
        pBandTik=p_band_tik, pBandPinv=p_band_pinv, pShepard=alpha_p,
    )


def jumpiness(values):
    values = np.asarray(values, dtype=float)
    d = np.abs(np.diff(values))
    if d.size < 3 or not np.any(d > 0):
        return 0.0, 0.0
    iMax = int(np.argmax(d))
    rest = np.delete(d, iMax)
    med = np.median(rest[rest > 0]) if np.any(rest > 0) else 1e-300
    return float(d[iMax]), float(d[iMax] / max(med, 1e-300))


os.makedirs(args.out, exist_ok=True)
D0 = 1.0
sVals = np.linspace(0.0, args.jitterAmp, args.nDist)

print(f'dx={dx}  support={support/dx:.2f} dx  rhoFluid={args.rhoFluid}  rhoGrad={args.rhoGrad}/dx  '
      f'tikhonovLambda={args.tikhonovLambda}')
print()
hdr = (f'{"shape":10} {"ratio(rhoRamped)":>17} {"ratio(rhoBandTik)":>18} {"ratio(rhoBandPinv)":>19} '
      f'{"ratio(pMls)":>12} {"ratio(pBandTik)":>16} {"ratio(pBandPinv)":>17}')
print(hdr)

rows = []
for shape in [f'line-{n}' for n in args.lineCounts]:
    base = baseCluster(shape, D0)
    rng = np.random.default_rng(zlib.crc32(shape.encode()))
    direction = rng.normal(size=base.shape)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.where(norms > 0, norms, 1.0)
    # fixed per-particle noise (a stand-in for real SPH density scatter around
    # the analytic profile from particle disorder/discretization) -- generated
    # ONCE per shape so it travels with each particle across the whole sweep,
    # exactly like `direction` does, rather than being redrawn every sample
    noiseRng = np.random.default_rng(zlib.crc32((shape + '-noise').encode()))
    fieldNoise = noiseRng.normal(scale=args.fieldNoise, size=base.shape[0]) \
        if args.fieldNoise > 0 else None

    series = {k: [] for k in ['rhoRamped', 'pMls', 'rhoBandTik', 'rhoBandPinv',
                              'pBandTik', 'pBandPinv', 'nNb']}
    for s in sVals:
        r = evaluateAll(base, s, direction, shape, fieldNoise=fieldNoise)
        for k in series:
            series[k].append(r[k])

    jr = {k: jumpiness(series[k]) for k in series if k != 'nNb'}
    print(f'{shape:10} {jr["rhoRamped"][1]:17.1f} {jr["rhoBandTik"][1]:18.1f} '
          f'{jr["rhoBandPinv"][1]:19.1f} {jr["pMls"][1]:12.1f} {jr["pBandTik"][1]:16.1f} '
          f'{jr["pBandPinv"][1]:17.1f}')
    rows.append(dict(shape=shape, sVals=sVals.tolist(), **series))

np.save(os.path.join(args.out, 'sweep.npy'), rows, allow_pickle=True)
print()
print('ratio(*) = jumpiness (largest consecutive-sample jump / typical local step).')
print('rhoRamped/pMls = CURRENT production (already-ramped density; still-hard pressure).')
print('*BandTik = centroid-transform + Tikhonov-damped eigen-inverse (smooth).')
print('*BandPinv = centroid-transform + torch.linalg.pinv (paper\'s literal "safe SVD").')
print()

# ------------------------------------------------------- detailed trace at
# the exact discrete-neighbour-entry crossing that broke the existing ramp
print('=' * 88)
print('Detailed trace across the line-10 nNb 8->9 crossing (broke the existing ramp -- see')
print('probe_boundaryExtrapolationSwitchDiscontinuity.py\'s writeup)')
print('=' * 88)
rhoTrue = args.rhoFluid + args.rhoGrad * (-0.5)   # analytic (noise-free) value at the
                                                   # k=0 boundary row, xb_y = -0.5 dx
print(f'analytic (noise-free) true wall density at xb: {rhoTrue:.6f}')
print(f'(errors below are value - {rhoTrue:.6f}; the field carries +-{args.fieldNoise:.4f} '
      f'IID per-particle noise, so some residual error away from any crossing is expected -- '
      f'the question is whether a crossing adds a LARGE additional error on top of that floor)')
print()
for row in rows:
    if row['shape'] != 'line-10':
        continue
    nNb = np.array(row['nNb'])
    flip = np.where(np.diff(nNb) != 0)[0]
    if flip.size == 0:
        print('no neighbour-count change found in this sweep range')
        break
    i = flip[0]
    lo, hi = max(0, i - 4), min(len(row['sVals']), i + 5)
    print(f'{"s":>8} {"nNb":>4} {"err(ramped)":>12} {"err(bandTik)":>13} {"err(bandPinv)":>14}')
    for k in range(lo, hi):
        eR = row['rhoRamped'][k] - rhoTrue
        eT = row['rhoBandTik'][k] - rhoTrue
        eP = row['rhoBandPinv'][k] - rhoTrue
        marker = '  <-- crossing' if k in (i, i + 1) else ''
        print(f'{row["sVals"][k]:8.4f} {nNb[k]:4d} {eR:12.5f} {eT:13.5f} {eP:14.5f}{marker}')
