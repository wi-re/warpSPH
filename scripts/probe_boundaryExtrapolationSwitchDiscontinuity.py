#!/usr/bin/env python
"""Does the MLS-vs-Shepard fallback SWITCH in the wall/ghost extrapolation
create real, physically-sized discontinuities for few-particle boundary
stencils, and is that a plausible driver of the still-open "few particle
large sheet on a boundary blows up" item (DELTASPH_VALIDATION_PLAN.md)?

Motivating hypothesis (user-supplied): every boundary-extrapolation routine
in this codebase reduces to the same two-tier idea -- a 1st-order MLS fit
(`interpolateLiuLiu`'s `rho_proj`/`p_proj`, "extends" the field along its
local gradient) with a 0th-order Shepard fallback ("mirrors": zero gradient,
same idea as `liu/interp.py`'s `liuMirror` vs `liuExtend` naming) for stencils
too thin to trust the gradient. Three concrete implementations of that idea
exist in this repo, at three different levels of "hard":

  1. `liu/interp.py`'s `liuExtend`/`liuMirror` -- an UNUSED (dead-code) pair
     with the crudest switch: `neighCounts < neighborThreshold` (bare count,
     NO conditioning/determinant gate at all).
  2. The historical `computeMdbcDensity` (mDBC wall density) -- USED to have
     a hard `where(wellConditioned, rho_proj, shepard)` switch
     (DELTASPH_VALIDATION_PLAN.md 5.2.3 / the `afb6e59`/`80eabb9` history,
     lines ~1990-2010): "adjacent boundary particles snapped between the
     exact MLS value and ~rho0, a visible step right where the surface meets
     the wall." FIXED: now a smooth `w = wDet * wN` ramp on
     `numNeighbors`/`|det(A_g)|`. This is the delta-SPH path (`deltaSPH.py`
     calls `computeMdbcDensity`, pressure follows via EOS) -- already
     protected.
  3. `modules/incompressible/wallPressure.py`'s `wallPressureExtrapolation`,
     `mode='mls'` -- STILL a hard `where(wellConditioned, p_proj, p_b)`
     switch (lines ~260-262), same shape as #2's pre-fix bug. Live in
     `omniIncompressible`/`dfsphReference`/`incompressible.py` (not the
     delta-SPH branch, but the exact same codebase and the exact same
     mechanism previously root-caused for density).

This probe builds synthetic few-particle constellations near a flat wall --
single / pair / triangle / 2x2 square / an N-particle line, each with and
without positional jitter -- and sweeps a CONTINUOUS physical control (the
cluster's normal distance from the wall, i.e. "how thin is the sheet") to
see whether the discrete `wellConditioned` (or bare neighbour-count) flip
that occurs somewhere along that continuous sweep produces an O(1) jump in
the extrapolated field relative to the smooth trend on either side -- for
(a) the OLD reconstructed hard density switch, (b) the CURRENT smooth-ramp
`computeMdbcDensity` (should show no such jump -- the fixed control), and
(c) the CURRENT `wallPressureExtrapolation(mode='mls')` (should show a real
jump if the hypothesis holds for that still-live call site).

A second experiment holds geometry fixed at a marginal, near-threshold
configuration and Monte-Carlo jitters it at a small fixed amplitude, to
see whether an infinitesimal perturbation of otherwise-identical stencils
produces a bimodal (jumped) vs. unimodal (smooth) output distribution.

    python scripts/probe_boundaryExtrapolationSwitchDiscontinuity.py [--dx 0.01]
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--c0', type=float, default=40.0)
ap.add_argument('--supportRatio', type=float, default=4.0,
                help='support/dx; 4.0 matches cases/dambreak.py n_h (production)')
ap.add_argument('--layers', type=int, default=5)
ap.add_argument('--rhoFluid', type=float, default=1.03,
                help='density at the cluster centroid (rho0=1)')
ap.add_argument('--rhoGrad', type=float, default=0.02,
                help='density gradient along the wall normal, per dx -- gives '
                     'the MLS fit something real to extrapolate (0 = the '
                     'degenerate constant-field case where MLS collapses '
                     'onto Shepard by construction)')
ap.add_argument('--lineCounts', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 18],
                help='particle counts for the "line" shape sweep')
ap.add_argument('--nDist', type=int, default=240, help='samples in the distance sweep')
ap.add_argument('--jitterAmp', type=float, default=1.5,
                help='max scatter amplitude swept in experiment 1, in dx '
                     '(0 = perfectly flat sheet -> this value of scatter)')
ap.add_argument('--jitterTrials', type=int, default=400)
ap.add_argument('--out', default=os.path.join('scratchpad', 'boundary_switch_discontinuity'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')

import numpy as np
import torch

from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList, OperationDirection)
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.modules.liu import interpolateLiuLiu, determinantThresholdFor
from warpSPH.modules.incompressible.wallPressure import wallPressureExtrapolation

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
dx = args.dx
RHO0 = 1.0
support = args.supportRatio * dx

NHAT = np.array([0.0, 1.0])   # wall normal, points from solid INTO the fluid (fluid at y>0)
THAT = np.array([1.0, 0.0])


class _Cfg:
    pass


def shapeOffsets(name):
    """(tangential, normal) offsets in dx, normal >= 0, defining a cluster's
    shape at unit spacing; the whole cluster is later placed at a swept
    normal distance from the wall."""
    if name == 'single':
        return [(0, 0)]
    if name == 'pair':
        return [(0, 0), (1, 0)]
    if name == 'triangle':
        return [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]
    if name == 'square':
        return [(0, 0), (1, 0), (0, 1), (1, 1)]
    if name.startswith('line'):
        n = int(name.split('-')[1])
        c = (n - 1) / 2.0
        return [(i - c, 0.0) for i in range(n)]
    raise ValueError(name)


def baseCluster(shape, distDx):
    """(N,2) absolute positions for `shape` at nominal normal distance
    `distDx` (in dx) from the wall plane, tangentially centred at 0."""
    offs = shapeOffsets(shape)
    return np.array([(distDx + n) * dx * NHAT + t * dx * THAT for (t, n) in offs])


def buildState(fluidPositions):
    """Wall band (kind 1) + mirrored ghosts (kind 2) + a fluid cluster
    (kind 0) at the given absolute positions. Mirrors
    `probe_mdbcSparseFluid.py`'s construction (with its ghost-mirror sign
    corrected -- see the comment below)."""
    pos, vel, kinds, goff = [], [], [], []
    nSpan = int(np.ceil((6 * support) / dx)) + 4
    for k in range(args.layers):
        base = -(k + 0.5) * dx * NHAT
        for s in range(-nSpan, nSpan + 1):
            p = base + s * dx * THAT
            pos.append(p.tolist()); vel.append([0.0, 0.0]); kinds.append(1)
            # r_b - r_g: boundary at -(k+.5)dx*NHAT, ghost MIRRORED into the
            # fluid at +(k+.5)dx*NHAT (probe_mdbcSparseFluid.py has this sign
            # flipped -- its ghosts land at -(3k+1.5)dx*NHAT, deeper in the
            # solid, so they never gather any fluid neighbour; see writeup).
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
    # A REAL spatial gradient in the fluid cluster, not a constant -- with a
    # constant field the 1st-order MLS fit's gradient term is exactly zero,
    # so it collapses onto the 0th-order Shepard value regardless of
    # conditioning and the hard-vs-ramped switch becomes invisible by
    # construction (confirmed: an earlier constant-density version of this
    # probe showed wellConditioned flipping 30-85% of MC trials while every
    # output stayed pinned to the same float, because there was nothing for
    # the two tiers to disagree about). rho = rhoFluid + rhoGrad * (y/dx),
    # y measured along the wall normal -- a stand-in for the near-wall
    # hydrostatic/impact gradient the MLS extrapolation exists to capture.
    yCoord = torch.tensor(np.asarray(fluidPositions), device=dev, dtype=DT) @ \
        torch.tensor(NHAT, device=dev, dtype=DT)
    st.densities[fluidStart:] = args.rhoFluid + args.rhoGrad * (yCoord / dx)
    st.pressures = args.c0 ** 2 * (st.densities - RHO0)
    return st


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


def evaluate(st):
    """One point of the sweep: returns numNeighbors/det/wellConditioned at
    the ghost stencil that sees the cluster, plus the three densities/pressure
    of interest, all read at the wall-particle row nearest the cluster (the
    one with the largest ghost neighbour count -- the row actually coupled to
    the few-particle cluster)."""
    cfg, scheme = cfgScheme()
    lo = st.positions.min(dim=0).values - 8 * dx
    hi = st.positions.max(dim=0).values + 8 * dx
    cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
    cfg.dx = dx

    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)

    # -- raw MLS/Shepard tiers at the ghost nodes (same call computeMdbcDensity
    # and wallPressureExtrapolation('mls') make internally)
    rho_i, grad_i, nNb, A_g, b, wc = interpolateLiuLiu(
        st.positions[st.kinds == 2], referenceParticles=st,
        referenceQuantities=st.densities, config=cfg,
        neighbor_threshold=4, direction=OperationDirection.FluidToGhost,
        supportScale=1.0, adjacency=adj.hashMap if hasattr(adj, 'hashMap') else None)
    det = torch.linalg.det(A_g).abs()

    ghostMask = st.kinds == 2
    bIdx = st.ghostIndices[ghostMask]
    relPos = -st.ghostOffsets[ghostMask]
    drho = -torch.einsum('nu,nu->n', relPos, grad_i)
    rho_proj_g = rho_i + drho
    shepDen = A_g[:, 0, 0]
    shepard_g = torch.where(shepDen > 0, b[:, 0] / shepDen.clamp_min(1e-12),
                            torch.full_like(shepDen, RHO0))
    shepard_g = torch.clamp(shepard_g, min=RHO0)

    if nNb.numel() == 0 or int(nNb.max()) == 0:
        j = None
    else:
        j = int(torch.argmax(nNb))

    # -- (a) reconstructed OLD hard switch (pre afb6e59/80eabb9), mapped onto
    # the wall-particle row it feeds
    rho_hard_g = torch.where(wc, rho_proj_g, shepard_g)
    rho_hard_full = st.densities.clone()
    rho_hard_full[bIdx] = rho_hard_g

    # -- (b) CURRENT production smooth ramp (deltaSPH's actual wall density)
    rho_ramped_full = computeMdbcDensity(st, cfg, scheme, adj)

    # -- (c) CURRENT production wallPressureExtrapolation('mls') -- still a
    # hard switch; used by omniIncompressible/dfsphReference/incompressible.py
    fluidMask = st.kinds == 0
    p_mls = wallPressureExtrapolation(st, cfg, adj, st.pressures, fluidMask,
                                      mode='mls', clampNonNeg=False)

    if j is None:
        return dict(nNb=0, det=0.0, wc=False, bIdx=None,
                    rhoHard=RHO0, rhoRamped=RHO0, pMls=0.0)
    bi = int(bIdx[j])
    return dict(
        nNb=int(nNb[j]), det=float(det[j]), wc=bool(wc[j]),
        bIdx=bi,
        rhoHard=float(rho_hard_full[bi]),
        rhoRamped=float(rho_ramped_full[bi]),
        pMls=float(p_mls[bi]),
    )


def jumpiness(distances, values):
    """Max |consecutive difference| divided by the median |consecutive
    difference| away from that max -- large ratio = one dominant jump amid an
    otherwise-smooth trend; ~1 = uniformly smooth/noisy, no standout jump."""
    values = np.asarray(values, dtype=float)
    d = np.abs(np.diff(values))
    if d.size < 3 or not np.any(d > 0):
        return 0.0, 0.0
    iMax = int(np.argmax(d))
    rest = np.delete(d, iMax)
    med = np.median(rest[rest > 0]) if np.any(rest > 0) else 1e-300
    return float(d[iMax]), float(d[iMax] / max(med, 1e-300))


# --------------------------------------------------------------- experiment 1
# A perfectly flat cluster/line is ALWAYS coplanar in the normal direction --
# det(A_g) pinned near 0 regardless of distance from the wall, so a distance
# sweep alone never crosses the wellConditioned gate for these shapes (checked
# directly: it doesn't). The physically meaningful continuous control is
# instead how far off perfectly-flat the sheet is -- so fix each cluster at a
# nominal distance d0 and walk it along ONE fixed random per-particle
# direction (drawn once, seeded) by a scalar amplitude `s`, from a perfectly
# flat sheet (s=0) up to `args.jitterAmp` dx of scatter. This is a smooth,
# continuous, reproducible path through configuration space -- any output
# discontinuity along it is attributable purely to the switch logic, not to a
# different physical scenario.
os.makedirs(args.out, exist_ok=True)
shapes = ['single', 'pair', 'triangle', 'square'] + [f'line-{n}' for n in args.lineCounts]
D0 = 1.0   # nominal distance, in dx, well inside the support

print(f'dx={dx}  support={support/dx:.2f} dx  rho_fluid={args.rhoFluid}  d0={D0} dx  '
      f'detFloor(Wendland2)={determinantThresholdFor(KernelFunctions.Wendland2):.2e}')
print('(single/pair/triangle/square have <=4 particles, so numNeighbors>4 can')
print(' never hold for them -- they are expected to sit permanently on the')
print(' fallback branch; included as the negative control.)')
print()
print(f'{"shape":10} {"maxJump(hard)":>14} {"ratio(hard)":>12} '
      f'{"maxJump(ramped)":>16} {"ratio(ramped)":>14} {"maxJump(pMls)":>14} {"ratio(pMls)":>12} '
      f'{"nNb@flip":>9} {"det@flip":>10} {"s@flip":>8}')

summary = []
sVals = np.linspace(0.0, args.jitterAmp, args.nDist)
for shape in shapes:
    base = baseCluster(shape, D0)
    rng = np.random.default_rng(abs(hash(shape)) % (2 ** 32))
    direction = rng.normal(size=base.shape)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.where(norms > 0, norms, 1.0)

    rhoHard, rhoRamped, pMls, nNbs, dets, wcs = [], [], [], [], [], []
    for s in sVals:
        st = buildState(base + s * dx * direction)
        r = evaluate(st)
        rhoHard.append(r['rhoHard']); rhoRamped.append(r['rhoRamped']); pMls.append(r['pMls'])
        nNbs.append(r['nNb']); dets.append(r['det']); wcs.append(r['wc'])

    jH, rH = jumpiness(sVals, rhoHard)
    jR, rR = jumpiness(sVals, rhoRamped)
    jP, rP = jumpiness(sVals, pMls)
    wcs = np.array(wcs)
    flipIdx = np.where(np.diff(wcs.astype(int)) != 0)[0]
    flipAt = f'{nNbs[flipIdx[0]]}->{nNbs[flipIdx[0]+1]}' if flipIdx.size else 'none'
    detAtFlip = f'{dets[flipIdx[0]]:.2e}' if flipIdx.size else 'n/a'
    sAtFlip = f'{sVals[flipIdx[0]]:.3f}' if flipIdx.size else 'n/a'
    print(f'{shape:10} {jH:14.4e} {rH:12.1f} {jR:16.4e} {rR:14.1f} {jP:14.4e} {rP:12.1f} '
          f'{flipAt:>9} {detAtFlip:>10} {sAtFlip:>8}')
    summary.append(dict(shape=shape, sVals=sVals.tolist(), rhoHard=rhoHard,
                        rhoRamped=rhoRamped, pMls=pMls, nNb=nNbs, det=dets, wc=wcs.tolist()))

np.save(os.path.join(args.out, 'sweep.npy'), summary, allow_pickle=True)
print()
print('Columns: maxJump = largest |consecutive-sample difference| along the')
print('sheet-scatter sweep; ratio = that jump / the median step size elsewhere')
print('in the same sweep (>>1 means one dominant discontinuity, ~1 means smooth).')
print('rhoHard/pMls are hard-switch outputs (expect large ratios); rhoRamped is')
print('the CURRENT production computeMdbcDensity (expect ratio ~ O(1), the fix).')
print()

# --------------------------------------------------------------- experiment 2
# Fixed near-threshold geometry (the marginal scatter amplitude s* found
# above), true Monte-Carlo jitter with INDEPENDENT random directions at small
# fixed amplitude around it: bimodal (jumped) vs. unimodal (smooth) output
# under an infinitesimal perturbation of an otherwise-identical stencil.
print('=' * 78)
print('Same total scatter magnitude s* as the flip found in experiment 1, but a')
print('FRESH independent random direction per trial -- does "how thick is the')
print('sheet" alone decide the branch, or does the direction/arrangement matter?')
print('=' * 78)

for entry in summary:
    wc = np.array(entry['wc'])
    flipIdx = np.where(np.diff(wc.astype(int)) != 0)[0]
    if flipIdx.size == 0:
        continue
    sMarg = 0.5 * (entry['sVals'][flipIdx[0]] + entry['sVals'][flipIdx[0] + 1])
    shape = entry['shape']
    base = baseCluster(shape, D0)
    rng = np.random.default_rng(1)
    hardVals, rampedVals, pMlsVals, wcFlags = [], [], [], []
    for _ in range(args.jitterTrials):
        disp = rng.normal(size=base.shape)
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp / np.where(norms > 0, norms, 1.0)
        st = buildState(base + sMarg * dx * disp)
        r = evaluate(st)
        hardVals.append(r['rhoHard']); rampedVals.append(r['rhoRamped'])
        pMlsVals.append(r['pMls']); wcFlags.append(r['wc'])
    hardVals = np.array(hardVals); rampedVals = np.array(rampedVals)
    pMlsVals = np.array(pMlsVals); wcFlags = np.array(wcFlags)
    print(f'\n--- {shape}  (marginal scatter s*={sMarg:.3f} dx) ---')
    print(f'  wellConditioned flips in {wcFlags.mean()*100:.1f}% of {args.jitterTrials} trials')
    print(f'  rhoHard:   std={hardVals.std():.4e}  range=[{hardVals.min():.4f}, {hardVals.max():.4f}]')
    print(f'  rhoRamped: std={rampedVals.std():.4e}  range=[{rampedVals.min():.4f}, {rampedVals.max():.4f}]')
    print(f'  pMls:      std={pMlsVals.std():.4e}  range=[{pMlsVals.min():.4f}, {pMlsVals.max():.4f}]')
    # bimodality proxy: gap between the two branch means vs. within-branch std
    if wcFlags.any() and not wcFlags.all():
        gapHard = abs(hardVals[wcFlags].mean() - hardVals[~wcFlags].mean())
        withinHard = 0.5 * (hardVals[wcFlags].std() + hardVals[~wcFlags].std()) + 1e-300
        gapRamped = abs(rampedVals[wcFlags].mean() - rampedVals[~wcFlags].mean())
        withinRamped = 0.5 * (rampedVals[wcFlags].std() + rampedVals[~wcFlags].std()) + 1e-300
        print(f'  bimodality (branch-gap / within-branch std): '
              f'hard={gapHard/withinHard:.2f}   ramped={gapRamped/withinRamped:.2f}')

print()
print(f'Saved raw sweep data to {os.path.join(args.out, "sweep.npy")}')
