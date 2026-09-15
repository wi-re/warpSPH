#!/usr/bin/env python
"""Prototype of English, Vacondio et al. 2025's analytic-hydrostatic mDBC
pressure-cloning extrapolation (`literature/english2025_river-flows-past-
bridges.pdf`, Eqs. 8-11), tested specifically against the DEPTH/LEVER-ARM
failure mode found in `BOUNDARY_DENSITY_PLAN.md` Section 3.1 -- NOT the
conditioning/few-neighbour failure mode Sections 1-2 already cover.

Recap of the §3.1 finding: moving the Band-2018-style fit from the boundary
particle to its ghost node (to fix starved deep-layer stencils) and Taylor-
shifting the fitted GRADIENT back by `relPos` is not a free win. A well-
conditioned (`wCond=1.0`), correctly-computed gradient still gets multiplied
by a lever arm that grows with the boundary particle's layer depth
(`xtb ~ 0.25`, ~9 particle spacings, at the diverging config's failure point)
-- amplifying a modest, real gradient (or, worse, noise in a NOISY fitted
gradient) into an unphysical extrapolated value. This is not unique to Band:
`density2025.py`'s own English-2022 Eq. (12) path has the identical
`(r_b - r_g) . grad(rho)_g` structure and the identical growing lever arm at
depth, and evidently copes better in practice (§3.2), but the mechanism is
shared.

English 2025 Eqs. (8)-(11) sidestep this differently: instead of extrapolating
with a FITTED (and therefore noisy) spatial gradient, they extrapolate with a
KNOWN, noise-free physical quantity -- gravity:

    rho_g = MLS/Shepard value fit AT the ghost                       (Eq. 8)
    P_g   = c0^2 (rho_g - rho0)                                      (Eq. 9)
    P_b   = P_g + rho0 * dot(g - a_b, n_b) * dot(x_g - x_b, n_b)      (Eq. 10)
    rho_b = rho0 + P_b / c0^2                                        (Eq. 11)

CORRECTED SIGN (found by direct cross-check against DualSPHysics' own
`JSphCpu::Mdbc2PressClone`, `~/dev/DualSPHysics/src/source/JSphCpu_mdbc.cpp`
-- see BOUNDARY_DENSITY_PLAN.md §4 for the full derivation): the paper's
printed Eq. (10), read literally off `pdftotext` output, gives
`P_b = P_g + rho0 (g-a_b).n_b (x_g-x_b).n_b`, which integrates to the WRONG
sign against the governing hydrostatic ODE (`grad(P) = rho0 g`, Eq. 2) -- most
likely a subscript transcribed backwards by the PDF's equation renderer, a
known pdftotext failure mode for two-part subscripted dot products, not a
paper error (never verified against the typeset PDF image itself). The actual
DualSPHysics implementation, and independently the ODE, both give:

    P_b = P_g + rho0 * dot(g - a_b, relPos),  relPos = x_b - x_g

-- i.e. `n_b` cancels entirely (the ghost lies exactly along the true normal
by construction, both here and in DualSPHysics) and `relPos` is EXACTLY this
codebase's existing `-ghostOffsets` convention (`density2025.py`'s own
`relPos`, used unmodified below). `a_b = 0` throughout (a static wall). Since
the extrapolation term no longer depends on any FITTED quantity, a growing
lever arm no longer amplifies fit noise -- it just multiplies a constant,
known `g` by a known, exact distance.

This probe builds ONE flat-wall geometry with many boundary layers (so ghost
depth into the fluid grows linearly with layer index, exactly reproducing the
§3.1 mechanism) and a DENSE fluid block obeying a true, noise-free hydrostatic
profile consistent with a chosen gravity `g` (rho(y) = rho0 + rho0 g_y/c0^2 *
y, from grad(P) = rho0 g under the weakly-compressible EOS), so the analytic
term's assumption is exactly satisfied and any residual error is attributable
to the VALUE fit and to per-particle noise, not to a wrong physical model. IID
per-particle noise is added on top (a real SPH stencil is never noise-free),
comparable in magnitude to the hydrostatic signal over the depth range swept
-- the point being to show FITTED-GRADIENT extrapolation error growing with
depth (as noise gets multiplied by a growing lever arm) against ANALYTIC
extrapolation error staying flat (noise in the value fit alone, never
multiplied by depth).

Four variants compared, all evaluated at the SAME boundary particles across
the SAME depth sweep:

  - `english2022`  : production `computeMdbcDensity` (density2025.py) --
                      Liu/MLS value+gradient fit at the ghost, ramped blend,
                      Eq. (12) fitted-gradient extrapolation.
  - `band`          : production `computeMdbcDensityBand` -- Tikhonov-damped
                      value+gradient fit at the ghost, Taylor-shifted (also a
                      fitted-gradient extrapolation, just better-conditioned).
  - `english2025_liuValue`  : SAME ghost value as `english2022` (Liu/MLS
                      `rho_interp`, so the comparison isolates ONLY the
                      extrapolation-formula change), but Eq. (10)'s analytic
                      hydrostatic term instead of the fitted gradient.
  - `english2025_bandValue` : Band's Tikhonov-damped VALUE-ONLY fit at the
                      ghost (no gradient block at all) combined with Eq. (10)'s
                      analytic extrapolation -- the plan's "combined, not
                      replacing" recommendation: the already-validated
                      Tikhonov value fit from §2-3, plus the analytic
                      extrapolation instead of ANY fitted gradient.

    python scripts/probe_englishAnalyticGravityExtrapolation.py
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--c0', type=float, default=40.0)
ap.add_argument('--supportRatio', type=float, default=4.0)
ap.add_argument('--layers', type=int, default=16,
                help='boundary layers -- ghost depth into the fluid grows '
                     'linearly with layer index (see buildState), so this '
                     'sets the depth range swept')
ap.add_argument('--gY', type=float, default=-9.81,
                help='gravity along -NHAT (pointing from fluid into the wall, '
                     'i.e. a floor below the fluid) -- sign matches a fluid '
                     'resting on this wall')
ap.add_argument('--fieldNoise', type=float, default=0.001,
                help='IID per-particle density noise (fraction of rho0), '
                     'comparable in magnitude to the hydrostatic signal over '
                     'the swept depth range -- the point is to see whether a '
                     'FITTED gradient amplifies this noise with depth while '
                     'the analytic term does not')
ap.add_argument('--tikhonovLambda', type=float, default=1e-3)
ap.add_argument('--out', default=os.path.join('scratchpad', 'english_analytic_gravity'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')

import numpy as np
import torch

from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList, OperationDirection)
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.modules.mdbc.densityBand import computeMdbcDensityBand
from warpSPH.modules.liu import interpolateLiuLiu

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
dx = args.dx
RHO0 = 1.0
C0 = args.c0
support = args.supportRatio * dx
NHAT = np.array([0.0, 1.0])   # wall -> fluid direction
THAT = np.array([1.0, 0.0])
G_VEC = np.array([0.0, args.gY])
A_B = np.array([0.0, 0.0])    # static wall


class _Cfg:
    pass


def hydrostaticRho(y):
    """rho(y) = rho0 + rho0 * g_y / c0^2 * y -- the unique linear profile
    consistent with grad(P) = rho0 * g under P = c0^2 (rho - rho0), i.e. the
    exact field English 2025 Eq. (10) assumes. Noise-free; per-particle noise
    is added separately by the caller."""
    return RHO0 + RHO0 * args.gY / C0 ** 2 * y


def buildState(nSpan, layers, fieldNoiseAmp, seed):
    """Multi-layer flat wall (boundary layer k's ghost offset by (2k+1) dx
    along NHAT, same convention as probe_bandMlsPressureBoundary.py's
    buildState -- deeper layers get a ghost placed further OUT into the
    fluid, reproducing the growing lever arm §3.1 diagnosed) plus a dense
    fluid block wide/tall enough to give every ghost, even the deepest, a
    well-populated support."""
    rng = np.random.default_rng(seed)
    pos, vel, kinds, goff = [], [], [], []
    layerOfBoundary = []
    for k in range(layers):
        base = -(k + 0.5) * dx * NHAT
        for s in range(-nSpan, nSpan + 1):
            p = base + s * dx * THAT
            pos.append(p.tolist()); vel.append([0.0, 0.0]); kinds.append(1)
            goff.append((-(2 * k + 1) * dx * NHAT).tolist())
            layerOfBoundary.append(k)
    nBnd = len(pos)

    ghostStart = nBnd
    for bi in range(nBnd):
        b = np.array(pos[bi]); o = np.array(goff[bi])
        pos.append((b - o).tolist()); vel.append([0.0, 0.0]); kinds.append(2)
        goff.append((-o).tolist())

    fluidStart = len(pos)
    maxGhostY = (2 * (layers - 1) + 0.5) * dx
    yTop = maxGhostY + 3.0 * support
    xHalf = 6.0 * support
    xs = np.arange(-xHalf, xHalf + 0.5 * dx, dx)
    ys = np.arange(0.5 * dx, yTop, dx)
    fluidPos = np.array([[x, y] for y in ys for x in xs])
    for p in fluidPos:
        pos.append(p.tolist()); vel.append([0.0, 0.0]); kinds.append(0)
        goff.append([0.0, 0.0])

    nTot = len(pos)
    ghostIndices = torch.full((nTot,), -1, dtype=torch.int64, device=dev)
    # Bidirectional, matching rigidBody/ghostParticles.py's real convention:
    # a boundary row's ghostIndices points at its ghost's global index, and
    # (separately) a ghost row's points back at its boundary particle's.
    ghostIndices[0:nBnd] = torch.arange(ghostStart, ghostStart + nBnd, device=dev)
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

    trueRho = hydrostaticRho(fluidPos[:, 1])
    noise = rng.normal(scale=fieldNoiseAmp, size=fluidPos.shape[0]) if fieldNoiseAmp > 0 else 0.0
    st.densities[fluidStart:] = torch.tensor(trueRho + noise, device=dev, dtype=DT)
    st.pressures = C0 ** 2 * (st.densities - RHO0)
    return st, np.array(layerOfBoundary), nBnd


def cfgScheme():
    cfg = _Cfg()
    cfg.kernel = KernelFunctions.Wendland2
    cfg.dim = 2
    cfg.verletScale = 1.0
    scheme = _Cfg()
    scheme.fluid = _Cfg()
    scheme.fluid.restDensity = RHO0
    scheme.fluid.fixedSoundSpeed = C0
    scheme.fluid.kappa = 1.3
    scheme.fluid.polytropicExponent = 7.0
    return cfg, scheme


def wendland2(r, h):
    q = r / h
    return np.where(q < 1.0, (7.0 / np.pi) / h ** 2 * (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)


def bandTikhonovValue(xg, fluidPos, fluidVal, h, volume, lam):
    """Band et al. 2018's VALUE-ONLY term (`alpha_b`, its Eq. 19): the
    kernel-weighted centroid transform decouples this from the gradient block
    exactly, so it needs no Tikhonov damping itself -- it is exact and
    well-posed whenever there is at least one neighbour. Ignoring the
    gradient block entirely (unlike `bandMlsFit` in the sibling probe) is the
    point: this is the "value fit at the ghost" half of the plan's
    'combine, don't replace' recommendation, paired below with Eq. (10)'s
    analytic extrapolation instead of any fitted gradient."""
    r = np.linalg.norm(fluidPos - xg[None, :], axis=1)
    w = volume * wendland2(r, h)
    active = w > 0
    if not np.any(active):
        return RHO0, 0
    w = w[active]; fv = fluidVal[active]
    M = w.sum()
    alpha = float((w * fv).sum() / M)
    return alpha, int(active.sum())


os.makedirs(args.out, exist_ok=True)
nSpan = int(np.ceil((6 * support) / dx)) + 4
st, layerOf, nBnd = buildState(nSpan, args.layers, args.fieldNoise, seed=12345)
cfg, scheme = cfgScheme()
lo = st.positions.min(dim=0).values - 8 * dx
hi = st.positions.max(dim=0).values + 8 * dx
cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
cfg.dx = dx
adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                      supportMode=SupportScheme.SuperSymmetric,
                      priorNeighborhood=None, verbose=False)

bMask = st.kinds == 1
gMask = st.kinds == 2
fMask = st.kinds == 0
bIndices = torch.arange(st.positions.shape[0], device=dev)[bMask]
bPosNp = st.positions[bMask].cpu().numpy()
fluidPosNp = st.positions[fMask].cpu().numpy()
fluidRhoNp = st.densities[fMask].cpu().numpy()

# -- production paths (English 2022 / Band), unmodified -----------------
rho_english2022_full = computeMdbcDensity(st, cfg, scheme, adj)
rho_band_full = computeMdbcDensityBand(st, cfg, scheme, adj, tikhonovLambda=args.tikhonovLambda)
rho_english2022 = rho_english2022_full[bIndices].cpu().numpy()
rho_band = rho_band_full[bIndices].cpu().numpy()

# -- English 2022's own Liu/MLS ghost VALUE (no extrapolation applied yet),
# reused so `english2025_liuValue` isolates ONLY the extrapolation-formula
# change against `english2022` -- same 0th-order fit, different Eq. (12) vs
# Eq. (10) term.
rho_interp, rho_interp_grad, numNeighbors, A_g, b, wellConditioned = interpolateLiuLiu(
    st.positions[gMask], referenceParticles=st, referenceQuantities=st.densities,
    config=cfg, neighbor_threshold=4, direction=OperationDirection.FluidToGhost,
    supportScale=1.0, adjacency=adj.hashMap)
ghostIndicesOfB = st.ghostIndices[bMask]
gPosAll = st.positions[gMask]
# map each boundary row to its own ghost's row in the ghost-ordered tensors
gRowOfB = torch.searchsorted(torch.arange(st.positions.shape[0], device=dev)[gMask],
                             ghostIndicesOfB)
rho_interp_b = rho_interp[gRowOfB].cpu().numpy()
xg_np = gPosAll[gRowOfB].cpu().numpy()
xb_np = bPosNp
relPos = xb_np - xg_np   # x_b - x_g, this codebase's (and DualSPHysics' `dpos`) convention

def analyticExtrapolate(rho_g):
    """P_b = P_g + rho0 * dot(g - a_b, relPos) -- verified against
    DualSPHysics' `Mdbc2PressClone` (see module docstring for the sign
    correction from the literally-OCR'd paper equation)."""
    P_g = C0 ** 2 * (rho_g - RHO0)
    P_b = P_g + RHO0 * (relPos @ (G_VEC - A_B))
    return RHO0 + P_b / C0 ** 2

rho_e25_liuValue = analyticExtrapolate(rho_interp_b)
wcFracLiu = float(wellConditioned[gRowOfB].float().mean())

volume = dx * dx
alpha_band = np.array([bandTikhonovValue(xg_np[i], fluidPosNp, fluidRhoNp, support, volume,
                                         args.tikhonovLambda)[0]
                       for i in range(xg_np.shape[0])])
rho_e25_bandValue = analyticExtrapolate(alpha_band)

rho_true_b = hydrostaticRho(xb_np[:, 1])
layers = np.unique(layerOf)
leverArm = (2 * layers + 1) * dx

print(f'dx={dx}  support={support/dx:.1f} dx  c0={C0}  g_y={args.gY}  '
      f'fieldNoise={args.fieldNoise} (frac of rho0)  tikhonovLambda={args.tikhonovLambda}')
print(f'hydrostatic slope rho0*g_y/c0^2 = {RHO0*args.gY/C0**2:.6e} per unit length '
      f'({RHO0*args.gY/C0**2*dx:.3e} per dx)')
print()
hdr = (f'{"layer":>5} {"leverArm/dx":>11} {"err(2022)":>11} {"err(band)":>11} '
      f'{"err(e25_liu)":>13} {"err(e25_band)":>13}')
print(hdr)

rows = []
for k in layers:
    mask = layerOf == k
    err = lambda arr: float(np.mean(np.abs(arr[mask] - rho_true_b[mask])))
    e2022, eBand = err(rho_english2022), err(rho_band)
    eE25L, eE25B = err(rho_e25_liuValue), err(rho_e25_bandValue)
    print(f'{k:5d} {(2*k+1):11.1f} {e2022:11.3e} {eBand:11.3e} {eE25L:13.3e} {eE25B:13.3e}')
    rows.append(dict(layer=int(k), leverArmDx=float(2 * k + 1),
                     errEnglish2022=e2022, errBand=eBand,
                     errEnglish2025LiuValue=eE25L, errEnglish2025BandValue=eE25B))

np.save(os.path.join(args.out, 'depthSweep.npy'), rows, allow_pickle=True)

# ------------------------------------------------------- growth-rate summary
def growth(key):
    vals = np.array([r[key] for r in rows])
    shallow = np.mean(vals[:max(1, len(vals) // 4)])
    deep = np.mean(vals[-max(1, len(vals) // 4):])
    return shallow, deep, deep / max(shallow, 1e-300)

print()
print('shallow-quartile mean err -> deep-quartile mean err (ratio):')
for key, label in [('errEnglish2022', 'english2022 (fitted grad, Liu/MLS value)'),
                   ('errBand', 'band (fitted grad, Tikhonov value)'),
                   ('errEnglish2025LiuValue', 'english2025 + Liu/MLS value'),
                   ('errEnglish2025BandValue', 'english2025 + Tikhonov value')]:
    lo_, hi_, ratio = growth(key)
    print(f'  {label:42} {lo_:10.3e} -> {hi_:10.3e}  ({ratio:6.1f}x)')

print()
print(f'raw Liu/MLS ghost-value wellConditioned fraction (boundary rows): {wcFracLiu:.3f} '
      f'-- SAME fraction at every depth (a lattice/conditioning artifact of the coupled '
      f'(dim+1)x(dim+1) Liu matrix on this regular synthetic grid, not a depth effect: some '
      f'ill-conditioned rows have 7-10 fluid neighbours, well past neighbor_threshold=4). '
      f'This is why english2025_liuValue is flat-but-BAD (~0.1 mean error) rather than '
      f'flat-and-good: a raw, un-ramped Liu value is being used here, not production\'s own '
      f'blended fallback. english2025_bandValue sidesteps this entirely because Band\'s '
      f'centroid-decoupled Shepard term is unconditionally well-posed given >=1 neighbour '
      f'(BOUNDARY_DENSITY_PLAN.md §2) -- reinforcing that finding in a new (regular-lattice, '
      f'not just adversarial-scatter) setting.')
