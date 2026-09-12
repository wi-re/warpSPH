#!/usr/bin/env python
"""Does the mDBC wall generate pressure against *sparse* fluid it shouldn't?

A few particles sitting near a wall -- one, a pair, a small block -- carry
`rho = rho0` exactly (density is integrated, not summed, so an isolated
particle keeps its initial value). The correct boundary response is therefore
**zero**: the MLS fit of a constant field must return `rho_b = rho0`, and where
the stencil is too thin the blend in `computeMdbcDensity` is supposed to fall
back to Shepard / rest density, which is also `rho0`. Any `p_b != 0` here is
spurious, and a spurious `p_b` kicks whichever few fluid particles happen to be
adjacent -- with no penetration required, which is what the M31 / sloshingTank
failures look like (`DELTASPH_VALIDATION_PLAN.md` 5.10 / 5.12).

Sweeps, against a synthetic flat wall in both orientations (ceiling: solid
above, fluid below; side wall: solid left, fluid right):

  * cluster shape -- single particle, tangential pair (`dx` apart), normal
    pair, 2x2 block, 3x3 block;
  * normal distance of the nearest cluster particle from the surface;
  * velocity -- at rest, receding, approaching, and several tangential speeds
    (the ceiling case the report calls out specifically).

Reports `max |p_b|` over the wall particles near the cluster, plus the ghost
stencil's own conditioning (`numNeighbors`, `|det A_g|`, and the blend weight
`w`) so a nonzero response can be read straight off as either "bad fit" or
"fallback not engaging".

    python scripts/probe_mdbcSparseFluid.py [--dx 0.01] [--c0 40] [--out DIR]
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--c0', type=float, default=40.0, help='sound speed for p_b = c0^2 (rho_b - rho0)')
ap.add_argument('--supportRatio', type=float, default=2.5, help='support / dx')
ap.add_argument('--layers', type=int, default=5, help='boundary layers')
ap.add_argument('--out', default=os.path.join('scratchpad', 'mdbc_sparse'))
ap.add_argument('--tol', type=float, default=1e-4,
                help='|p_b| above this (in units of rho0 c0^2) is flagged')
ap.add_argument('--boundarySupportScale', type=float, default=1.0,
                help='multiply the boundary+ghost particles support by this before '
                     'building the neighbour list (Marrone 2011 uses a larger search '
                     'radius for the boundary interpolation)')
ap.add_argument('--surfaceMask', action='store_true',
                help='flag the fluid cluster as free-surface (mask_i=1), which forces '
                     'the Antuono symmetric branch -- the realistic case for an '
                     'isolated near-wall particle')
ap.add_argument('--rhoSweep', action='store_true',
                help='sweep the cluster density above AND below rho0 and report the '
                     'pressure jump at the wall instead of the rho0-only table')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch

from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList)
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.modules.liu import interpolateLiuLiu, determinantThresholdFor
from warpSPH.modules.pressure import computePressureForceSurfaceAware
from warpSPH.enumTypes import PressureForceScheme
from warpSPHCore import OperationDirection

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
dx = args.dx
RHO0 = 1.0
support = args.supportRatio * dx

#: inward normal (points from the solid into the fluid) and the tangent
WALLS = {
    'ceiling': (np.array([0.0, -1.0]), np.array([1.0, 0.0])),
    'sidewall': (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
}

#: cluster offsets in (tangential, normal) multiples of dx, normal >= 0 away
#: from the surface
SHAPES = {
    'single': [(0, 0)],
    'pair-tangential': [(0, 0), (1, 0)],
    'pair-normal': [(0, 0), (0, 1)],
    'block-2x2': [(0, 0), (1, 0), (0, 1), (1, 1)],
    'block-3x3': [(i, j) for i in range(3) for j in range(3)],
}

DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0]      # nearest particle, in dx past the surface

#: (label, normal component, tangential component) -- normal +1 = receding
VELOCITIES = [
    ('at rest',        0.0,  0.0),
    ('receding  n=+1', +1.0, 0.0),
    ('approaching n=-1', -1.0, 0.0),
    ('tangential t=0.5', 0.0, 0.5),
    ('tangential t=1',  0.0, 1.0),
    ('tangential t=2',  0.0, 2.0),
    ('tangential t=5',  0.0, 5.0),
]


class _Cfg:
    pass


def buildState(nhat, that, shape, distDx, vN, vT):
    """Synthetic wall band + a small fluid cluster. The wall surface is the
    plane through the origin with inward normal `nhat`; boundary layer `k` sits
    at `-(k + 0.5) dx * nhat` and its ghost mirrors to `+(k + 0.5) dx * nhat`."""
    pos, vel, kinds, goff = [], [], [], []

    # wall spans well past the support radius either side of the cluster
    nSpan = int(np.ceil((6 * support) / dx))
    for k in range(args.layers):
        base = -(k + 0.5) * dx * nhat
        for s in range(-nSpan, nSpan + 1):
            p = base + s * dx * that
            pos.append(p.tolist()); vel.append([0.0, 0.0]); kinds.append(1)
            goff.append(((2 * k + 1) * dx * nhat).tolist())   # r_b - r_g
    nBnd = len(pos)

    ghostStart = nBnd
    for bi in range(nBnd):
        b = np.array(pos[bi]); o = np.array(goff[bi])
        pos.append((b - o).tolist()); vel.append([0.0, 0.0]); kinds.append(2)
        goff.append((-o).tolist())

    fluidStart = len(pos)
    v = (vN * nhat + vT * that).tolist()
    for (t, n) in shape:
        p = (distDx + n) * dx * nhat + t * dx * that
        pos.append(p.tolist()); vel.append(v); kinds.append(0)
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
    if args.boundarySupportScale != 1.0:
        st.supports[st.kinds != 0] = support * args.boundarySupportScale
    st.masses = torch.full((nTot,), dx * dx, device=dev, dtype=DT)
    st.densities = torch.full((nTot,), RHO0, device=dev, dtype=DT)
    return st, nBnd, fluidStart


def run(nhat, that, shape, distDx, vN, vT):
    st, nBnd, fluidStart = buildState(nhat, that, shape, distDx, vN, vT)

    cfg = _Cfg()
    cfg.kernel = KernelFunctions.Wendland2
    cfg.dim = 2
    cfg.verletScale = 1.0
    lo = st.positions.min(dim=0).values - 8 * dx
    hi = st.positions.max(dim=0).values + 8 * dx
    cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
    cfg.dx = dx

    scheme = _Cfg()
    scheme.fluid = _Cfg()
    scheme.fluid.restDensity = RHO0
    scheme.fluid.fixedSoundSpeed = args.c0
    scheme.fluid.kappa = 1.3
    scheme.fluid.polytropicExponent = 7.0

    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)

    merged = computeMdbcDensity(st, cfg, scheme, adj)
    rho_b = merged[st.kinds == 1]
    p_b = args.c0 ** 2 * (rho_b - RHO0)

    # stencil conditioning at the ghost nodes, same call computeMdbcDensity makes
    rho_i, grad_i, nNb, A_g, b, wc = interpolateLiuLiu(
        st.positions[st.kinds == 2], referenceParticles=st,
        referenceQuantities=st.densities, config=cfg,
        neighbor_threshold=4, direction=OperationDirection.FluidToGhost,
        supportScale=1.0, adjacency=adj.hashMap if hasattr(adj, 'hashMap') else None)
    det = torch.linalg.det(A_g).abs()
    detFloor = determinantThresholdFor(cfg.kernel)
    wDet = torch.clamp((det - detFloor) / (detFloor * 0.25), 0.0, 1.0)
    wN = torch.clamp((nNb.to(det.dtype) - 4.0) / 1.0, 0.0, 1.0)
    w = wDet * wN

    # only wall particles whose ghost can actually see the cluster matter
    active = nNb > 0
    j = int(torch.argmax(p_b.abs()))
    return dict(
        maxAbsP=float(p_b.abs().max()),
        pAt=float(p_b[j]),
        rhoAt=float(rho_b[j]),
        nActive=int(active.sum()),
        nNbMax=int(nNb.max()) if nNb.numel() else 0,
        wMax=float(w.max()) if w.numel() else 0.0,
        wAtActive=float(w[active].max()) if int(active.sum()) else 0.0,
    )


#: A lone or near-surface particle is never at exactly rho0 -- it is usually
#: *below* it, and can be above it after an impact. Both sides matter: the
#: fallbacks in `computeMdbcDensity` (`clamp(shepardDensity, min=rho0)` and the
#: `numNeighbors > 1` gate that otherwise returns rho0) are one-sided, so they
#: can pin `rho_b = rho0` against sub-rho0 fluid while letting super-rho0 fluid
#: through. That asymmetry is a *rectifying* error: the wall refuses to follow
#: the fluid into tension but follows it into compression, leaving a pressure
#: jump `dp = p_fluid - p_b` across the interface with a consistent sign.
RHO_SWEEP = [0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10]


def runRho(nhat, that, shape, distDx, rhoFluid):
    """Same configuration, but the cluster carries `rhoFluid` instead of rho0.
    Returns the wall's response and the resulting pressure jump."""
    st, nBnd, fluidStart = buildState(nhat, that, shape, distDx, 0.0, 0.0)
    st.densities[st.kinds == 0] = rhoFluid

    cfg = _Cfg()
    cfg.kernel = KernelFunctions.Wendland2
    cfg.dim = 2
    cfg.verletScale = 1.0
    lo = st.positions.min(dim=0).values - 8 * dx
    hi = st.positions.max(dim=0).values + 8 * dx
    cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
    cfg.dx = dx

    scheme = _Cfg()
    scheme.fluid = _Cfg()
    scheme.fluid.restDensity = RHO0
    scheme.fluid.fixedSoundSpeed = args.c0
    scheme.fluid.kappa = 1.3
    scheme.fluid.polytropicExponent = 7.0

    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)

    merged = computeMdbcDensity(st, cfg, scheme, None)

    _, _, nNb, A_g, _, _ = interpolateLiuLiu(
        st.positions[st.kinds == 2], referenceParticles=st,
        referenceQuantities=st.densities, config=cfg,
        neighbor_threshold=4, direction=OperationDirection.FluidToGhost,
        supportScale=1.0, adjacency=adj.hashMap if hasattr(adj, 'hashMap') else None)

    rho_b = merged[st.kinds == 1]
    active = nNb > 0
    # the wall particles that can actually see the cluster are the ones whose
    # reading matters; elsewhere rho_b is the inert dry-wall rest value
    rhoWall = float(rho_b[active].mean()) if int(active.sum()) else RHO0
    rhoWallExtreme = float(rho_b[active][torch.argmax((rho_b[active] - RHO0).abs())]) \
        if int(active.sum()) else RHO0
    p_fluid = args.c0 ** 2 * (rhoFluid - RHO0)
    p_wall = args.c0 ** 2 * (rhoWallExtreme - RHO0)

    # -- the actual pressure acceleration, via the same term the scheme uses.
    # The jump alone does not fix the sign: by pair force `(p_i + p_j)` a
    # mirrored wall is *more* attractive, by pressure gradient a mirrored wall
    # gives zero force. Only the assembled operator settles it.
    st.densities = merged
    st.pressures = args.c0 ** 2 * (merged - RHO0)
    st.surfaceIndicators = torch.zeros(st.positions.shape[0], device=dev, dtype=torch.int32)
    if args.surfaceMask:
        st.surfaceIndicators[st.kinds == 0] = 1
    scheme.pressureForceTerm = PressureForceScheme.Antuono
    dvdt = computePressureForceSurfaceAware(st, cfg, scheme, adj)
    aFluid = dvdt[st.kinds == 0]
    # signed normal acceleration: + = away from the wall, - = pulled INTO it
    nvec = torch.tensor(nhat, device=dev, dtype=DT)
    aN = float((aFluid @ nvec).mean())

    return dict(rhoWall=rhoWall, rhoWallExtreme=rhoWallExtreme,
                pFluid=p_fluid, pWall=p_wall, dp=p_fluid - p_wall,
                aN=aN, aMag=float(aFluid.norm(dim=1).max()),
                nActive=int(active.sum()), nNbMax=int(nNb.max()) if nNb.numel() else 0)


os.makedirs(args.out, exist_ok=True)
pScale = RHO0 * args.c0 ** 2

if args.rhoSweep:
    print(f'dx={dx}  support={support/dx:.2f} dx  layers={args.layers}  c0={args.c0}')
    print('Cluster carries rho_fluid; wall should follow it so the jump dp -> 0.')
    print('dp = (p_fluid - p_b) / (rho0 c0^2);  dp < 0 pulls the particle INTO the wall.')
    for wallName, (nhat, that) in WALLS.items():
        for shapeName in ('single', 'pair-tangential', 'block-2x2', 'block-3x3'):
            for dist in (0.5, 1.0):
                print(f'\n--- {wallName} / {shapeName} / d={dist}dx ---')
                print(f'{"rho_fluid":>10} {"rho_b":>10} {"p_fluid":>11} {"p_b":>11} '
                      f'{"dp":>11} {"a_normal":>12} {"nNbMax":>7}')
                for rf in RHO_SWEEP:
                    r = runRho(nhat, that, SHAPES[shapeName], dist, rf)
                    print(f'{rf:10.3f} {r["rhoWallExtreme"]:10.5f} '
                          f'{r["pFluid"]/pScale:11.4f} {r["pWall"]/pScale:11.4f} '
                          f'{r["dp"]/pScale:11.4f} {r["aN"]:12.3e} {r["nNbMax"]:7d}')
    raise SystemExit(0)

flagged = []

print(f'dx={dx}  support={support/dx:.2f} dx  layers={args.layers}  c0={args.c0}')
print(f'p_b reported as |p_b| / (rho0 c0^2); flag threshold {args.tol:g}')
print()
for wallName, (nhat, that) in WALLS.items():
    for shapeName, shape in SHAPES.items():
        print(f'--- {wallName} / {shapeName} ' + '-' * (52 - len(wallName) - len(shapeName)))
        print(f'{"velocity":>18} | ' + ' '.join(f'{d:>10.1f}dx' for d in DISTANCES))
        for (vlabel, vN, vT) in VELOCITIES:
            cells = []
            for dist in DISTANCES:
                r = run(nhat, that, shape, dist, vN, vT)
                rel = r['maxAbsP'] / pScale
                cells.append(f'{rel:12.3e}')
                if rel > args.tol:
                    flagged.append((wallName, shapeName, vlabel, dist, rel, r))
            print(f'{vlabel:>18} | ' + ' '.join(cells))
        print()

print('=' * 78)
if not flagged:
    print(f'PASS: every configuration produced |p_b| <= {args.tol:g} rho0 c0^2')
else:
    print(f'FLAGGED {len(flagged)} configurations with spurious wall pressure:')
    for (wallName, shapeName, vlabel, dist, rel, r) in flagged[:40]:
        print(f'  {wallName:9s} {shapeName:16s} {vlabel:18s} d={dist:4.1f}dx  '
              f'|p_b|={rel:.3e}  rho_b={r["rhoAt"]:.5f}  '
              f'nNbMax={r["nNbMax"]}  w(active)={r["wAtActive"]:.3f}')
    if len(flagged) > 40:
        print(f'  ... and {len(flagged) - 40} more')
