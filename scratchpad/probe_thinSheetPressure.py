"""DELTASPH_VALIDATION_PLAN.md 5.11's open instrument, finally built: does an
SPH pressure force blow up near an edge purely from truncated kernel support
-- no wall, no resume, no chaos, no density/velocity gradient at all?

Hypothesis (textbook SPH, not new physics): the symmetric pair term
`Sum_j V_j (P_i+P_j) grad_i W_ij` used by `nonConservative` and (whenever its
switch selects the symmetric branch) `Antuono` equals `~grad(P)` only when the
kernel-gradient sum `Sum_j V_j grad_i W_ij` itself vanishes -- true in the
bulk, false near any truncated support (a free surface, an edge, a corner).
For a spatially UNIFORM pressure field P0 (zero true gradient, so the
physically-correct force is exactly zero everywhere) the symmetric term
reduces to `2 P0 * Sum_j V_j grad_i W_ij`: a genuine force proportional to P0
and to how truncated the local support is, not an artifact of density
variation. The plain antisymmetric `conservative` term
`Sum_j V_j (P_j-P_i) grad_i W_ij` is *exactly* zero for uniform P at any
truncation, term-by-term, since P_j-P_i=0 for every pair.

This predicts, and this probe measures directly by building one thick slab
(vacuum above, real fluid rows below) and evaluating every row -- i.e. sweeping
depth-from-the-open-edge in dx within a single, consistent configuration:
`nonConservative` force should be large right at the edge and decay into the
bulk; `conservative` should be ~0 everywhere; `Antuono` should match
`conservative` (~0) whenever its switch picks the antisymmetric branch --
P0 < 0 and not flagged free-surface -- and match `nonConservative`'s edge
artifact whenever it doesn't: P0 >= 0 (always, per the equation itself) or
flagged free-surface (mask_i=1, unconditionally). The free-surface case is the
mechanism DELTASPH_VALIDATION_PLAN.md 5.14 traced for the sloshingTank
ceiling-hover particle (isolated free-surface film, P_i<0, forced symmetric
via mask); the P0>=0 case is the candidate mechanism for 5.13 item 2's
right-wall kick (wall-adjacent thinning under mild ambient compression,
forced symmetric regardless of any free-surface flag).

Two further sweeps settle the resume-fidelity "chaos" question from the same
plan section's item 2 addendum. `--phaseShifts` offsets a perfect lattice by a
sub-dx amount relative to the domain edge: result (see plan) -- **no effect
at all**, force depends only on integer depth, not sub-dx phase, because a
perfect lattice is translation-invariant. `--jitter` instead displaces every
particle by an independent random offset (real disorder, not a coherent
shift): with as little as 0.1 dx of jitter, forces at depths 3-4 -- exactly
zero on the perfect lattice -- jump to the same order of magnitude as the
deterministic depth-0/1 edge effect, and vary 5-7x between otherwise-identical
random seeds. Disorder, not systematic edge distance, is what makes this
mechanism reach several particle-layers deep and become essentially
unpredictable -- which is the direct mechanism behind both the right-wall
kick extending past the very first wall-adjacent row, and the resumed vs.
archived trajectory divergence in that investigation.

    python scratchpad/probe_thinSheetPressure.py
"""
from __future__ import annotations

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--supportRatio', type=float, default=4.0, help='support / dx (h/dx=2, support=2h)')
ap.add_argument('--nRows', type=int, default=24)
ap.add_argument('--depths', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
ap.add_argument('--c0', type=float, default=40.0)
ap.add_argument('--p0Frac', type=float, nargs='+', default=[-0.05, 0.05],
                help='P0 as a fraction of rho0*c0^2, swept both signs')
ap.add_argument('--phaseShifts', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5],
                help='sub-dx shifts of the whole slab relative to the domain edge, in units of dx')
ap.add_argument('--jitter', type=float, default=0.0,
                help='disorder: random per-particle position jitter, in units of dx (0 = perfect lattice)')
ap.add_argument('--jitterSeeds', type=int, nargs='+', default=[0],
                help='repeat the jittered build with different seeds to show run-to-run spread')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPHCore import KernelFunctions, DomainDescription, SupportScheme, buildVerletList
from warpSPH.modules.pressure import computePressureForceSurfaceAware
from warpSPH.enumTypes import PressureForceScheme

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
dx = args.dx
RHO0 = 1.0
support = args.supportRatio * dx


class _Cfg:
    pass


def buildSlab(nRows: int, phaseShift: float, nCols: int = 61, jitter: float = 0.0, seed: int = 0):
    """Periodic-in-x, finite-in-y slab: row 0 is the open top edge (vacuum
    above), rows increase downward into the (effectively infinite, `nRows`
    large relative to `support`) bulk. `phaseShift` (in dx) offsets the whole
    slab relative to y=0 -- irrelevant physically (translation invariance) but
    tests whether the force is sensitive to where exactly the edge sits
    relative to the lattice/domain, i.e. tests the symmetry-sensitivity
    reading directly. `jitter` displaces every particle by an independent
    random offset (uniform in [-jitter, +jitter]*dx per axis) to test whether
    real disorder, not lattice phase, is what drives sensitivity."""
    xs = (torch.arange(nCols, device=dev, dtype=DT) - nCols // 2) * dx
    ys = -(torch.arange(nRows, device=dev, dtype=DT) + phaseShift) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    if jitter > 0.0:
        g = torch.Generator(device=dev).manual_seed(seed)
        pos = pos + (torch.rand(pos.shape, generator=g, device=dev, dtype=DT) * 2 - 1) * jitter * dx

    st = _Cfg()
    st.positions = pos
    st.velocities = torch.zeros_like(pos)
    st.kinds = torch.zeros(pos.shape[0], dtype=torch.int32, device=dev)
    st.supports = torch.full((pos.shape[0],), support, device=dev, dtype=DT)
    st.masses = torch.full((pos.shape[0],), dx * dx, device=dev, dtype=DT)
    st.densities = torch.full((pos.shape[0],), RHO0, device=dev, dtype=DT)
    st.UIDs = torch.arange(pos.shape[0], device=dev, dtype=torch.int64)
    return st, nCols


def rowIndex(nCols, nRows, row):
    # flat index for (middle column, given row) in a (nCols, nRows) meshgrid
    # flattened row-major over (col, row) -> flat = col*nRows + row
    return (nCols // 2) * nRows + row


scheme = _Cfg()
scheme.fluid = _Cfg()
scheme.fluid.restDensity = RHO0
scheme.fluid.fixedSoundSpeed = args.c0

print(f'dx={dx}  support={support/dx:.2f}dx  nRows={args.nRows}  depths swept={args.depths}')

sweepLabel, sweepValues = (('jitter seed', args.jitterSeeds) if args.jitter > 0.0
                           else ('edge phase shift', args.phaseShifts))

for sweepValue in sweepValues:
    phaseShift, seed = (0.0, sweepValue) if args.jitter > 0.0 else (sweepValue, 0)
    st, nCols = buildSlab(args.nRows, phaseShift, jitter=args.jitter, seed=seed)
    L = nCols * dx
    yMin = float(st.positions[:, 1].min()) - 2 * support
    yMax = float(st.positions[:, 1].max()) + 2 * support
    domain = DomainDescription(
        torch.tensor([-L / 2, yMin], device=dev, dtype=DT),
        torch.tensor([L / 2, yMax], device=dev, dtype=DT),
        torch.tensor([True, False], device=dev, dtype=torch.bool), 2)
    cfg = _Cfg()
    cfg.kernel = KernelFunctions.Wendland2
    cfg.dim = 2
    cfg.domain = domain

    adj = buildVerletList(st, domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)

    unit = 'dx' if args.jitter == 0.0 else ''
    print(f'\n--- {sweepLabel} = {sweepValue}{unit} (jitter={args.jitter}dx) ---')
    print(f'{"depth":>5} {"nNb":>4} | ' +
          ' | '.join(f'P0={f:+.2f}: cons/nonCons/Antuono/Antuono+SF' for f in args.p0Frac))

    for depth in args.depths:
        i0 = rowIndex(nCols, args.nRows, depth)
        d = (st.positions - st.positions[i0]).norm(dim=1)
        nNb = int((d < support).sum()) - 1

        cells = []
        for p0Frac in args.p0Frac:
            P0 = p0Frac * RHO0 * args.c0 ** 2
            st.pressures = torch.full((st.positions.shape[0],), P0, device=dev, dtype=DT)
            st.surfaceIndicators = torch.zeros(st.positions.shape[0], device=dev, dtype=torch.int32)

            forces = {}
            for term in (PressureForceScheme.conservative, PressureForceScheme.nonConservative,
                         PressureForceScheme.Antuono):
                scheme.pressureForceTerm = term
                dvdt = computePressureForceSurfaceAware(st, cfg, scheme, adj)
                forces[term.name] = float(dvdt[i0].norm())

            st.surfaceIndicators[i0] = 1
            scheme.pressureForceTerm = PressureForceScheme.Antuono
            dvdt = computePressureForceSurfaceAware(st, cfg, scheme, adj)
            forces['Antuono_maskedSF'] = float(dvdt[i0].norm())

            cells.append(f"{forces['conservative']:8.3g}/{forces['nonConservative']:8.3g}/"
                         f"{forces['Antuono']:8.3g}/{forces['Antuono_maskedSF']:8.3g}")
        print(f'{depth:5d} {nNb:4d} | ' + ' | '.join(cells))
