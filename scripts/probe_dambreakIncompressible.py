"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 19, 2026-08-29): the incompressible
scheme on a dam break — the first free-surface scenario it is asked to do that
is not `rotatingSquarePatch`.

Every incompressible case in this document so far is periodic or wall-bounded.
The one free-surface case, `rotatingSquarePatch`, is broken in a way [BK] §5
documents as a method limitation, and it is a hard test: a square in rigid
rotation has four convex free-surface corners and the arms it grows are
surface-tension-sensitive. A dam break is the easier free-surface scenario —
gravity-driven, one mostly-flat free surface, and a well-known shape — so it
answers "does this scheme do free surfaces at all" without the corners
confounding it.

`deltaSPH` is the control. It is the validated scheme on this geometry, and it
differs in exactly the way that matters here: it *integrates* density
(continuity), so a free-surface particle's density is whatever it was advected
with, while the incompressible path *re-sums* it, so a free-surface particle
reads the truncated support directly (~0.5 rho0 at a flat surface). What the
constant-density solve then does about that deficit is the thing to watch.

Reported per sample:

  KE            kinetic energy. The two schemes should agree on the collapse
                if the physics is right; they do not.
  front         max x of any fluid particle -- the surge front. The classic
                dam-break observable.
  height        max y of any fluid particle -- the residual column.
  nLow          fluid particles below 0.9 rho0, i.e. the population that reads
                as free surface under a summation density. Its *trend* is the
                diagnostic: a free surface does not go away, so if this decays
                the surface is being compacted rather than resolved.
  rhoMin/Max    fluid density bounds.

  --mode run       the trajectory comparison above.
  --mode surface   the free-surface *geometry*, which is what makes the
                   density recovery interesting. Under a summation density a
                   particle at a flat free surface reads about `0.5 rho0`
                   because half its kernel support is empty -- and it does, at
                   first (min fluid density 0.518 at t=0.02), then recovers to
                   ~`rho0` while the fluid still plainly has a free surface.
                   Neighbour counts settle what happened, because they are
                   geometry rather than a field the solver can adjust: a
                   surface particle that still has three quarters of the bulk
                   count but reads `rho0` must have its neighbours packed
                   closer than the bulk's. "Surface" here is the topmost
                   particle in each `dx`-wide column -- a purely geometric
                   definition that uses no density at all, so it cannot be
                   circular.

                   **Read the cross-scheme row with care.** By any fixed time
                   the two schemes are at genuinely different flow states (see
                   the front column above), so `deltaSPH` is not a controlled
                   comparison for this, only a sanity check that the
                   measurement is not an artifact of the definition. The clean
                   test is DF against itself with the shift disabled at the
                   surface, which `divergenceFree.py` has commented out and
                   which the plan's "Known-open" entry explains is untestable
                   as written.

Usage:
  python scripts/probe_dambreakIncompressible.py --nx 64 --tLimit 1.5
  python scripts/probe_dambreakIncompressible.py --schemes divergenceFree
  python scripts/probe_dambreakIncompressible.py --mode surface --times 0.02 0.5
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='run', choices=['run', 'surface'])
parser.add_argument('--schemes', nargs='*', default=['divergenceFree', 'deltaSPH'])
parser.add_argument('--times', nargs='*', type=float, default=[0.02, 0.5],
                    help="--mode surface: the times to take the geometry at")
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--tLimit', type=float, default=1.5)
parser.add_argument('--samples', type=int, default=15)
parser.add_argument('--lowDensity', type=float, default=0.9,
                    help="threshold defining the 'reads as free surface' population")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import math

import torch

from warpSPH.runner.cli import caseMain
from warpSPH.cases.freeSurface.dambreak import dambreakCase as case

_origDiag = case.diagnostics


def diagnostics(ctx, state):
    out = dict(_origDiag(ctx, state))
    p = state.state
    fluid = p.kinds == 0
    pos = p.positions[fluid]
    rho = p.densities[fluid]
    out.update(
        front=pos[:, 0].max().detach().cpu().item(),
        height=pos[:, 1].max().detach().cpu().item(),
        nLow=int((rho < args.lowDensity).sum().detach().cpu().item()),
        nFluid=int(fluid.sum().detach().cpu().item()),
    )
    return out


def runScheme(scheme, tLimit, diag=None):
    """One run of `dambreak` under `scheme`, to `tLimit`."""
    if diag is not None:
        case.diagnostics = diag
    argv = ['--nx', str(args.nx), '--tLimit', str(tLimit), '--scheme', scheme,
            '--quiet', '--no-store', '--no-plot']
    if scheme == 'divergenceFree':
        # The PPE derivation is specific to semi-implicit Euler; the case's own
        # default is rungeKutta2, which would solve each RK stage as if it were
        # final and then blend, so the blended velocity is not divergence-free.
        # (`DFSPH_IMPROVEMENT_PLAN.md` "Nothing enforces semiImplicitEuler".)
        # `cflFactor`'s shipped default (0.3) is `deltaSPH`'s own h-based
        # acoustic constant; `dambreakTimestep` reinterprets it as Bender &
        # Koschier's dx-based advective constant under this scheme. Unlike
        # `randomFlowIncompressible --bounded`, this case is NOT stable at
        # their published 0.4 (NaN by step 30) or even 0.3 (NaN by step 76) --
        # measured in `DFSPH_IMPROVEMENT_PLAN.md` Part 20. 0.2 is the
        # recommended value.
        argv += ['--integrationScheme', 'semiImplicitEuler', '--cflFactor', '0.2']
    try:
        return caseMain(case, argv=argv)
    finally:
        case.diagnostics = _origDiag


if args.mode == 'surface':
    from warpSPHCore import SupportScheme, buildVerletList

    print(f"\n=== dambreak free-surface geometry, nx={args.nx} ===")
    print("'surface' = the topmost fluid particle in each dx-wide column, which "
          "uses no\ndensity at all. An un-compacted flat free surface would keep "
          "about half the\nbulk's neighbours; what it reads for density is then "
          "the question.\n")
    hdr = (f"{'scheme':>16} {'t':>6} {'n surf':>7} {'surf nbrs':>10} "
           f"{'bulk nbrs':>10} {'ratio':>7} {'surf rho':>10} {'bulk rho':>10} "
           f"{'min surf rho':>13}")
    print(hdr)
    print('-' * len(hdr))
    for scheme in args.schemes:
        for tLimit in args.times:
            r = runScheme(scheme, tLimit)
            st = r.state.state
            fluid = st.kinds == 0
            pos, rho = st.positions, st.densities
            adj = buildVerletList(st, r.ctx.config.domain, verletScale=1.0,
                                  supportMode=SupportScheme.SuperSymmetric,
                                  priorNeighborhood=None, verbose=False)
            i, j = adj.i, adj.j
            pair = fluid[i] & fluid[j]
            counts = torch.zeros(pos.shape[0], device=pos.device)
            counts.index_add_(0, i[pair],
                              torch.ones(int(pair.sum()), device=pos.device))
            dx = r.ctx.config.dx
            col = torch.round(pos[fluid][:, 0] / dx).long()
            y = pos[fluid][:, 1]
            cf, rf = counts[fluid], rho[fluid]
            top = torch.zeros_like(y, dtype=torch.bool)
            for c in torch.unique(col):
                idx = torch.nonzero(col == c).squeeze(-1)
                top[idx[y[col == c].argmax()]] = True
            deep = ~top
            print(f"{scheme:>16} {tLimit:6.3f} {int(top.sum()):7d} "
                  f"{float(cf[top].mean()):10.2f} {float(cf[deep].mean()):10.2f} "
                  f"{float(cf[top].mean() / cf[deep].mean()):7.3f} "
                  f"{float(rf[top].mean()):10.5f} {float(rf[deep].mean()):10.5f} "
                  f"{float(rf[top].min()):13.5f}", flush=True)
    print("\n`deltaSPH` integrates its density rather than summing it, so its "
          "`surf rho`\ncolumn says nothing about packing -- its `ratio` is the "
          "only comparable cell,\nand even that is at a different flow state by "
          "the same time.")
    raise SystemExit

results = {}
for scheme in args.schemes:
    r = runScheme(scheme, args.tLimit, diag=diagnostics)
    results[scheme] = r

    tr = [row for row in r.trajectory
          if all(not isinstance(v, float) or math.isfinite(v) for v in row.values())]
    print(f"\n=== dambreak --scheme {scheme}, nx={args.nx} "
          f"({len(tr) - 1} steps, {r.wallTime:.0f}s, diverged={r.diverged}) ===")
    hdr = (f"{'t':>7} {'KE':>9} {'front':>8} {'height':>8} {'nLow':>7} "
           f"{'nLow/N':>8} {'rhoMin':>9} {'rhoMax':>9}")
    print(hdr)
    print('-' * len(hdr))
    step = max(1, (len(tr) - 1) // args.samples)
    for row in tr[::step] + [tr[-1]]:
        print(f"{row['t']:7.3f} {row['kineticEnergy']:9.4f} {row['front']:8.4f} "
              f"{row['height']:8.4f} {row['nLow']:7d} "
              f"{row['nLow'] / row['nFluid']:8.4f} {row['minDensity']:9.5f} "
              f"{row['maxDensity']:9.5f}", flush=True)

if len(results) > 1:
    print("\n=== side by side ===")
    hdr = f"{'t':>7}" + ''.join(f"{s + ' KE':>16}{s + ' front':>19}" for s in args.schemes)
    print(hdr)
    series = {}
    for s, r in results.items():
        tr = [row for row in r.trajectory
              if all(not isinstance(v, float) or math.isfinite(v) for v in row.values())]
        series[s] = tr
    n = min(len(v) for v in series.values())
    step = max(1, n // args.samples)
    for i in range(0, n, step):
        line = f"{series[args.schemes[0]][i]['t']:7.3f}"
        for s in args.schemes:
            line += f"{series[s][i]['kineticEnergy']:16.4f}{series[s][i]['front']:19.4f}"
        print(line)
