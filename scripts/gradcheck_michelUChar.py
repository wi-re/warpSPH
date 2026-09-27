#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/shifting/wp_michelUChar.py --
Stage A of PST_ALE_PLAN.md.

`computeUCharWarp` computes Michel et al. 2022's (`literature/michel2022`)
Eq. (20) characteristic velocity `U_char_i = max_j |(u_j-u_i).x_hat_ij|`, a
`wp.max` neighbor reduction. Built from the start with the two-pass
argmax/re-evaluate split `modules/shockCapturing/wp_vsig.py` needed a real
bug hunt to arrive at (see `gradcheck_shockCapturing.py`'s docstring): a
forward-only pass finds the winning neighbor index (never differentiated),
and a single re-evaluation outside any loop computes the actual
differentiable value for that one index. Gradchecked against positions and
velocities on a hand-picked, tie-free 3-particle line (asymmetric positions,
a strong velocity spread so the true argmax neighbor has a comfortable
margin over every other candidate, including the self-term which is skipped
entirely).

    python scripts/gradcheck_michelUChar.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, build_grid_adjacency, compute_densities, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.shifting.wp_michelUChar import computeUCharWarp

DIM = 1


def _run() -> bool:
    # Same tie-free construction as gradcheck_shockCapturing.py's _run_vsig:
    # asymmetric positions, strong outward velocity spread, so each
    # particle's true argmax neighbor has a comfortable margin (>=1.0, vs.
    # eps=1e-6) over every other candidate.
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-0.5], [0.0], [0.3]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((3,), 1.5, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocities = torch.tensor([[2.5], [0.0], [-1.5]], dtype=DTYPE, device=DEVICE, requires_grad=True)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    def f(pos, sup, mass, dens, vel):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeUCharWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=vel,
            adjacency=adjacency,
        )

    print("\n=== computeUCharWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_grid_consistency() -> bool:
    # Forward-value-only: AdjacencyList and grid (CompactHashMap) traversal
    # must agree on the same particle set -- guards the missing
    # compact-support-filter class of bug gradcheck_shockCapturing.py's
    # docstring (finding 4) describes for the sibling wp_vsig.py kernel.
    DIM2 = 2
    domain = make_domain(dim=DIM2)
    torch.manual_seed(0)
    N2 = 6
    positions = torch.randn(N2, DIM2, dtype=DTYPE, device=DEVICE) * 0.5
    supports = torch.full((N2,), 1.2, dtype=DTYPE, device=DEVICE)
    masses = torch.full((N2,), 1.0, dtype=DTYPE, device=DEVICE)
    velocities = torch.randn(N2, DIM2, dtype=DTYPE, device=DEVICE) * 2.0

    adjacency_list, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    grid, _ = build_grid_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency_list, mode=SupportScheme.Gather).detach()

    def run(adj):
        p = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
        return computeUCharWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=velocities,
            adjacency=adj,
        )

    print("\n=== computeUCharWarp: AdjacencyList vs. grid traversal forward-value consistency ===")
    try:
        out_adj = run(adjacency_list)
        out_grid = run(grid)
        maxDiff = (out_adj - out_grid).abs().max().item()
        ok = maxDiff < 1e-9
        print(("PASSED" if ok else "FAILED") + f" (max abs diff: {maxDiff})")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= _run()
    ok &= _run_grid_consistency()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
