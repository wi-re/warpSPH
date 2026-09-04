#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/util/{wp_sum,wp_numNeighbors}.py --
Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`warpSum` (`sumOverNeighbors`'s underlying kernel) sums a caller-supplied
per-particle scalar field over each particle's kernel-support neighborhood:
`out_i = sum_j queryValues[j] * (w_ij > 0)`. Purely linear accumulation of a
real-valued input array with no post-loop nonlinearity -- the safe shape
every prior tier has confirmed, and distinct from `computeAlphaWarp`
(`gradcheck_incompressible.py`) and barecasco's normals
(`gradcheck_surfaceDetection.py`), both of which apply a nonlinear op to a
loop-accumulated value *after* the loop. Gradchecked here against
`queryValues` -- clean.

`countNeighborsWarp` is a genuine discrete count (`out += 1` per neighbor
inside the kernel-support gate) -- the same "not in scope for gradcheck"
shape as `wp_maronne.py`'s and barecasco's second output
(`gradcheck_surfaceDetection.py`), but with one structural difference worth
calling out: its output dtype is taken from `queryParticles.kinds` (a
genuine `wp.int32` array), not from `densities` (float), so the
`_dtype_is_float` bridge fix (added for `gradcheck_mdbc.py`'s multi-output
finding) applies *directly* at the top level here rather than only inside a
comparison the Python wrapper performs afterward -- `requires_grad` is
`False` on the very output `warpWrapper2` hands back, before any caller-side
op. Verified below as a regression check on that fix, the same role
`gradcheck_liu.py`'s `nnbrs` assertion plays for its own multi-output kernel.

    python scripts/gradcheck_util.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.util.wp_numNeighbors import countNeighborsWarp
from warpSPH.modules.util.wp_sum import warpSum

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    return domain, positions, supports, masses, densities, adjacency, kinds


def _run_warpSum() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    values = torch.randn(N, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(v):
        p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities, kinds=kinds)
        return warpSum(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryValues=v,
            adjacency=adjacency,
        )

    print("\n=== warpSum: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (values,), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_countNeighbors_not_in_scope() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities, kinds=kinds)
    out = countNeighborsWarp(
        queryParticles=p,
        operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
        domain=domain,
        adjacency=adjacency,
    )
    print("\n=== countNeighborsWarp: not in scope, verifying int32/requires_grad=False ===")
    ok = out.dtype == torch.int32 and out.requires_grad is False
    print(("PASSED" if ok else "FAILED") + f" (dtype={out.dtype}, requires_grad={out.requires_grad})")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= _run_warpSum()
    ok &= _run_countNeighbors_not_in_scope()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
