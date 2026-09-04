#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/liu/wp_mat.py -- Tier 2 of
docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeLiuMatricesWarp` builds the Liu-style moving-least-squares moment
matrix and its companion Shepard/gradient vector: for each query point it
accumulates `[<1>, <x>]`/`[[<1>,<x>],[grad<1>,grad<x>]]`-shaped sums over a
scalar reference field (`referenceQuantities`), one term per neighbor,
purely linearly (`out_vec += temp_vec`, `out_mat += temp_mat`, `sh_out +=
out_shep`, `nnbrs += nbrs`) with no nonlinear reduction of any of them after
the loop -- unlike `computeAlphaWarp` (`gradcheck_incompressible.py`) and
barecasco's `wp.normalize` (`gradcheck_surfaceDetection.py`), this module
never reads back a loop-accumulated variable through a nonlinear op, so it
does not hit that bug class. It shares the family's usual different-array
apparent-volume ternary (`V_j = mj / rhoj if not useVolume else Vj`,
confirmed non-issue since Tier 0).

Its kernel is a **four-output** launch -- `(shep_out, vector_out, matrix_out,
numNeighbors_out)`, mixing three float-family outputs with one `wp.int32`
neighbor-count array -- structurally the same multi-output, mixed-dtype
shape that broke `computeMdbcNoPenShiftWarp` (Tier 2's `mdbc` script) before
the `_dtype_is_float` bridge fix. This script's `nnbrs` output not requiring
grad (verified below) is exactly that fix doing its job: without it, this
would crash with the same `RuntimeError` the mdbc finding fixed, since
`nnbrs`/`numNeighbors_out` is a genuine `wp.int32` array. Also note
`queryPositions` is a *separate* tensor argument from `referenceParticles`
(not read off it) -- the kernel indexes `queryPositions[i]` for the query
position but still reads `referenceParticles`'s per-index correction data
(`getL_i`/`getCRK_i`/...) at the *same* index `i`, so this script uses one
particle set for both, matching every real call site's usage.

    python scripts/gradcheck_liu.py
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

from warpSPH.modules.liu.wp_mat import computeLiuMatricesWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    quantities = torch.randn(N, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return domain, positions, supports, masses, densities, adjacency, kinds, quantities


def _run() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, quantities = _build_case()

    def f(pos, sup, mass, q):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        shep, vec, mat, nnbrs = computeLiuMatricesWarp(
            queryPositions=pos,
            referenceParticles=p,
            referenceQuantities=q,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )
        assert not nnbrs.requires_grad, "int32 neighbor-count output must not require grad (_dtype_is_float guard)"
        return shep, vec, mat

    print("\n=== computeLiuMatricesWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, quantities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = _run()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
