#!/usr/bin/env python3
"""torch.autograd.gradcheck against
modules/surfaceDetection/wp_nearestSurfaceNormal.py -- Stage A of
PST_ALE_PLAN.md.

`computeNearestSurfaceNormalWarp` computes Michel et al. 2022's
(`literature/michel2022`) Eq. (47) "inherited normal" `n_tilde_i` and its
paired distance `d_i^FS`: a free-surface particle uses its own normal and
`d^FS=0`; any other particle inherits the normal of, and its distance to, the
*nearest* free-surface-particle neighbor. Same two-pass split as
`modules/shockCapturing/wp_vsig.py` / `modules/shifting/wp_michelUChar.py`,
argmin variant: a forward-only pass finds the winning neighbor index (never
differentiated), then a single re-evaluation outside any loop computes the
actual differentiable distance and gathers the normal for that one index.

`freeSurfaceMask` is a discrete selection gate (`if mask[j] <= 0.5: continue`
in the forward-only pass), not a continuously-blended value anywhere in the
kernel, so it is passed fixed (no `requires_grad`) -- unlike
`dilateSurfaceMaskWarp`'s mask (`gradcheck_surfaceDetection.py`'s
`_run_dilate`), which *is* summed continuously and is checked with
`requires_grad=True` there. Gradchecked here against positions (feeding
`d^FS`) and normals (feeding `n_tilde` via a plain gather) on a hand-picked,
tie-free 5-particle line with the two endpoints marked as the free-surface
set.

    python scripts/gradcheck_nearestSurfaceNormal.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.surfaceDetection.wp_nearestSurfaceNormal import computeNearestSurfaceNormalWarp

DIM = 1


def _run() -> bool:
    # Endpoints (idx 0, 4) marked free-surface; the three interior particles
    # each have an unambiguous nearest marked neighbor with a comfortable
    # margin (>=0.2, vs. eps=1e-6): idx1(-0.4) -> idx0 (margin 0.8),
    # idx2(0.1) -> idx4 (margin 0.2), idx3(0.5) -> idx4 (margin 1.0).
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-1.0], [-0.4], [0.1], [0.5], [1.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((5,), 3.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((5,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    normals = torch.tensor([[1.0], [0.3], [-0.2], [0.4], [-1.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    freeSurfaceMask = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], dtype=DTYPE, device=DEVICE)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    def f(pos, sup, mass, dens, nrm):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        d_fs, n_tilde = computeNearestSurfaceNormalWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            freeSurfaceMask=freeSurfaceMask,
            normals=nrm,
            adjacency=adjacency,
        )
        return d_fs, n_tilde

    print("\n=== computeNearestSurfaceNormalWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, normals)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_forward_sanity() -> bool:
    # Not-in-scope / free-surface-particle branches, checked directly against
    # hand-computed expectations rather than gradcheck (which only exercises
    # the interior-particle argmin branch above).
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-1.0], [-0.4], [0.1], [0.5], [1.0]], dtype=DTYPE, device=DEVICE)
    supports = torch.full((5,), 3.0, dtype=DTYPE, device=DEVICE)
    masses = torch.full((5,), 1.0, dtype=DTYPE, device=DEVICE)
    normals = torch.tensor([[1.0], [0.3], [-0.2], [0.4], [-1.0]], dtype=DTYPE, device=DEVICE)
    freeSurfaceMask = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], dtype=DTYPE, device=DEVICE)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()

    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    d_fs, n_tilde = computeNearestSurfaceNormalWarp(
        queryParticles=p,
        operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
        domain=domain,
        freeSurfaceMask=freeSurfaceMask,
        normals=normals,
        adjacency=adjacency,
    )

    expected_d = torch.tensor([0.0, 0.6, 0.9, 0.5, 0.0], dtype=DTYPE)
    expected_n = torch.tensor([[1.0], [1.0], [-1.0], [-1.0], [-1.0]], dtype=DTYPE)

    print("\n=== computeNearestSurfaceNormalWarp: forward-value sanity check ===")
    ok = torch.allclose(d_fs, expected_d, atol=1e-9) and torch.allclose(n_tilde, expected_n, atol=1e-9)
    print(("PASSED" if ok else "FAILED") + f" (d_fs={d_fs.tolist()}, n_tilde={n_tilde.squeeze(-1).tolist()})")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= _run_forward_sanity()
    ok &= _run()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
