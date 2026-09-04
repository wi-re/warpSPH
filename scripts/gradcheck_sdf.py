#!/usr/bin/env python3
"""torch.autograd.gradcheck against geometry/sdf.py's sampleSDF and
regions/domainSDF.py's sampleDomainSDF -- Tier 0 of docs/historic_plans/CLEANUP_PLAN.md's
Phase 4.1 gradcheck rollout.

Different code shape from the other 24 gradcheck targets: no Warp kernel at
all. Both functions compute a scalar SDF value `d = sdf(x_)` in plain
PyTorch and then hand-roll the gradient (surface normal) via
`torch.autograd.grad(d, x_, create_graph=True, retain_graph=True)`, with the
result conditionally detached depending on whether the *caller's* `x`
requires grad. That dual-mode design -- pay for a retained graph only when a
caller will actually backprop through the normal -- is deliberate and, once
exercised, correct; what a forward-value check alone cannot catch is whether
the differentiable branch (`x.requires_grad == True`) actually works.

It doesn't, as shipped: both functions build `x_` via a bare `x.clone()`
with no `.detach()`, then unconditionally do `x_.requires_grad = True`. When
the caller's `x` already requires grad, `x.clone()` is a *non-leaf* tensor
(it already has `requires_grad=True`, inherited, and a `CloneBackward`
grad_fn) -- and PyTorch raises `RuntimeError: you can only change
requires_grad flags of leaf variables` the moment you touch that flag on a
non-leaf tensor, even though the value isn't changing. So the
`x.requires_grad == True` branch -- the one that matters for AD, since it's
exactly what a differentiable rollout needs (position gradients flowing
through an SDF-based boundary/region term) -- crashes outright today. No
caller in this repo currently passes a `requires_grad=True` x (see
caseUtils/weaklyCompressible.py's `sampleSDF`/`sampleDomainSDF` call sites),
so this has been a latent gap rather than a live failure, the same shape as
the `computeCompSPHBalanceTermWarp` segfault Phase 4.2 found: invisible
until forward-mode AD actually threads a tangent through this path.

Fixed in both files (geometry/sdf.py, regions/domainSDF.py) by guarding the
flag assignment: only set `requires_grad = True` when `x_` doesn't already
have it (i.e., only on the fresh-leaf branch). Verified separately that the
guarded version still correctly chains second-order gradients back to the
original tensor when it does require grad.

This script gradchecks the *fixed* functions: the full (d, grad) output
tuple against x, for both the requires_grad=True case (previously an
outright crash, now the differentiable path Phase 4 cares about) and a
sanity pass over the requires_grad=False case (must still return detached
tensors with no graph, unchanged from before).

Points are sampled away from the SDF's singular locus (the circle's center)
so the analytic gradient is smooth there -- gradcheck's finite-difference
comparison isn't a fair test at the one point where an SDF's gradient is a
genuine subgradient discontinuity, and that's a modeling fact about SDFs,
not something this file could fix.

    python scripts/gradcheck_sdf.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch

from _gradcheck_common import DEVICE, DTYPE
from warpSPHCore import DomainDescription

from warpSPH.geometry.sdf import getSDF, sampleSDF
from warpSPH.regions.domainSDF import sampleDomainSDF


def _circle_points(n: int = 5, radius: float = 3.0) -> torch.Tensor:
    """n points on a circle of the given radius, well away from the origin
    (the sdCircle singular locus) and from any box/domain corner."""
    theta = torch.linspace(0.3, 2.0, n, dtype=DTYPE, device=DEVICE)
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)


def check_crash_regression(label: str, fn) -> bool:
    """Regression guard for the fixed bug: x.requires_grad=True must not raise."""
    x = _circle_points().requires_grad_(True)
    print(f"\n=== {label}: requires_grad=True no longer raises ===")
    try:
        d, grad = fn(x)
        ok = d.requires_grad and grad.requires_grad
        print("PASSED" if ok else "FAILED (output did not require grad)")
        return ok
    except RuntimeError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def check_detached_when_not_required(label: str, fn) -> bool:
    """x.requires_grad=False must still return plain detached tensors."""
    x = _circle_points()
    print(f"\n=== {label}: requires_grad=False still returns detached output ===")
    d, grad = fn(x)
    ok = not d.requires_grad and not grad.requires_grad
    print("PASSED" if ok else "FAILED (output unexpectedly required grad)")
    return ok


def run_gradcheck(label: str, fn) -> bool:
    x = _circle_points().requires_grad_(True)
    print(f"\n=== {label}: torch.autograd.gradcheck (d, grad) against x ===")
    try:
        ok = torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    torch.manual_seed(0)

    # A hand-rolled circle SDF, isolating sampleSDF's own autograd/detach
    # logic from getSDF's torch.vmap wrapping.
    radius = 1.0

    def plain_circle_sdf(p: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(p, dim=-1) - radius

    def sample_plain_circle(x, invert=False):
        return sampleSDF(x, plain_circle_sdf, invert=invert)

    # The real production path: getSDF's torch.vmap-wrapped sdCircle.
    vmapped_circle = getSDF("circle")["function"]

    def sample_vmap_circle(x, invert=False):
        return sampleSDF(x, lambda p: vmapped_circle(p, radius), invert=invert)

    domain = DomainDescription(
        min=torch.tensor([-10.0, -10.0], dtype=DTYPE, device=DEVICE),
        max=torch.tensor([10.0, 10.0], dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False, False], device=DEVICE),
        dim=2,
    )

    def sample_domain(x, invert=False):
        return sampleDomainSDF(x, domain, invert=invert)

    targets = [
        ("sampleSDF(plain circle)", sample_plain_circle),
        ("sampleSDF(getSDF('circle'), vmap)", sample_vmap_circle),
        ("sampleDomainSDF", sample_domain),
    ]

    ok = True
    for label, fn in targets:
        for invert in (False, True):
            tag = f"{label}, invert={invert}"
            ok &= check_crash_regression(tag, lambda x, fn=fn, invert=invert: fn(x, invert=invert))
            ok &= check_detached_when_not_required(tag, lambda x, fn=fn, invert=invert: fn(x, invert=invert))
            ok &= run_gradcheck(tag, lambda x, fn=fn, invert=invert: fn(x, invert=invert))

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 0.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
