#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/surfaceDetection/{wp_barecasco,wp_dilate,
wp_maronne}.py -- Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

Three modules, three different differentiability shapes:

`computeBarecascoSurfaceDetectionWarp` is a two-stage kernel. Its first output,
`outputNormals`, accumulates a per-particle "cover vector" `sum_j -n_ij` linearly
over a dynamic-length neighbor loop, inside `computeBarecascoSurfaceDetection_
Func_Adjacency_first` (a `@wp.func`, itself looping over `numOffsets` and calling a
second `@wp.func` for the per-offset neighbor sum), and only then does the enclosing
`@wp.kernel` apply `wp.normalize` to the *returned* value. Calling this script
surfaced a real, previously-latent bug -- but not, on closer inspection, the same
adjoint bug `gradcheck_incompressible.py` found and fixed for `computeAlphaWarp`:
`barecascoThreshold` (the second output's angle threshold) was passed to the kernel
launch as a bare Python `float`, which Warp infers as `float32` regardless of the
active precision, so *any* caller running at float64 (this script; also any real run
under `warpSPHBootstrap.bootstrap(precision='float64')`) crashed with a
kernel-argument dtype mismatch before ever reaching the differentiability question.
**Fixed** by wrapping it in `scalar_t(...)`, the same pattern every sibling module in
this file family already uses for its own scalar knobs (e.g.
`computeVelocityDiffusionDeltaSPH`'s `scalar_t(alpha)`).

With that fixed, an *initial* gradcheck attempt against `outputNormals` using this
script's usual `line_case` fixture (5 evenly-spaced particles) failed with an
all-zero analytical gradient -- at first glance the same shape as the
`computeAlphaWarp` bug. It is not: `computeAlphaWarp`'s squaring happened in the
*same* `@wp.func` scope as the loop that accumulated its operand (confirmed the
fix by moving it to the caller); `wp.normalize` here is *already* in a different
scope (the kernel) than the accumulation (the func), the same shape already
confirmed safe by that fix's own minimal repro extended to two levels of function
nesting. The real cause was the test case: `line_case`'s 5 particles are exactly
evenly spaced, so the middle particle's cover vector -- the symmetric sum of unit
vectors to two neighbors on each side -- passes extremely close to zero, and
`wp.normalize`'s own derivative genuinely blows up near a zero-norm input
(confirmed: the "failure" was a huge, sign-inconsistent finite-difference artifact
at exactly that particle, not the `computeAlphaWarp` bug's clean all-zero
analytical-vs-nonzero-numerical signature). Re-run below on an asymmetric case that
keeps every particle's cover vector comfortably away from zero -- clean.

Barecasco's *second* output (`outputValues`, the surface-indicator count from
an angle/norm threshold test) and `computeMaronneSurfaceDetection`'s output
(a geometric-condition neighbor count, thresholded at `< 0.5` in plain
PyTorch afterward) are both, by construction, discrete counts of boolean
conditions -- not `wp.psi.py`'s "no backward pass at all" shape, but the SPH
equivalent: their true derivative with respect to position is zero almost
everywhere and undefined exactly at a condition boundary, so a finite
difference probe that happens to cross one gives a spuriously huge
mismatch (confirmed: a plain gradcheck attempt against barecasco's second
output produces exactly this artifact, not a meaningful bug report). These
are **not in scope** for `torch.autograd.gradcheck`, the same call this
plan's Tier 2 already made for `modules/adaptiveSupport/wp_psi.py`; what *is*
verified below is that `computeMaronneSurfaceDetection`'s final output
(after its own `< 0.5` comparison) correctly reports `requires_grad=False`
-- comparison ops break the autograd graph in plain PyTorch by design, this
is not a Warp-side concern, and this check exists so a future refactor that
moved the threshold *before* returning from Warp (reintroducing exactly the
question this note is settling) would be caught.

`dilateSurfaceMaskWarp` differs from both: it sums `freeSurfaceMask[j]`
(a real-valued input array, not a boolean) gated by a kernel-support
connectivity test, with no post-loop nonlinearity -- linear-accumulation-only,
the shape every prior tier has confirmed safe. Gradchecked here against
`freeSurfaceMask` (the only continuously-meaningful differentiable input;
positions only gate discrete connectivity, same non-differentiable-by-design
adjacency contract `_gradcheck_common.py`'s own `build_adjacency` documents)
-- clean.

    python scripts/gradcheck_surfaceDetection.py
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

from warpSPH.modules.surfaceDetection.wp_barecasco import computeBarecascoSurfaceDetectionWarp
from warpSPH.modules.surfaceDetection.wp_dilate import dilateSurfaceMaskWarp
from warpSPH.modules.surfaceDetection.wp_maronne import computeMaronneSurfaceDetection

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    return domain, positions, supports, masses, densities, adjacency, kinds


def _build_barecasco_case():
    # An asymmetric line, unlike _build_case()'s line_case: every particle's cover
    # vector (the sum of unit vectors to its neighbors) is kept comfortably away
    # from zero, so wp.normalize's own derivative -- which genuinely blows up near a
    # zero-norm input -- never gets exercised near its singularity. line_case's
    # perfectly evenly-spaced particles put the middle one's cover vector right at
    # that singularity (two neighbors on each side cancel out almost exactly), which
    # is what made an earlier version of this script misdiagnose a finite-difference
    # artifact there as the computeAlphaWarp adjoint bug (see this file's docstring).
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[0.0], [0.4], [1.1], [2.3], [4.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((5,), 2.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((5,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    return domain, positions, supports, masses, densities, adjacency, kinds


def _run_barecasco_normals() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_barecasco_case()

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        normals, _vals = computeBarecascoSurfaceDetectionWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )
        return normals

    print("\n=== computeBarecascoSurfaceDetectionWarp [outputNormals]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_barecasco_not_in_scope() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities, kinds=kinds)
    _normals, vals = computeBarecascoSurfaceDetectionWarp(
        queryParticles=p,
        operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
        domain=domain,
        adjacency=adjacency,
    )
    print("\n=== computeBarecascoSurfaceDetectionWarp [outputValues]: not in scope ===")
    print(f"  discrete surface-indicator count, dtype={vals.dtype}, values={vals.detach().tolist()}")
    print("  (boolean-threshold count; see this script's docstring)")
    return True


def _run_maronne_not_in_scope() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    normals = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities, kinds=kinds)
    out = computeMaronneSurfaceDetection(
        queryParticles=p,
        operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
        domain=domain,
        surfaceNormals=normals,
        adjacency=adjacency,
    )
    print("\n=== computeMaronneSurfaceDetection: not in scope, verifying requires_grad=False ===")
    ok = out.requires_grad is False
    print(("PASSED" if ok else "FAILED") + f" (requires_grad={out.requires_grad})")
    return ok


def _run_dilate() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    mask = torch.rand(N, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(m):
        p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities, kinds=kinds)
        return dilateSurfaceMaskWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            freeSurfaceMask=m,
            adjacency=adjacency,
        )

    print("\n=== dilateSurfaceMaskWarp: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (mask,), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= _run_barecasco_normals()
    ok &= _run_barecasco_not_in_scope()
    ok &= _run_maronne_not_in_scope()
    ok &= _run_dilate()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
