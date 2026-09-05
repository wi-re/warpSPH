#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/incompressible/{wp_alpha,wp_wallMoment}.py
-- Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeAlphaWarp` computes the DFSPH pressure-solver diagonal coefficient
`alpha_i = area_i/m_i * dot(sumA, sumA) + area_i * sumB`, where `sumA`/`sumB` are
accumulated over a per-particle, runtime-length neighbor loop
(`modules/incompressible/wp_alpha.py`'s `computeAlpha_Func_i_first`).

Getting a clean gradcheck out of this surfaced (and then fixed) a real, previously
undiagnosed member of the "reverse-mode AD through a loop silently produces a wrong
adjoint" bug family already documented for `correctGradientCRK` (Tier 1) and
`computeVsigWarp`'s `wp.max` (Tier 2): a **linear** accumulation over a dynamic
(runtime, per-particle) trip count -- the same "independently-computed-then-summed"
pattern Tiers 0-2 have repeatedly confirmed safe on its own -- followed by a
**nonlinear read of the accumulated variable, in the *same* `@wp.func`/`@wp.kernel`
scope as the loop that produced it**, silently zeroes that contribution's adjoint.
Confirmed with a from-scratch minimal Warp kernel with no SPH machinery at all: `s =
sum(x[j]*x[j] for j in range(n))` then `out = s*s`, both in one kernel body, with `n`
a *kernel argument* (dynamic trip count) -- all-zero analytical gradient against a
correct nonzero numerical one. The identical computation with the loop's own
accumulation isolated inside a separate `@wp.func` that *returns* the raw sum, and the
squaring done by the *caller* on that return value, differentiates correctly --
confirmed with the same minimal repro, including with an extra level of function
nesting matching this module's own inner-loop/outer-loop `Func_i`/`Func_Adjacency`
shape. Scope, not loop-nesting depth, is what matters: any number of loops is fine as
long as the last one's own accumulation finishes and *returns* before a nonlinear op
reads it, in a different function.

`computeAlpha_Func_i_first` originally did exactly the broken thing: accumulate
`sumA`/`sumB` over its own neighbor loop, then compute `alpha = areaI/mi *
wp.dot(sumA, sumA) + areaI*sumB` and `return alpha` -- all in the same function body
as the loop. **Fixed** by returning the raw `(sumA, sumB)` instead, and moving the
`wp.dot(sumA, sumA)` reduction into the caller (`computeAlpha_Func_Adjacency_first`),
computed there from the tuple `computeAlpha_Func_i_first` now returns, then
accumulated linearly into `out`. This is a pure textual relocation, not a
mathematical change: `sumA`/`sumB` are still freshly zeroed and fully accumulated
inside one call to `computeAlpha_Func_i_first` per outer offset either way, so the
per-offset value being squared is identical before and after -- only which function's
compiled scope contains the squaring changed, and that is what fixes the adjoint.

This gives a concrete, reusable recipe for any future module that hits this shape:
if a `@wp.func` accumulates a value over a loop and then reduces it nonlinearly before
returning, split it into two functions -- one that returns the raw accumulator, one
(or the caller) that applies the nonlinear reduction to the returned value.

`computeWallMomentWarp` (`wp_wallMoment.py`) is the kernel-weighted position
moment behind Eq. (61)'s wall pressure, `sum_j V_j rho_j (x_i - x_j) W_ij`. It is
a plain linear accumulation with no nonlinear reduction, so it is not in the
bug family above -- but its adjoint runs through `computeDistanceVec` and
`computeKernelCRK` with positions on *both* sides of the product (once in
`x_ij`, once inside `W_ij`), which is the shape most likely to lose a term, so
it is checked rather than assumed. Run with mixed kinds, since the operator is
directional (`FluidToBoundary`) and the directionality mask is a `continue`
inside the neighbour loop.

    python scripts/gradcheck_incompressible.py
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

from warpSPH.modules.incompressible.wp_alpha import computeAlphaWarp
from warpSPH.modules.incompressible.wp_wallMoment import computeWallMomentWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    areas = torch.rand(N, dtype=DTYPE, device=DEVICE).abs() + 0.5
    areas.requires_grad_(True)
    return domain, positions, supports, masses, densities, adjacency, kinds, areas


def _run(label: str, includeBoundaryReaction: bool = True, staticKinds: bool = False) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, areas = _build_case()

    if staticKinds:
        # Mark two of the five as static (`kind == 1`), so that with
        # `includeBoundaryReaction=False` a single launch takes *both* sides of
        # the `sumB` branch -- the accumulator is skipped for some neighbors of
        # a given `i` and taken for others. A branch that skips an accumulation
        # inside the neighbor loop is exactly the shape this file's docstring
        # is about, so it gets its own gradcheck rather than an argument that
        # it is obviously safe.
        kinds = kinds.clone()
        kinds[1] = 1
        kinds[3] = 1

    def f(pos, sup, mass, area):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        return computeAlphaWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryApparentAreas=area,
            adjacency=adjacency,
            includeBoundaryReaction=includeBoundaryReaction,
        )

    print(f"\n=== computeAlphaWarp ({label}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, areas)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _runWallMoment(label: str, direction) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, _ = _build_case()
    kinds = kinds.clone()
    kinds[1] = 1
    kinds[3] = 1

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens,
                          kinds=kinds)
        return computeWallMomentWarp(
            queryParticles=p,
            operationProperties=OperationProperties(
                kernel=KERNEL, supportMode=SupportScheme.Gather,
                operationMode=direction),
            domain=domain,
            adjacency=adjacency,
        )

    print(f"\n=== computeWallMomentWarp ({label}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities.detach().clone().requires_grad_(True))
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = _run("includeBoundaryReaction=True")
    ok &= _run("includeBoundaryReaction=False, mixed kinds",
               includeBoundaryReaction=False, staticKinds=True)
    from warpSPHCore.enumTypes import OperationDirection
    ok &= _runWallMoment("FluidToBoundary, mixed kinds",
                         OperationDirection.FluidToBoundary)
    ok &= _runWallMoment("AllToAll", OperationDirection.TrueAllToToAll)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
