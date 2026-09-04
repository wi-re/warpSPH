#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/shockCapturing/{wp_computeM,wp_vsig}.py
-- Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeMWarp` computes the Cullen-Dehnen-style renormalization matrix
`sum_j m_j outer(x_ij, gradw_ij)` (matrix-valued output, no scheme-specific
scalar knobs beyond positions/supports/masses/densities) -- gradchecks clean.

`computeVsigWarp` computes the signal velocity `max_j(c_bar - mu_ij)` used by
shock-capturing switches. Getting a clean test case for it surfaced, and then
fixed, three real bugs -- the last one, previously written up as "confirmed
but not fixable here, upstream-only", turned out to have a real local fix:

1. **Fixed.** `computeVsigWarp` had `referenceVelocities = queryVelocities`
   as a fallback when the caller doesn't supply `referenceParticles`, but no
   equivalent `referenceCs = queryCs` fallback -- so a caller that passes
   `queryCs=` explicitly (exactly what a bare-`ParticleState` gradcheck case
   does, and what every real scheme call site does too, just via a
   `.soundspeeds` attribute that happens to be on the same object) fell
   through to a **size-1 dummy tensor read out of bounds** the moment
   `referenceParticles` (defaulting to `queryParticles`) lacked a
   `.soundspeeds` attribute -- silently reading uninitialized/stray memory,
   which is why this script's first attempts here gave different, wrong
   forward *values* on every run. Same missing-fallback shape as Phase 4.2's
   `computeCompSPHBalanceTermWarp` finding, and it turned out **not to be
   unique to vsig**: the identical gap (`referenceCs`/`referenceAlphas`/
   `referencePressures` never defaulted to their `query*` counterparts) was
   present in all 7 files that share this parameter family --
   `modules/{compSPH,crk}/{accel,dudt}.py` and
   `modules/dissipation/{wp_conductivity,wp_diffusion,wp_dissipation}.py`,
   all previously gradchecked "clean" in Tiers 0/1 only because those scripts
   always populate `.soundspeeds`/`.alphas`/`.pressures` on the *same* state
   object passed as both query and reference. Fixed identically in all 8
   files (this one plus the 7). A separate, unrelated `torch.scalar_t32`
   typo (not a real torch dtype -- should have been `get_torch_precision()`)
   in the same dummy-tensor fallback line was fixed alongside it, present in
   the same 8 files.
2. **Fixed.** With the memory bug out of the way, `computeVsigWarp`'s backward
   pass was *deterministically wrong* for 2 of 3 particles in a hand-picked,
   tie-free case. Root cause: `out = wp.max(out, vsigs)` was a **loop-carried
   variable reassigned via a nonlinear op inside the neighbor `for` loop** --
   the same underlying bug *class* Tier 1 found in `correctGradientCRK`'s
   hand-written accumulation loop, but a different concrete shape: here the
   adjoint sometimes attributed the gradient to the *wrong* neighbor or
   dropped it to zero, even though the forward value was correct. This was
   initially written up as needing an upstream `warp-lang` fix, since
   swapping `wp.max` for the logically-equivalent `wp.where` made no
   difference. **A real local fix was found**, once `computeAlphaWarp`'s
   related (but not identical) finding showed that *scope*, not the presence
   of a loop, is what matters (see `gradcheck_incompressible.py`'s docstring):
   `computeVsig_Func_i` (the old, single, in-loop-max function) was split into
   `computeVsig_Func_i_argmax` -- a forward-only pass that finds *which*
   neighbor achieves the max and its index, via the same loop-carried
   `wp.max` pattern as before, but whose only outputs consumed by the caller
   are a `wp.bool`/`wp.int32` pair (indices are never differentiated by Warp,
   so a wrong adjoint on this function's own unused return value is
   harmless) -- and `computeVsig_valueAt`, which recomputes the actual vsig
   value for that *one already-known* index, with no loop at all, so it
   differentiates via Warp's ordinary automatic diff exactly like every other
   per-neighbor SPH formula in this codebase. The caller accumulates that
   single recomputed value linearly across offsets, the same safe
   "accumulate-in-callee, reduce-in-caller" shape `computeAlphaWarp`'s fix
   established. **This is a general recipe**: a loop-carried nonlinear
   *reduction* (as opposed to a post-loop nonlinear *read* of an
   otherwise-linear accumulator) can be worked around by separating "which
   candidate wins" (forward-only, discrete, never needs a correct gradient)
   from "what is that candidate's value" (recomputed once, ordinary code).
3. **Found and fixed, separately, while chasing (2).** `cs_i =
   queryCs[i] if individual_cs else scalar_t(1.0)` and the equivalent
   `cs_j` line are a same-array-vs-default ternary on a `wp.bool` *kernel
   argument* condition -- exactly the shape the `access_optional` fix (added
   to warpSPHCore during the Phase 4.1 "Regression" writeup) was built for,
   but `wp_vsig.py` was never one of the 8 files that fix's original sweep
   touched. It silently zeroed `d(vsig)/d(cs)` entirely (confirmed: forward
   value visibly depends on `cs_i`/`cs_j` via `c_bar`, but the analytical
   gradient was uniformly zero against a numerically nonzero ground truth) --
   invisible before now because this script's gradcheck was already failing
   for reason (2), so this second, independent bug was masked underneath it.
   Fixed by routing both through `access_optional`, matching every other
   file in this family.

With all three fixed, `computeVsigWarp` gradchecks clean, both with
`individual_cs` on and off. `_run_vsig` below now runs a genuine gradcheck.

**A fourth, separate, pre-existing correctness bug was found and fixed while
verifying (2)'s restructuring against grid traversal** (the `useAdjacency=False`,
`checkOffset`-based path -- distinct from the `AdjacencyList` path every other
gradcheck script in this family exercises, and the one every real call site of
`computeVsigWarp` currently uses). `computeVsig_Func_i_argmax`'s per-neighbor loop
had **no compact-support filter at all** -- unlike every sibling module in this
file family (`wp_dilate.py`, `wp_sum.py`, ...), which all gate their neighbor
contribution on `w_ij > 0` or an equivalent radius check, every entry `checkOffset`
returned was treated as a genuine neighbor unconditionally. This was invisible for
the `AdjacencyList` path (`radiusSearchCompactHashMap` already returns an exact,
pre-filtered neighbor list, so the missing filter was a no-op there) but wrong for
grid traversal, where `checkOffset` returns every particle in a nearby *cell* --
coarser than the exact kernel support radius. Confirmed directly: on the same
particle configuration, `AdjacencyList` and grid-traversal `computeVsigWarp` results
disagreed (by as much as the largest candidate's full value, for particles whose true
winner is the compact-support-respecting self-term) before this fix, and matched
exactly, and matched a from-scratch brute-force reference, after adding a
`computePairwiseSupport`-based radius check. `_run_grid_consistency` below is the
regression guard for this -- not a gradcheck (no gradients involved), a forward-value
agreement check between the two traversal modes on the same particles.

    python scripts/gradcheck_shockCapturing.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, build_grid_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.shockCapturing.wp_computeM import computeMWarp
from warpSPH.modules.shockCapturing.wp_vsig import computeVsigWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    return domain, positions, supports, masses, densities, adjacency, kinds


def _run_computeM() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeMWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )

    print("\n=== computeMWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_vsig() -> bool:
    # Hand-picked, tie-free case: asymmetric positions and a strong outward
    # velocity spread so each particle's true argmax neighbor has a
    # comfortable margin (>=1.0, vs. eps=1e-6) over every other candidate,
    # including the self-term (mu_ij == 0 identically there) -- rules out
    # near-tie/branch-switch artifacts as an explanation for any mismatch.
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-0.5], [0.0], [0.3]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((3,), 1.5, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocities = torch.tensor([[2.5], [0.0], [-1.5]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    soundspeeds = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    def f(pos, sup, mass, dens, vel, cs):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeVsigWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=vel,
            queryCs=cs,
            adjacency=adjacency,
        )

    print("\n=== computeVsigWarp [individual_cs=True]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, soundspeeds)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_vsig_no_individual_cs() -> bool:
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-0.5], [0.0], [0.3]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((3,), 1.5, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocities = torch.tensor([[2.5], [0.0], [-1.5]], dtype=DTYPE, device=DEVICE, requires_grad=True)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    def f(pos, sup, mass, dens, vel):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeVsigWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=vel,
            adjacency=adjacency,
        )

    print("\n=== computeVsigWarp [individual_cs=False]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_grid_consistency() -> bool:
    # Forward-value-only: AdjacencyList and grid (CompactHashMap) traversal must
    # agree on the same particle set, since they're two implementations of the
    # same neighbor search. No gradients involved -- this guards the missing
    # compact-support-filter bug described in this file's module docstring
    # (finding 4), which only manifests in grid mode.
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
        return computeVsigWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=velocities,
            adjacency=adj,
        )

    print("\n=== computeVsigWarp: AdjacencyList vs. grid traversal forward-value consistency ===")
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
    ok &= _run_computeM()
    ok &= _run_vsig()
    ok &= _run_vsig_no_individual_cs()
    ok &= _run_grid_consistency()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
