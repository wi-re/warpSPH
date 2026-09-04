#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/mdbc/wp_nopenshift.py -- Tier 2 of
docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeMdbcNoPenShiftWarp` computes the mDBC no-penetration velocity
correction: for each fluid query particle, it walks its boundary neighbors
and, where a hand-picked geometric test (`condition_a`/`condition_b`) and a
closing-velocity test (`vfc < 0`) both fire, accumulates a per-dimension
correction term built from the fluid/boundary relative velocity and the
boundary's ghost-offset-derived normal, then averages by the neighbor count
that actually contributed. Getting a clean run out of this module surfaced
three real, previously-latent bugs -- none in the differentiability logic
this script set out to check, all in getting the kernel to run under
gradient tracking *at all*, which nothing in this repo had ever done before
for a kernel on this exact code path:

1. **Fixed upstream, in warpSPHCore.** `zero_like_warp(outCtr)`, called on
   the kernel's int32-vector output-counter array, crashed the Warp compiler
   itself (`AttributeError: 'Var' object has no attribute 'is_builtin'`).
   Root cause: `warpSPHCore/math/wp_zero.py`'s `zero_like` overload for a
   length-1 `wp.array(dtype=vector(..., dtype=wp.int32))` built its zero
   value via the generic call form `vector(length=1, dtype=wp.int32)(0)` --
   every analogous float16/32/64 overload instead returns a *concrete*
   pre-declared class (`vec1f(0.0)`, `vec1h(0.0)`, `vec1d(0.0)`, from
   `wp_vec1.py`), but no `vec1i` class existed, so the int32 length-1 case
   was the one path still using the generic-instantiation form, and that
   form doesn't codegen on the warp-lang version now installed here
   (1.15.0; see this file's warp-lang-version note below). Fixed by adding
   `vec1i` to `wp_vec1.py` (mirroring `vec1f`/`vec1h`/`vec1d` exactly) and
   pointing the zero_like overload at it, matching every other dtype's
   pattern instead of being the one exception.
2. **Fixed here.** `norm_j = safe_sqrt(...) + 1e-12` -- a bare Python float
   literal added to a `scalar_t` value. Warp infers a bare float literal as
   `wp.float32` regardless of the active precision, so under this repo's
   `warpSPHCore_PRECISION=float64` (required for gradcheck's numerical
   Jacobian) this is a genuine `float64 + float32` type mismatch, caught
   only because this is the first time this file has ever been exercised at
   float64 -- every other `+ eps` site in this same file already goes
   through `scalar_t(...)` (see line ~416), this was the one straggler.
   Fixed by wrapping it the same way.
3. **Fixed upstream, in warpSPHCore -- the significant one.**
   `warpSPHCore/autograd/launcher.py`'s `launch_kernel` unconditionally sets
   `output.requires_grad = requires_grad` on *every* output of a multi-output
   kernel once any input requires grad -- with no check that the output's own
   dtype can legally carry a gradient. This kernel is a multi-output launch
   (`(nopenshift, nopenCounter)`, one float array and one int32-vector
   array), and `wp.to_torch` on the int32 output then raised `RuntimeError:
   only Tensors of floating point and complex dtype can require gradients`.
   Every gradcheck script before this one only ever exercised
   single-output-float kernels, so this is the first time a mixed-dtype
   multi-output launch was run under `requires_grad` in this repo at all --
   a structural gap in the bridge itself, same shape as the plan's other
   cross-repo AD-bridge findings (`asScalarArg`, the CRK accumulation-axis
   bug). Fixed by gating `requires_grad` on the output's own dtype
   (`_dtype_is_float`, checking `_wp_scalar_type_` for vector/matrix dtypes)
   -- purely additive: every existing single-float-output kernel keeps
   getting `requires_grad=True` exactly as before, only a non-float output
   now correctly stays `requires_grad=False`.

With all three fixed, the kernel finally *ran* under `gradcheck` -- and
found one more real hazard, this time worked around in this script's test
fixture rather than in source: `computeMdbcNoPenShift_Func_i` computes
`apparentVolume = mj / rhoj` unconditionally for every neighbor (it's dead
for this call -- `useVolume` is always False here, so the value is never
read downstream), using densities read off `referenceState`. This script's
first case passed `densities=None`, which a bare `ParticleState` defaults to
a zero-filled array, so `rhoj = 0` and `apparentVolume = mj / 0 = inf` --
forward-dead, but reverse-mode AD still built an adjoint through the
division, and that `inf`/`nan` propagated back into every mass gradient for
the early-returned boundary-particle threads (a NaN Jacobian entry, not
caught by any forward-value check). Exactly the same bug *class* Tier 1
found in `crk/limiter.py`'s self-interaction 0/0 -- a computed-but-discarded
expression whose singularity still poisons the backward pass -- in a
different guise (division by an absent field rather than a same-array
ternary or a loop-carried max). Not fixed in source here, since
`apparentVolume` being computed unconditionally is a pattern shared by
every sibling module in this family and changing it is out of this script's
scope; worked around by giving this case real, nonzero densities (via
`compute_densities`, this family's usual real-Density-op fixture) instead
of `None`. Left as a note for whoever eventually revisits `useVolume`
plumbing in this file: a caller that (legitimately, per every other
gradcheck script's own `hasattr`-fallback precedent) omits densities on a
bare `ParticleState` will hit this same NaN the moment gradients are
requested through this path.

Once all four issues were addressed, `computeMdbcNoPenShiftWarp` gradchecks
clean. The masks (`condition_a`/`b`/`c`, `mask`, `vfc < 0`) are plain
`if`/`else` selection between two float values, not loop-carried
reassignment (contrast Tier 2's shockCapturing `wp.max` finding), and every
neighbor's contribution is independently computed then summed
(`out += tempOut`) -- the same linear-accumulation shape Tiers 0/1 already
confirmed is safe. Only positions/velocities/ghost-offsets feed the
differentiable output; supports/masses only gate which branch fires
(`w_ij > 0` via the kernel, `dp_i` via mass) and are included anyway for
completeness.

The hand-picked case below places one fluid query particle between two
boundary neighbors, chosen (like gradcheck_shockCapturing.py's vsig case)
with enough margin from every branch threshold (`1.25*dp_i`, `0.75*norm_j`,
`1.75*dp_i`, `vfc < 0`) that finite-difference perturbation in gradcheck's
numerical Jacobian cannot flip a branch -- one boundary neighbor is placed
to land in the "closing" (`vfc < 0`, correction applied) branch and the
other in the "opening" (`vfc > 0`, correction suppressed) branch, so both
paths are exercised in the same case.

    python scripts/gradcheck_mdbc.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import OperationDirection, SupportScheme

from warpSPH.modules.mdbc.wp_nopenshift import computeMdbcNoPenShiftWarp

DIM = 1


def _build_case():
    domain = make_domain(dim=DIM)
    # particle 0: fluid query. particles 1, 2: boundary neighbors, one on
    # each side, with ghost offsets sized so condition_b has clear margin.
    positions = torch.tensor([[0.0], [-0.3], [0.4]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocities = torch.tensor([[1.0], [0.0], [0.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    # ghost offsets only meaningful on the boundary rows; the fluid row's is
    # unused (never read for i, only referenceGhostOffsets[j]) but supplied
    # for shape uniformity.
    ghostOffsets = torch.tensor([[0.0], [-0.6], [0.6]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    kinds = torch.tensor([0, 1, 1], dtype=torch.int32, device=DEVICE)

    # Densities must be real (nonzero), not None -- see this script's
    # docstring: an unpopulated ParticleState.densities defaults to a
    # zero-filled array, and the dead `apparentVolume = mj / rhoj` local one
    # frame down divides by that zero. The NaN this produces is never read
    # (apparentVolume feeds no output here), but poisons the adjoint anyway.
    adjacency, _kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather).detach()
    return domain, positions, supports, masses, velocities, ghostOffsets, densities, kinds, adjacency


def _run() -> bool:
    domain, positions, supports, masses, velocities, ghostOffsets, densities, kinds, adjacency = _build_case()

    def f(pos, sup, mass, vel, offsets):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        out, _ctr = computeMdbcNoPenShiftWarp(
            queryParticles=p,
            operationProperties=OperationProperties(
                kernel=KERNEL,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.BoundaryToFluid,
            ),
            domain=domain,
            adjacency=adjacency,
            queryVelocities=vel,
            queryOffsets=offsets,
        )
        return out

    print("\n=== computeMdbcNoPenShiftWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, velocities, ghostOffsets)
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
