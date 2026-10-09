#!/usr/bin/env python3
"""torch.autograd.gradcheck against the four partial-sum modules the analytic-boundary detector and shifting are built from
(`modules/surfaceDetection/wp_barecascoCover.py`, `wp_barecascoCone.py`, `modules/shifting/wp_deltaShiftRaw.py`). They are warpSPH's
`computeBarecascoSurfaceDetectionWarp` / `computeDeltaShiftWarp` cut open so that the wall's share of a sum can be added BEFORE the decision is taken (see
`modules/analyticBoundary/detector.py`), so each has the differentiability shape of the piece it was cut from:

`computeBarecascoCoverWarp`   the cover vector `sum_j x_ij / |x_ij|` of `computeBarecascoSurfaceDetection`'s first stage, WITHOUT the
                              `wp.normalize` that kernel applies afterwards: accumulate-only, the safe shape (`gradcheck_surfaceDetection.py`).
                              Gradchecked against positions on an asymmetric 2D set (a symmetric one puts a cover vector at zero, where
                              the sum of unit vectors is not differentiable) -- see `_build_case`.
`computeDeltaShiftRawWarp`    `sum_j m_j / (2 (rho_i + rho_j)) [1 + R (W_ij / W0)^n] grad_i W_ij`, i.e. `computeDeltaShiftWarp`'s sum
                              without the CFL / Mach scaling and the cap (`gradcheck_deltaShift.py`): accumulate-only. Against positions,
                              supports, masses and densities.
`computeBarecascoConeCountWarp`  the wedge count: a sum of indicator functions (angle test), derivative zero almost everywhere and undefined on
                              the wedge boundary, so a finite-difference probe that crosses one produces a spurious mismatch -- NOT in scope for
                              gradcheck, the same call `gradcheck_surfaceDetection.py` makes for barecasco's second output. What is checked
                              instead is the exact value on a case with a known count (`_run_cone_count_values`).

A fourth cut, `min_j n_i . n_j` for the curvature gate of the shifting, was dropped from the series: `solveShifting` keeps `_curvatureGate` (torch, over
the Verlet list), and the kernel's `out = wp.min(out, ...)` inside the neighbour loop is the loop-carried nonlinear reduction `gradcheck_shockCapturing.py`
documents (the adjoint differentiated against the FIRST neighbour, not the minimising one, in this script's case; the forward value was exact).

    python scripts/gradcheck_analyticBoundary.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import math
import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.shifting.wp_deltaShiftRaw import computeDeltaShiftRawWarp
from warpSPH.modules.surfaceDetection.wp_barecascoCone import computeBarecascoConeCountWarp
from warpSPH.modules.surfaceDetection.wp_barecascoCover import computeBarecascoCoverWarp

DIM = 2
H = 1.0
XY = [[0.0, 0.0], [0.4, 0.1], [0.9, -0.2], [0.3, 0.8], [1.25, 0.7], [-0.5, 0.5], [0.7, 1.4]]          # asymmetric, every particle has neighbours


def _build_case():
    domain = make_domain(dim=DIM)
    positions = torch.tensor(XY, dtype=DTYPE, device=DEVICE, requires_grad=True)
    n = positions.shape[0]
    supports = torch.full((n,), H, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    r = torch.cdist(positions.detach(), positions.detach())
    gap = float((r - H).abs()[r > 0].min())
    assert gap > 1e-3, f"a pair sits {gap:.3g} from the support edge: the gate would be crossed by the finite-difference probe"
    return domain, positions, supports, masses, densities, adjacency, kinds


def _check(name, f, inputs) -> bool:
    print(f"\n=== {name}: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _props():
    return OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather)


def _run_cover() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()

    def f(pos, sup):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=densities.detach(), kinds=kinds)
        return computeBarecascoCoverWarp(queryParticles=p, operationProperties=_props(), domain=domain, adjacency=adjacency)

    return _check("computeBarecascoCoverWarp", f, (positions, supports))


def _run_delta_shift_raw(R, label) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeDeltaShiftRawWarp(queryParticles=p, operationProperties=_props(), domain=domain, R=R, n=4, W0=2.0, adjacency=adjacency)

    return _check(f"computeDeltaShiftRawWarp [{label}]", f, (positions, supports, masses, densities))


def _run_cone_count_values() -> bool:
    """The wedge count has no meaningful derivative (see the docstring); its value must be right: with the cover axis along +x and a half angle
    of 45 degrees, a particle counts exactly the neighbours inside the wedge."""
    print("\n=== computeBarecascoConeCountWarp: value check (no gradcheck: discrete count) ===")
    domain = make_domain(dim=DIM)
    xy = [[0.0, 0.0], [0.5, 0.1], [0.5, 0.45], [0.5, 0.6], [-0.5, 0.0]]        # from particle 0: (0.5, 0.1) in the wedge, (0.5, 0.45) in, (0.5, 0.6) out (50 deg), (-0.5, 0) out
    positions = torch.tensor(xy, dtype=DTYPE, device=DEVICE)
    n = positions.shape[0]
    supports = torch.full((n,), H, dtype=DTYPE, device=DEVICE)
    masses = torch.ones(n, dtype=DTYPE, device=DEVICE)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=torch.ones(n, dtype=DTYPE, device=DEVICE), kinds=kinds)
    axes = torch.zeros(n, DIM, dtype=DTYPE, device=DEVICE)
    axes[:, 0] = 1.0
    got = computeBarecascoConeCountWarp(queryParticles=p, operationProperties=_props(), domain=domain, coverAxes=axes, halfAngle=math.pi / 4, adjacency=adjacency).reshape(-1)
    r = torch.cdist(positions, positions)
    d = positions[None, :, :] - positions[:, None, :]                                               # d[i, j] = x_j - x_i
    cosang = d[..., 0] / r.clamp(min=1e-30)
    inside = (r > 0) & (r <= H) & (cosang >= math.cos(math.pi / 4))
    ref = inside.sum(1).to(DTYPE)
    ok = bool(torch.allclose(got.to(DTYPE), ref))                                                   # neighbours j != i within the support whose direction x_j - x_i is within the half angle of the axis
    print(f"kernel {got.tolist()}  brute force {ref.tolist()}")
    print("PASSED" if ok else "FAILED")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    ok = _run_cover()
    ok &= _run_delta_shift_raw(0.0, "R = 0")
    ok &= _run_delta_shift_raw(0.2, "R = 0.2, n = 4")
    ok &= _run_cone_count_values()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
