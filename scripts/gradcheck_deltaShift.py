#!/usr/bin/env python3
"""torch.autograd.gradcheck against sample/wp_deltaShift.py -- Tier 2 of
docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeDeltaShiftWarp` computes the Fourtakas-style particle-shifting
displacement `sum_j (1 + R*k^n) * (m_j/(rho_i+rho_j)/2) * gradw_ij`, used to
regularize particle distributions between steps. Every term is computed
per-neighbor and summed linearly (`out += shiftAmount`) with no post-loop
nonlinear reduction of the accumulated vector -- the safe shape confirmed by
`warpSum`/`dilateSurfaceMaskWarp` (`gradcheck_util.py`,
`gradcheck_surfaceDetection.py`), not the "loop-accumulate then square/
normalize" shape that broke `computeAlphaWarp` and barecasco's normals.
Shares the family's usual different-array apparent-volume ternary
(`apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]`,
confirmed non-issue since Tier 0) and a `wp.pow(k, n)` term with an
integer-valued exponent (`n=4` here), which stays well away from `k=0`
(`k = w_ij/W_0`, and `w_ij>0` is the same loop's own gate) for this script's
regular line case. Gradchecked against positions/supports/masses/densities
-- clean.

`_run_volume_weighted` covers the `volumeWeighted=True` addition
(PST_ALE_PLAN.md Stage A): selects the plain apparent volume
(`referenceVolumes[j]`, already computed in this kernel and previously
discarded) as the per-neighbor weight instead of the mean-density term,
matching Michel et al. 2022 Eq. (2)-(3)'s weighting exactly. Same
accumulate-only shape, and `apparentVolume` is already the AD-safe path ~20
other kernels in this codebase use (PST_ALE_PLAN.md Part 4.3), so no new risk
-- included as a regression guard on the added branch, not because the shape
changed.

    python scripts/gradcheck_deltaShift.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import KERNEL, build_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    dx = (positions[1, 0] - positions[0, 0]).item()
    return domain, positions, supports, masses, densities, adjacency, kinds, dx


def _run() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, dx = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeDeltaShiftWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            CFL=0.4,
            computeMach=False,
            c_max=1.0,
            rho0=1.0,
            dx=dx,
            adjacency=adjacency,
        )

    print("\n=== computeDeltaShiftWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_volume_weighted() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, dx = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeDeltaShiftWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            CFL=0.4,
            computeMach=False,
            c_max=1.0,
            rho0=1.0,
            dx=dx,
            R=0.2,
            n=4,
            volumeWeighted=True,
            adjacency=adjacency,
        )

    print("\n=== computeDeltaShiftWarp [volumeWeighted=True]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities)
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
    ok &= _run_volume_weighted()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
