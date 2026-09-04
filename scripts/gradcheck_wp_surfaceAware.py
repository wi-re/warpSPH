#!/usr/bin/env python3
"""torch.autograd.gradcheck against computePressureSurfaceAwareWarp -- Tier 0
of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

modules/pressure/wp_surfaceAware.py backs deltaSPH's and Monaghan's pressure
force (computePressureForceSurfaceAware) and was flagged, not measured, by
the 4.1 inventory: it contains the exact "ternary assigned to a local
variable, where both branches index the same Warp array" shape that silently
zeroed an adjoint in warpSPHCore's Interpolate operator upstream (fixed in
warp-lang 1.17.0.dev3, not yet in the 1.12.0 this repo is pinned to). Two
instances, both in computePressureSurfaceAware_Func_i:

    P_j = referencePressures[j] if referencePressures.shape[0] > 1 else referencePressures[0]
    mask_j = referenceSurfaceMask[j] if referenceSurfaceMask.shape[0] > 1 else 0

and the query-side twin in computePressureSurfaceAware_Func_Adjacency:

    P_i = queryPressures[i] if queryPressures.shape[0] > 1 else queryPressures[0]

The dangerous branch is the broadcast one (`shape[0] == 1`, reading index 0
for every particle) -- if the ternary zeros the adjoint the way it did for
Interpolate, d(output)/d(pressures) would come back incorrectly zero instead
of the sum of every neighbor pair's contribution. This script runs the
per-particle-pressure case (safe branch, `shape[0] > 1`) and the broadcast
case (suspect branch, `shape[0] == 1`) side by side, across every
PressureForceScheme value, so a difference in outcome between the two would
be diagnostic on its own even without knowing the bug shape in advance.

Differentiable inputs checked: positions, supports, masses, densities,
queryPressures (and, since referencePressures defaults to queryPressures
when not passed, the same tensor doubles as the reference-side check).

    python scripts/gradcheck_wp_surfaceAware.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import OperationDirection, SupportScheme

from warpSPH import PressureForceScheme
from warpSPH.modules.pressure.wp_surfaceAware import computePressureSurfaceAwareWarp


def run_gradcheck(pressure_term: PressureForceScheme, pressure_shape: str) -> bool:
    domain = make_domain()
    positions, supports, masses = line_case(5)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.SuperSymmetric)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.SuperSymmetric)

    n = positions.shape[0]
    if pressure_shape == "per-particle":
        pressures = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    else:  # "broadcast" -- exercises the `shape[0] == 1` ternary branch
        pressures = torch.randn(1, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, press):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computePressureSurfaceAwareWarp(
            p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.SuperSymmetric, operationMode=OperationDirection.AllToAll),
            domain=domain,
            adjacency=adjacency,
            queryPressures=press,
            pressureTerm=pressure_term,
        )

    print(f"\n=== {pressure_term.name} ({pressure_shape} pressures): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, pressures)
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

    ok = True
    for pressure_term in PressureForceScheme:
        for pressure_shape in ("per-particle", "broadcast"):
            ok &= run_gradcheck(pressure_term, pressure_shape)

    print()
    if ok:
        print("ALL PASSED -- including the broadcast-pressure ternary branch; no adjoint-zeroing found.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 0.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
