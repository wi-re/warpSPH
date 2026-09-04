#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/adaptiveSupport/{wp_omega,wp_psi0}.py
-- Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeOmegaWarp` computes the grad-h correction factor
(1 - sum_j m_j dW/dh, used to correct for the h-dependence of the kernel) and
`computePsi0Warp` computes the two `psi0`-family reference spacing estimates
Owen-style adaptive support uses to pick a target `h`. Both share the module
family's usual same-array-safe apparent-volume ternary and take only positions/
supports/masses/densities -- no scheme-specific scalar knobs.

`computePsi0Warp` returns `psi0**(1/dim)`/`psi0h**(1/dim)`: a fractional power
with an unbounded derivative at 0, but the self-interaction pair (`i==j`,
included in every adjacency list per Tier 1's own finding) contributes a
strictly positive kernel value at `r=0`, so `psi0` is bounded away from zero
for any of this script's cases and the singularity is never exercised.

    python scripts/gradcheck_adaptiveSupport.py
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

from warpSPH.modules.adaptiveSupport.wp_omega import computeOmegaWarp
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    return domain, positions, supports, masses, densities, adjacency, kinds


def _run(label, warp_fn) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warp_fn(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )

    print(f"\n=== {label}: torch.autograd.gradcheck ===")
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

    ok = True
    ok &= _run("computeOmegaWarp", computeOmegaWarp)
    ok &= _run("computePsi0Warp", computePsi0Warp)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
