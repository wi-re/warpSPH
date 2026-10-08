#!/usr/bin/env python3
"""torch.autograd.gradcheck of the simplified Godunov SPH pair rates (GODUNOV_SPH_PLAN layer 2, `modules/godunov`).

`computeGodunovWarp` -- `(a_i, du_i/dt)` -- first order (the particles' own states) and second order (the limited velocity Jacobian and the
`(rho, P)` gradients are inputs too), for the acoustic, adaptive and HLLC solvers. Same 1D line case as `gradcheck_dissipation.py`; the
second-order case breaks the line's mirror symmetry (the reconstructed state is clamped between the two particle values, a kink at
`s_i == s_j`).

    python scripts/gradcheck_godunov.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, line_case, make_compressible_state, make_domain
from warpSPHCore import OperationProperties
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.configurations.gsph import GSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import RiemannSolver, VelocityPairPolicy
from warpSPH.modules.godunov import computeGodunovWarp

DIM = 1
N = 5


def _run(solver, order) -> bool:
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    if order == 2:
        densities = (densities.detach() * (1.0 + 0.2 * torch.rand(N, dtype=DTYPE, device=DEVICE))).requires_grad_(True)
    velocities = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    internalEnergies = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    pressures = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    soundspeeds = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    params = GSPHConfig().diffusionParams
    params.riemannSolver = solver.value
    params.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    params.reconstructionEtaCrit = 0.25
    params.reconstructionEtaFold = 0.2
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds)
    if order == 2:
        J = (-(torch.rand(N, DIM, DIM, dtype=DTYPE, device=DEVICE) + 0.5)).requires_grad_(True)
        G = (torch.randn(N, 2, DIM, dtype=DTYPE, device=DEVICE) * 0.3).requires_grad_(True)
        inputs = inputs + (J, G)

    def f(pos, sup, mass, dens, vel, u, press, cs, *extra):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, kinds=kinds)
        return computeGodunovWarp(
            state, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather), domain=domain, params=params, gamma=1.4,
            order=order, adjacency=adjacency,
            queryVelocityTensor=extra[0] if order == 2 else None, queryStateGradients=extra[1] if order == 2 else None)

    print(f"\n=== computeGodunovWarp [{solver.name}, order {order}]: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - canary script
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:700]}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)
    ok = True
    for solver in (RiemannSolver.Acoustic, RiemannSolver.Adaptive, RiemannSolver.HLLC):
        for order in (1, 2):
            ok &= _run(solver, order)
    print("\nALL PASSED." if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
