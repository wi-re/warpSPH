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

from warpSPH.configurations.gsph import GSPHConfig, InutsukaGSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import RiemannSolver, StateLimiter, VelocityPairPolicy
from warpSPH.modules.godunov import computeGaussianDensityWarp, computeGodunovWarp, computeInutsukaWarp

DIM = 1
N = 5


def _run(solver, order, stateLimiter=StateLimiter.PairRatio, shockC=0.0) -> bool:
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
    params.stateLimiter = stateLimiter.value
    params.shockSwitchC = shockC
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

    print(f"\n=== computeGodunovWarp [{solver.name}, order {order}, {stateLimiter.name}{', shock switch' if shockC > 0 else ''}]: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - canary script
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:700]}")
        return False


def _runInutsuka(solver, order, cubic) -> bool:
    """`computeInutsukaWarp` (Inutsuka 2002): the Gaussian density and its gradient are inputs; `V_ij^2`, `s*`, the states at `s*`, the Gaussian cutoffs."""
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports * 1.5, masses, domain, mode=SupportScheme.SuperSymmetric)
    densities = (torch.rand(N, dtype=DTYPE, device=DEVICE) * 0.5 + 0.75).requires_grad_(True)
    gradRho = (torch.randn(N, DIM, dtype=DTYPE, device=DEVICE) * 0.2).requires_grad_(True)
    velocities = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    pressures = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    internalEnergies = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5)
    params = InutsukaGSPHConfig().diffusionParams
    params.riemannSolver = solver.value
    params.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    inputs = (positions, supports, masses, densities, gradRho, velocities, pressures)
    if order == 2:
        J = (-(torch.rand(N, DIM, DIM, dtype=DTYPE, device=DEVICE) + 0.5)).requires_grad_(True)
        G = (torch.randn(N, 2, DIM, dtype=DTYPE, device=DEVICE) * 0.3).requires_grad_(True)
        inputs = inputs + (J, G)

    def f(pos, sup, mass, dens, gr, vel, press, *extra):
        state = make_compressible_state(pos, sup, mass, dens, vel, internalEnergies, pressures=press, soundspeeds=torch.ones_like(press), kinds=kinds)
        return computeInutsukaWarp(
            state, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.SuperSymmetric), domain=domain, params=params, gamma=1.4,
            gradRho=gr, order=order, cubic=cubic, adjacency=adjacency,
            queryVelocityTensor=extra[0] if order == 2 else None, queryStateGradients=extra[1] if order == 2 else None)

    print(f"\n=== computeInutsukaWarp [{solver.name}, order {order}, {'cubic' if cubic else 'linear'} V]: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - canary script
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:700]}")
        return False


def _runGaussianDensity(widthFactor) -> bool:
    """`computeGaussianDensityWarp`: `(grad rho, rho)` of the symmetrised Gaussian sum against positions, supports and masses."""
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports * 1.5, masses, domain, mode=SupportScheme.SuperSymmetric)

    def f(pos, sup, mass):
        state = make_compressible_state(pos, sup, mass, torch.ones_like(mass), torch.zeros(N, DIM, dtype=DTYPE, device=DEVICE),
                                        torch.ones_like(mass), pressures=torch.ones_like(mass), soundspeeds=torch.ones_like(mass), kinds=kinds)
        return computeGaussianDensityWarp(state, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.SuperSymmetric), domain=domain,
                                          adjacency=adjacency, widthFactor=widthFactor)

    print(f"\n=== computeGaussianDensityWarp [width {widthFactor:.3f} h]: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (positions, supports, masses), eps=1e-6, atol=1e-5)
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
    # L1b: the papers' 1D state limiters (kinks where D . delta changes sign: the random inputs keep clear of them) and the shock switch
    for lim in (StateLimiter.VanLeerHarmonic, StateLimiter.VanLeerMonotonized, StateLimiter.InutsukaSign):
        ok &= _run(RiemannSolver.Adaptive, 2, lim)
    ok &= _run(RiemannSolver.Adaptive, 2, StateLimiter.VanLeerHarmonic, 3.0)
    # layer 3
    for wf in (1.4142135623730951, 1.0):
        ok &= _runGaussianDensity(wf)
    for order in (1, 2):
        for cubic in (True, False):
            ok &= _runInutsuka(RiemannSolver.Adaptive, order, cubic)
    print("\nALL PASSED." if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
