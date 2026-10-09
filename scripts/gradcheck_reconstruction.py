#!/usr/bin/env python3
"""torch.autograd.gradcheck of the Monaghan viscosity kernels under a reconstructing `velocityPairPolicy`
(AV_PLAN Phase 3, `modules/reconstruction`).

`computeViscosityWarp` and `computeThermalDissipationWarp` with `Linear`, `Limited` and `BalsaraLimited` pair
velocities and with the Balsara pair limiter (AV_PLAN Phase 4): the reconstruction adds the velocity Jacobian and the
Balsara factor as inputs (read on both sides of the pair), the van Leer limiter's guarded divisions, the close-pair
taper (`limiters.py`, whose AD guards were moved there verbatim) and the guarded `Bbar^p`. Checked against every
differentiable input including the Jacobians and B. Same 1D line case as `gradcheck_dissipation.py`; the
`Raw` path is that script's.

    python scripts/gradcheck_reconstruction.py
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

from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    VelocityPairPolicy, buildDefaultDiffusionParamsCompressibleSPH)
from warpSPH.modules.dissipation.wp_diffusion import computeViscosityWarp
from warpSPH.modules.dissipation.wp_dissipation import computeThermalDissipationWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    velocities = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    internalEnergies = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    pressures = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    soundspeeds = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    alphas = (torch.rand(N, dtype=DTYPE, device=DEVICE) * 0.5 + 0.25).requires_grad_(True)
    # Jacobians of one sign with a spread, so the van Leer ratio sits inside (0, inf) away from its kinks
    jacobians = (-(torch.rand(N, DIM, DIM, dtype=DTYPE, device=DEVICE) + 0.5)).requires_grad_(True)
    balsara = (torch.rand(N, dtype=DTYPE, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)

    return (domain, positions, supports, masses, densities, adjacency, kinds,
            velocities, internalEnergies, pressures, soundspeeds, alphas, jacobians, balsara)


def _run(label, warp_fn, needs_energies, policy, pairLimiter=False) -> bool:
    (domain, positions, supports, masses, densities, adjacency, kinds,
     velocities, internalEnergies, pressures, soundspeeds, alphas, jacobians, balsara) = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCompressibleSPH()
    diffusionParams.C_q = 2.0
    diffusionParams.velocityPairPolicy = policy.value
    diffusionParams.reconstructionEtaCrit = 0.25
    diffusionParams.reconstructionEtaFold = 0.2
    diffusionParams.balsaraPairLimiter = pairLimiter

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha, J, B):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        kwargs = dict(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
            queryVelocityTensor=J,
            queryBalsara=B,
        )
        if needs_energies:
            kwargs["conductivityParams"] = diffusionParams
        else:
            kwargs["viscosityParams"] = diffusionParams
        return warp_fn(**kwargs)

    print(f"\n=== {label} [{policy.name}{' + Balsara pair limiter' if pairLimiter else ''}]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas, jacobians, balsara)
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
    for policy, pairLimiter in ((VelocityPairPolicy.Linear, False), (VelocityPairPolicy.Limited, False),
                                (VelocityPairPolicy.BalsaraLimited, False), (VelocityPairPolicy.Raw, True)):
        ok &= _run("computeViscosityWarp", computeViscosityWarp, False, policy, pairLimiter)
        ok &= _run("computeThermalDissipationWarp", computeThermalDissipationWarp, True, policy, pairLimiter)

    print()
    print("ALL PASSED." if ok else "FAILED -- see this script's docstring.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
