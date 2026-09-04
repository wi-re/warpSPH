#!/usr/bin/env python3
"""torch.autograd.gradcheck against the shared dissipation kernels -- Tier 1
of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeViscosityWarp` (modules/dissipation/wp_diffusion.py),
`computeConductivityWarp` (modules/dissipation/wp_conductivity.py), and
`computeThermalDissipationWarp` (modules/dissipation/wp_dissipation.py) are
called together by `compressibleSPH_Monaghan` (schemes/monaghan.py), and
independently reused by `computeCompSPHAccelWarp`/`computeCompSPHdudtWarp`
(modules/compSPH/{accel,dudt}.py, already gradchecked clean in
gradcheck_compSPH.py) via the shared `computePi_actual` helper
(modules/dissipation/pi.py) -- so this script exercises them at their own,
Monaghan-shaped entry points rather than only indirectly through compSPH's.

Each function is checked against its direct differentiable inputs --
positions, supports, masses, densities, velocities, pressures, soundspeeds,
alphas (plus internalEnergies for conductivity/thermal dissipation, which
read it and viscosity does not) -- with all of them populated, matching
`compressibleSPH_Monaghan`'s real call sites (individual_cs/viscositySwitch/
explicitPressure all True). Same adjacency/support-mode simplification as
gradcheck_compSPH.py: `SupportScheme.Gather` uniformly.

    python scripts/gradcheck_dissipation.py
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

from warpSPH.configurations.moduleConfigurations.diffusionParameters import buildDefaultDiffusionParamsCompressibleSPH
from warpSPH.modules.dissipation.wp_conductivity import computeConductivityWarp
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

    return domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas


def _run(label, warp_fn, needs_energies) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCompressibleSPH()

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        kwargs = dict(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )
        if needs_energies:
            kwargs["conductivityParams"] = diffusionParams
        else:
            kwargs["viscosityParams"] = diffusionParams
        return warp_fn(**kwargs)

    print(f"\n=== {label}: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas)
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
    ok &= _run("computeViscosityWarp", computeViscosityWarp, needs_energies=False)
    ok &= _run("computeConductivityWarp", computeConductivityWarp, needs_energies=True)
    ok &= _run("computeThermalDissipationWarp", computeThermalDissipationWarp, needs_energies=True)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 1.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
