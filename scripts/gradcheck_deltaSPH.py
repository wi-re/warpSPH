#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/deltaSPH/{wp_densityDelta,wp_viscosityDelta}.py
-- Tier 2 of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeDensityDiffusionDeltaSPH` and `computeVelocityDiffusionDeltaSPH` are the
deltaSPH scheme's density- and velocity-diffusion correction terms
(schemes/deltaSPH.py). Both share the same `Func_i` shape as every other module
gradchecked so far: a same-array-safe apparent-volume ternary
(`apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]`, confirmed
non-issue per Tier 0's own findings since the two branches read different arrays)
and a kernel/kernel-gradient call through `computeKernelGradientCRK`.

`computeDensityDiffusionDeltaSPH` additionally takes `queryGradRho`/`queryGradRhoL`
(density-gradient fields, both plain and Lagrangian-corrected) as direct
differentiable inputs -- exercised here across all 5 `DensityDiffusionScheme`
values, since the scheme enum selects between different linear combinations of
those two gradient fields (see the module's `Func_i` for the branch table).

`computeVelocityDiffusionDeltaSPH` takes `inviscid` (bool) which switches between
an artificial-viscosity-style term and a physical-viscosity term (`alphaToNu`),
plus the standard "moving apart doesn't dissipate" kink
(`if mu_ij > 0: mu_ij = 0.0`) already gradchecked clean in this same shape by
gradcheck_dissipation.py's Monaghan viscosity. Checked in both `inviscid` modes.

    python scripts/gradcheck_deltaSPH.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.enumTypes import DensityDiffusionScheme
from warpSPH.modules.deltaSPH.wp_densityDelta import computeDensityDiffusionDeltaSPH
from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    return domain, positions, supports, masses, densities, adjacency, kinds


def _particles(pos, sup, mass, dens, kinds, velocities=None):
    p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
    if velocities is not None:
        p.velocities = velocities
    return p


def _run_density(label, densityScheme) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    gradRho = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    gradRhoL = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, gRho, gRhoL):
        p = _particles(pos, sup, mass, dens, kinds)
        return computeDensityDiffusionDeltaSPH(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            densityScheme=densityScheme,
            queryGradRho=gRho,
            queryGradRhoL=gRhoL,
            adjacency=adjacency,
        )

    print(f"\n=== computeDensityDiffusionDeltaSPH [{label}]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, gradRho, gradRhoL)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_velocity(label, inviscid) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()
    velocities = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, vel):
        p = _particles(pos, sup, mass, dens, kinds, velocities=vel)
        return computeVelocityDiffusionDeltaSPH(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            inviscid=inviscid,
            alpha=0.02,
            c_s=1.0,
            nu=5e-3,
            queryVelocities=vel,
            adjacency=adjacency,
        )

    print(f"\n=== computeVelocityDiffusionDeltaSPH [{label}]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    for scheme in DensityDiffusionScheme:
        ok &= _run_density(scheme.name, scheme)
    for inviscid in (True, False):
        ok &= _run_velocity(f"inviscid={inviscid}", inviscid)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
