#!/usr/bin/env python3
"""torch.autograd.gradcheck against CompSPH's core physics kernels -- Tier 1
of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`compSPH_step` (schemes/compSPH.py) always calls all three of
`computeCompSPHAccelWarp` (modules/compSPH/accel.py),
`computeCompSPHdudtWarp` (modules/compSPH/dudt.py), and
`computeCompSPHBalanceTermWarp` (modules/compSPH/balance.py) in sequence, so
they're gradchecked together here as the plan's "group that always fires
together in a real step" rather than as three separate scripts.

Each function is checked against its own direct differentiable inputs --
positions, supports, masses, densities, velocities, internalEnergies,
pressures, soundspeeds, alphas -- with `pressures`/`soundspeeds`/`alphas`
always populated (real CompSPH runs always populate them; see
make_compressible_state's docstring for what the unpopulated branch would
exercise instead, which is out of scope here). `ap_ij`/`av_ij`, the
pairwise accel/viscosity terms balance's own formula consumes, are treated
as independent leaves rather than chained through accel's own output --
operator-level testing, consistent with every other script in this family:
this checks balance's own backward against its declared inputs, not
accel-then-balance as one fused function. `dt` is left a plain float
throughout; its own differentiability (a materially different code path,
`asScalarArg` + a `wp.array`-typed kernel parameter) is
`gradcheck_scalarArg_dt.py`'s job, not this script's.

`run_balance_gamma_gradcheck` is a regression guard for a real bug, found
while building `examples/compressible/01-sod/sod_backprop.ipynb` (see
`BACKPROP_PLAN.md`): `gamma` used to be passed as `scalar_t(gamma)` into a
by-value `gamma: scalar_t` kernel parameter, severing its gradient while
`dt` -- two arguments earlier in the same call -- already had the
`asScalarArg` + `wp.array(dtype=scalar_t)` treatment. It produced no crash
and no zero gradient, only a silently incomplete one (the EOS route still
carried most of it), so nothing short of a check like this one catches it.
Only `EnergyScheme.CRK` reads `gamma` at all (`s_i = P_i / rho_i**gamma`);
under every other scheme the correct gradient is exactly zero, which is
also asserted below.

All three functions require an explicit adjacency (`if adjacency is None or
isinstance(adjacency, CompactHashMap): raise ValueError`), unlike Tier 0's
targets -- build_adjacency's CSR AdjacencyList satisfies that. `SupportScheme
.Gather` is used uniformly for the adjacency build and every operation here,
rather than mirroring compSPH_step's per-stage mode choice (`
KernelMeanSymmetric` for accel, `Gather` for dudt, `config.supportMode` for
balance) -- that choice is a scheme-tuning detail, not something that changes
which code path a gradcheck exercises, and using one mode throughout keeps
the adjacency shared across all three checks.

    python scripts/gradcheck_compSPH.py
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
from warpSPH.enumTypes import EnergyScheme
from warpSPH.modules.compSPH.accel import computeCompSPHAccelWarp
from warpSPH.modules.compSPH.balance import computeCompSPHBalanceTermWarp
from warpSPH.modules.compSPH.dudt import computeCompSPHdudtWarp

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


def run_accel_gradcheck() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCompressibleSPH()

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        return computeCompSPHAccelWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            conductivityParams=diffusionParams,
            adjacency=adjacency,
        )

    print("\n=== computeCompSPHAccelWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_dudt_gradcheck() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCompressibleSPH()

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        return computeCompSPHdudtWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            conductivityParams=diffusionParams,
            adjacency=adjacency,
        )

    print("\n=== computeCompSPHdudtWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_balance_gradcheck(scheme: EnergyScheme) -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas = _build_case()
    numEdges = adjacency.i.shape[0]
    ap_ij = torch.randn(numEdges, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    av_ij = torch.randn(numEdges, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, vel, u, press, ap, av):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, kinds=kinds)
        return computeCompSPHBalanceTermWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            energyScheme=scheme,
            dt=0.01,
            gamma=1.4,
            pairWise_pressureAccel=ap,
            pairWise_viscosityAccel=av,
            adjacency=adjacency,
        )

    print(f"\n=== computeCompSPHBalanceTermWarp ({scheme.name}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, ap_ij, av_ij)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_balance_gamma_gradcheck(scheme: EnergyScheme) -> bool:
    """Gradcheck balance's `gamma` itself, as a differentiable tensor argument."""
    domain, positions, supports, masses, densities, adjacency, kinds, velocities, internalEnergies, pressures, soundspeeds, alphas = _build_case()
    numEdges = adjacency.i.shape[0]
    ap_ij = torch.randn(numEdges, DIM, dtype=DTYPE, device=DEVICE)
    av_ij = torch.randn(numEdges, DIM, dtype=DTYPE, device=DEVICE)
    gamma = torch.tensor(1.4, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(g):
        state = make_compressible_state(positions, supports, masses, densities, velocities, internalEnergies, pressures=pressures, kinds=kinds)
        return computeCompSPHBalanceTermWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            energyScheme=scheme,
            dt=0.01,
            gamma=g,
            pairWise_pressureAccel=ap_ij,
            pairWise_viscosityAccel=av_ij,
            adjacency=adjacency,
        )

    print(f"\n=== computeCompSPHBalanceTermWarp ({scheme.name}), gamma: torch.autograd.gradcheck ===")
    try:
        ok = bool(torch.autograd.gradcheck(f, (gamma,), eps=1e-6, atol=1e-5))
        # Only CRK reads gamma; anywhere else a *non*-zero gradient would mean
        # the argument had leaked into a branch that should not see it.
        f(gamma).sum().backward()
        expectNonZero = scheme is EnergyScheme.CRK
        if gamma.grad is None:
            print("FAILED: no gradient reached gamma at all -- the kernel argument is "
                  "severed (a by-value scalar_t parameter, or a call site that "
                  "collapses the tensor to a float)")
            return False
        if (gamma.grad.abs().item() > 0) != expectNonZero:
            print(f"FAILED: gamma.grad = {gamma.grad.item()!r}, expected "
                  f"{'a non-zero' if expectNonZero else 'exactly zero'} gradient")
            return False
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return ok
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= run_accel_gradcheck()
    ok &= run_dudt_gradcheck()
    for scheme in EnergyScheme:
        ok &= run_balance_gradcheck(scheme)
    for scheme in EnergyScheme:
        ok &= run_balance_gamma_gradcheck(scheme)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 1.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
