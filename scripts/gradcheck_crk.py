#!/usr/bin/env python3
"""torch.autograd.gradcheck against CRKSPH's core physics kernels -- Tier 1
of docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`crkSPH_step` (schemes/crkSPH.py) always calls both `computeCrkSPHAccelWarp`
(modules/crk/accel.py) and `computeCrkSPHdudtWarp` (modules/crk/dudt.py) in
sequence with the same `crkState`/`apparentVolume`/`queryVelocityTensor`, so
they're checked together here, mirroring gradcheck_compSPH.py's grouping.

Both take a `crkState` (the CRK correction terms A/B/gradA/gradB) and an
`apparentVolume`, real call sites get both from warpSPHCore's
`computeCRKFactors` -- `_gradcheck_common.py`'s `compute_crk_state` wraps
that, treating the result as frozen/non-differentiable input (that op's own
backward is warpSPHCore's `gradcheck_crk_native.py`'s job, not this script's
-- see `compute_crk_state`'s docstring). `queryVelocityTensor` (the velocity
gradient matrix each particle carries into the CRK viscosity limiter) is,
like `ap_ij`/`av_ij` in gradcheck_compSPH.py, an independent differentiable
leaf rather than chained through `computeMomentumConsistent`'s own gradient
-- operator-level testing throughout this family.

Same adjacency/support-mode simplification as gradcheck_compSPH.py:
`SupportScheme.Gather` uniformly, rather than mirroring crkSPH_step's
per-stage mode choice.

**Found and fixed two independent bugs, 2026-08-11 -- both now resolved,
this script passes clean:**

1. **Fixed in this repo.** `modules/crk/limiter.py`'s `computeVanLeer`
   divided by `sgn(grad_j) * abs(grad_j)` (and the symmetric `grad_i` term)
   unconditionally, then patched a resulting NaN *value* after the fact
   (`if ri != ri: ri = 1.0`). The self-interaction pair (`x_ij == 0`, which
   every adjacency list here includes -- see build_adjacency's docstring)
   drives `grad_i == grad_j == 0`, so the division is a genuine 0/0.
   Forward-safe (the NaN gets overwritten), but not backward-safe: Warp's
   (and PyTorch's) reverse-mode AD differentiates the *expression that was
   evaluated*, not the value it was replaced with, so the singular local
   derivative poisons the whole adjoint. Fixed by guarding the division
   itself with an `if/else` before it happens, preserving the exact
   "no flow -> limiter is 1" semantics the NaN-patch was going for.

2. **Fixed upstream, in warpSPHCore.** With the limiter bug fixed (and,
   separately, with both limiters disabled entirely to rule them out),
   `torch.autograd.gradcheck` still failed with a finite but *wrong*
   Jacobian, not a NaN. Isolated by elimination (zeroing viscosity:
   `C_l = C_q = 0`, `alphas = 0`, `velocities = 0`; still failed) down to
   `pressureTerm_ij`'s `gradw_ij`, i.e. `modules/crk/accel.py`'s two
   `computeKernelGradientCRK` calls -- and further isolated by swapping in
   an identity CRK correction (`A=1, B=0, gradA=0, gradB=0`, which reduces
   `warpSPHCore.crk.kernel.correctGradientCRK` to just the plain kernel
   gradient): that passed, pointing at `correctGradientCRK`'s handling of a
   nonzero `B`/`gradB`. Root cause, confirmed by warpSPHCore's own fix: its
   `term4` computed `gradBi`'s contraction against `x_ij` with an explicit
   `for row / for col: product[row] += x_ij[col] * gradBi[row, col]` loop
   that contracted the wrong axis -- a known bug shape (using a
   loop-accumulated value nonlinearly within the same function silently
   produces wrong adjoints, a different failure mode than the "reentrancy"
   class in warpSPHCore's lessons-learned but the same root cause family:
   Warp's reverse-mode AD through hand-written index-accumulation loops is
   not to be trusted without a gradcheck). Fixed upstream by replacing the
   loop with `matmul(wp.transpose(gradBi), x_ij)` -- a single non-loop
   tensor contraction on the correct axis. **This means CRKSPH -- the
   production default scheme (schemeConfig.energyScheme = EnergyScheme.CRK)
   -- was not AD-correct with respect to position until this landed.**

    python scripts/gradcheck_crk.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_crk_state, line_case, make_compressible_state, make_domain
from warpSPHCore import OperationProperties
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.configurations.crkSPH import buildDefaultCRKViscosityParams, buildDefaultDiffusionParamsCRKSPH
from warpSPH.modules.crk.accel import computeCrkSPHAccelWarp
from warpSPH.modules.crk.dudt import computeCrkSPHdudtWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    apparentVolume, densities, crkState = compute_crk_state(positions, supports, masses, kinds, domain, adjacency, kernel=KERNEL)

    velocities = torch.randn(N, DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    internalEnergies = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    pressures = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    soundspeeds = (torch.rand(N, dtype=DTYPE, device=DEVICE) + 0.5).requires_grad_(True)
    alphas = (torch.rand(N, dtype=DTYPE, device=DEVICE) * 0.5 + 0.25).requires_grad_(True)
    velocityTensor = (torch.randn(N, DIM, DIM, dtype=DTYPE, device=DEVICE) * 0.1).requires_grad_(True)
    apparentVolume = apparentVolume.requires_grad_(True)

    return domain, positions, supports, masses, densities, adjacency, kinds, crkState, velocities, internalEnergies, pressures, soundspeeds, alphas, velocityTensor, apparentVolume


def run_accel_gradcheck() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, crkState, velocities, internalEnergies, pressures, soundspeeds, alphas, velocityTensor, apparentVolume = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCRKSPH()
    crkViscosityParams = buildDefaultCRKViscosityParams()

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha, gradV, vol):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        return computeCrkSPHAccelWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            conductivityParams=diffusionParams,
            crkViscosityParams=crkViscosityParams,
            queryVelocityTensor=gradV,
            queryVolumes=vol,
            crkState=crkState,
            adjacency=adjacency,
        )

    print("\n=== computeCrkSPHAccelWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas, velocityTensor, apparentVolume)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_dudt_gradcheck() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds, crkState, velocities, internalEnergies, pressures, soundspeeds, alphas, velocityTensor, apparentVolume = _build_case()
    diffusionParams = buildDefaultDiffusionParamsCRKSPH()
    crkViscosityParams = buildDefaultCRKViscosityParams()

    def f(pos, sup, mass, dens, vel, u, press, cs, alpha, gradV, vol):
        state = make_compressible_state(pos, sup, mass, dens, vel, u, pressures=press, soundspeeds=cs, alphas=alpha, kinds=kinds)
        return computeCrkSPHdudtWarp(
            queryParticles=state,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            conductivityParams=diffusionParams,
            crkViscosityParams=crkViscosityParams,
            queryVelocityTensor=gradV,
            queryVolumes=vol,
            crkState=crkState,
            adjacency=adjacency,
        )

    print("\n=== computeCrkSPHdudtWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, internalEnergies, pressures, soundspeeds, alphas, velocityTensor, apparentVolume)
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
    ok &= run_accel_gradcheck()
    ok &= run_dudt_gradcheck()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 1.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
