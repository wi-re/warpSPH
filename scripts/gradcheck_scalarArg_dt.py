#!/usr/bin/env python3
"""torch.autograd.gradcheck proof that warpSPHCore's differentiable-scalar
kernel-argument mechanism (asScalarArg + a wp.array(dtype=scalar_t) kernel
parameter read as param[0]) works in *this* repo's own environment.

Context (see docs/historic_plans/CLEANUP_PLAN.md, Phase 4.2): `computeCompSPHBalanceTermWarp`
(modules/compSPH/balance.py, shared by compSPH_step and crkSPH_step) used to
declare its `dt` parameter as a plain by-value `scalar_t`, and every call
site collapsed `dt` to a Python float via `.detach().cpu().item()` before
calling in -- confirmed to be a no-op at the time (warpSPHCore's autograd
bridge never carried gradients through non-array kernel scalars regardless
of whether the caller detached them). warpSPHCore has since grown
`asScalarArg` (autograd/scalar_arg.py) plus the wp.array(dtype=scalar_t) /
param[0] convention for opting a kernel parameter into differentiability,
proven there by `scripts/gradcheck_scalar_arg_native.py` on a demo kernel
built from an existing warpSPHCore primitive (Interpolate).

This script reruns that same proof in this repo's environment (own editable
install, own effective warp-lang version -- see the note below) to confirm
the mechanism itself is sound here too. It PASSES.

**Update: the production integration is done and verified, not reverted.**
An earlier version of this docstring reported that applying this exact
pattern to `computeCompSPHBalanceTermWarp` reliably segfaulted the moment a
gradient through `dt` was requested, and that the change had been reverted.
That segfault turned out to be a real, unrelated, pre-existing bug --
`referenceEnergies`/`referencePressures` were missing the query->reference
fallback `referenceVelocities` already had, so a standalone call (like this
script's, or any gradcheck-shaped call) reached the kernel launch with
`referenceEnergies=None` regardless of the dt work. Root-caused and fixed in
`modules/compSPH/balance.py` (see docs/historic_plans/CLEANUP_PLAN.md Phase 4.2 for the full
fallback-chain trace and `scripts/troubleshoot_balanceTerm_segfault.py` for
the harness that found it). With that fixed, the dt integration -- `dt:
wp.array(dtype=scalar_t)` on the top-level `@wp.kernel`, read once via
`dt[0]`, forwarded as a plain `scalar_t` into the nested `@wp.func` layers,
`asScalarArg` at the Python call site -- was re-applied and works cleanly:
`torch.autograd.gradcheck` against `computeCompSPHBalanceTermWarp` under
`EnergyScheme.monotonic` (the one scheme with a genuinely smooth,
non-zero-a.e. dt-dependence) passes for both 0-dim and 1-element `dt`, `CRK`
correctly resolves to `dt.grad == 0` (its output depends only on the sign of
the dt-carrying term, never its magnitude), and the plain-float regression
case still resolves to `requires_grad=False`. Full test suite: 42/42.

It was never the known "ternary assigned to a local, both branches index the
same array" adjoint-zeroing bug (see docs/lessons_learned.md in
warpSPHCore) -- that bug produces a silently-*wrong* zero gradient, not a
crash -- and the `EnergyScheme.monotonic`/`.hybrid` ternary rewrite applied
while chasing that theory turned out not to be what fixed anything either,
though it's kept (harmless, matches this codebase's convention).

**Environment note:** this repo's `warp` conda env had warp-lang 1.12.0
(matching pyproject.toml's unpinned `"warp-lang"` resolving to the last
PyPI release) at the start of this investigation, and 1.17.0.dev3
(installed from a local `~/dev/warp_dev` checkout) by the end of it --
changed by something outside this investigation, mid-session. Worth knowing
if numbers here don't match a fresh run: which warp-lang build is actually
active is no longer guaranteed to match what pyproject.toml implies.

    python scripts/gradcheck_scalarArg_dt.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
from typing import Any

import torch
import warp as wp

from warpSPHCore import (
    DomainDescription,
    OperationProperties,
    ParticleState,
    adjacencyData,
    asScalarArg,
    domainData,
    gridData,
    kernelState,
    launch_kernel,
    radiusSearchCompactHashMap,
    scalar_t,
    warpOperation,
    warpWrapper2,
)
from warpSPHCore.coreOperations.wp_interpolate import computeSPHInterpolation_Func_Adjacency
from warpSPHCore.enumTypes import (
    KernelFunctions,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2


@wp.kernel
def _scalarArgDemo_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,  # type: ignore
    correctionData: Any,
    kernelProperties: kernelState,

    referenceValues: wp.array(dtype=scalar_t),  # type: ignore
    dt: wp.array(dtype=scalar_t),  # type: ignore -- differentiable scalar, read via [0]

    outputValues: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    interpolated = computeSPHInterpolation_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        referenceValues,
        scalar_t(0.0),
    )
    outputValues[i] = interpolated * dt[0]


def _scalarArgDemo(queryParticles, referenceParticles, domain, adjacency, referenceValues, dt):
    outputSize = queryParticles.positions.shape[0]
    operationProperties = OperationProperties(
        kernel=KERNEL, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
    )
    return warpWrapper2(
        launcher=launch_kernel,
        kernel=_scalarArgDemo_Kernel,
        outputSizes=outputSize,
        outputDtypes=scalar_t,
        defaultStateArguments=(
            queryParticles, operationProperties, domain,
            None, None, adjacency, referenceParticles,
            None, None, None,
        ),
        additionalArguments=(
            referenceValues,
            asScalarArg(dt, device=queryParticles.positions.device),
        ),
    )


def _line_case(n: int, xmin: float = -1.0, xmax: float = 1.0):
    x = torch.linspace(xmin, xmax, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    positions = x.detach().clone().requires_grad_(True)
    spacing = (xmax - xmin) / max(n - 1, 1)
    h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def _make_domain(dim: int = 1, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


def _build_adjacency(positions, supports, masses, domain):
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.Gather)
    return adjacency, kinds


def _compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone()


def run_gradcheck(dt_shape: str) -> bool:
    domain = _make_domain()
    positions, supports, masses = _line_case(7)
    adjacency, kinds = _build_adjacency(positions, supports, masses, domain)
    densities = _compute_densities(positions, supports, masses, kinds, domain, adjacency)

    reference_values = torch.randn(7, dtype=DTYPE, device=DEVICE, requires_grad=True)
    dt = (
        torch.tensor(0.7, dtype=DTYPE, device=DEVICE, requires_grad=True)
        if dt_shape == "0-dim"
        else torch.tensor([0.7], dtype=DTYPE, device=DEVICE, requires_grad=True)
    )

    def f(pos, sup, mass, rval, dt_):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        return _scalarArgDemo(p, p, domain, adjacency, rval, dt_)

    print(f"\n=== demo kernel, dt {dt_shape}: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (positions, supports, masses, reference_values, dt), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    for dt_shape in ("0-dim", "1-element"):
        ok &= run_gradcheck(dt_shape)

    print()
    if ok:
        print("ALL PASSED -- the differentiable-scalar mechanism itself is sound here.")
        print("computeCompSPHBalanceTermWarp's own dt integration is also verified --")
        print("see this script's module docstring for how.")
    else:
        print("FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
