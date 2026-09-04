#!/usr/bin/env python3
"""Manual troubleshooting harness for the computeCompSPHBalanceTermWarp
segfault (docs/historic_plans/CLEANUP_PLAN.md, Phase 4.2).

STATUS 2026-08-15: **resolved -- the crash no longer reproduces.** Clean in
every mode (`forward`, `forward-grad`, `backward`) and for all six energy
schemes under `--scheme all`. The "only `equalWork` survives" finding below no
longer holds; all six pass.

Cause, per the user, was two things and *neither* was an out-of-bounds read in
the kernel -- the lead this investigation was chasing:

  1. **This harness was itself wrong.** It never built the reference states
     properly. The "cold process with fresh minimal allocations" theory below
     was the wrong track: the minimal case was not a stripped-down valid input,
     it was an invalid one.
  2. **A real bug in the call path**: when `referenceVolumes` was not passed
     explicitly it stayed `None` instead of falling back to `queryVolumes`, so
     a reference state with no volume member handed the kernel a null array.
     `computeCompSPHBalanceTermWarp` was one of several entry points affected.
     Fixed upstream in `warpSPHCore` on 2026-08-06 by commit `120c4bf`
     ("make the state path the primary one"), which added the fallback now at
     `warpSPHCore/operations.py:52`:

         referenceVolumes = referenceVolumes if referenceVolumes is not None else queryVolumes

CRK specifically was fixed by passing a fully set-up state. The rest of this
docstring is the original investigation, preserved as the elimination record --
but read its conclusions knowing the harness itself was the faulty input.

Background: a native segfault (confirmed via `python -X faulthandler`, not a
Python exception) reproduces when calling `computeCompSPHBalanceTermWarp`
(modules/compSPH/balance.py) standalone with a freshly-built, minimal
particle case. Ruled out so far, each with its own isolated test:

  * The EnergyScheme arithmetic itself -- isolated into a bare kernel with no
    neighbor loop, all 6 schemes (including monotonic/hybrid, and the ternary
    -> if/else rewrite already applied to those two): clean, correct output.
  * The full wp.static() 6-branch dispatch, same isolation: clean.
  * ap_ij/av_ij edge-count shape -- verified by direct inspection of the
    adjacency CSR structure (numNeighbors.sum() == len(adjacency.j), every
    beginIndex+numIndices-1 in bounds).
  * Domain size (oversized vs. snug-fit), dim (1 vs. 2), warp-lang version
    (crashes identically under pinned 1.12.0 and dev 1.17.0.dev3), and
    adjacency construction method (plain radiusSearchCompactHashMap/Gather
    vs. production's buildVerletList/SuperSymmetric) -- none of these change
    the outcome.
  * It is NOT scheme-specific: PdV, diminishing, monotonic, hybrid, and
    **CRK** (the production default -- the one all 42 real tests exercise)
    all crash standalone. Only `equalWork` (f=0.5 constant, touches none of
    the per-particle data) survives.

What's NOT yet tested: reproducing from *inside* a real step-function call
(compSPH_step / crkSPH_step), with a real CompSPHState and real
crkState/gradHState/queryVolumes rather than the None defaults every repro
so far has used, after the dozen-plus other kernel launches a real step does
first. If this is a genuine out-of-bounds read, a "cold" process with fresh,
minimal allocations is exactly the condition that would turn a silent
garbage read into a crash where a warmed-up process's memory layout might
currently mask it -- unconfirmed, but the most likely remaining lead.

Usage -- run with -X faulthandler so a crash gets a native backtrace
instead of a bare "Segmentation fault" with no context:

    python -X faulthandler scripts/troubleshoot_balanceTerm_segfault.py --mode forward
    python -X faulthandler scripts/troubleshoot_balanceTerm_segfault.py --mode forward-grad
    python -X faulthandler scripts/troubleshoot_balanceTerm_segfault.py --mode backward
    python -X faulthandler scripts/troubleshoot_balanceTerm_segfault.py --mode forward --scheme all
    python -X faulthandler scripts/troubleshoot_balanceTerm_segfault.py --mode forward --debug-mode

Each mode prints a line right before the kernel launch; if the process dies
with no further output, the launch itself is where it died. `--debug-mode`
turns on Warp's bounds-checked array access (`wp.config.mode = "debug"`,
`launch_array_access_mode = STRICT`) -- much slower (full kernel rebuild,
every module gets recompiled in debug mode) but converts a raw segfault into
a clean assertion naming the offending array, if the crash is an
out-of-bounds access. Already used once in this investigation and caught a
real (separate, unrelated) OOB assertion inside warpSPHCore's own
computeSPHDensity_Kernel on the same particle case -- worth knowing about
even though it fired before reaching the balance kernel.
"""

from __future__ import annotations

import argparse
import os


os.environ.setdefault("warpSPHCore_PRECISION", "float64")
from warpSPH.systems.compressibleMonaghan import CompressibleState

import sys

import torch
import warp as wp

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def build_case(dim: int, n: int, adjacency_kind: str):
    from warpSPHCore import DomainDescription, OperationProperties, ParticleState, warpOperation, buildVerletList, radiusSearchCompactHashMap
    from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme, WarpOperation

    kernel = KernelFunctions.Wendland2

    if dim == 1:
        domain = DomainDescription(
            min=torch.tensor([-1.5], dtype=DTYPE, device=DEVICE),
            max=torch.tensor([1.5], dtype=DTYPE, device=DEVICE),
            periodic=torch.tensor([False], device=DEVICE),
            dim=1,
        )
        x = torch.linspace(-1.0, 1.0, n, dtype=DTYPE, device=DEVICE)
        positions = x.unsqueeze(-1).contiguous()
        spacing = 2.0 / (n - 1)
    else:
        side = max(int(round(n ** 0.5)), 2)
        domain = DomainDescription(
            min=torch.tensor([-1.5] * dim, dtype=DTYPE, device=DEVICE),
            max=torch.tensor([1.5] * dim, dtype=DTYPE, device=DEVICE),
            periodic=torch.tensor([False] * dim, device=DEVICE),
            dim=dim,
        )
        coords = torch.linspace(-1.0, 1.0, side, dtype=DTYPE, device=DEVICE)
        grids = torch.meshgrid(*([coords] * dim), indexing="ij")
        positions = torch.stack([g.reshape(-1) for g in grids], dim=1).contiguous()
        n = positions.shape[0]
        spacing = 2.0 / (side - 1)

    h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    particles = CompressibleState(
        positions=positions,
        velocities = torch.zeros((n, dim), dtype=DTYPE, device=DEVICE),
        supports=supports, 
        masses=masses, 
        densities=None, 
        kinds=kinds,
        materials= torch.zeros(n, dtype=torch.int32, device=DEVICE),
        UIDs = torch.arange(n, dtype=torch.int64, device=DEVICE),
        UIDcounter = n,
        internalEnergies = torch.rand(n, dtype=DTYPE, device=DEVICE) + 0.5,
    )

    if adjacency_kind == "verlet":
        adjacency = buildVerletList(particles, domain, verletScale=1.1, supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)
        support_mode = SupportScheme.SuperSymmetric
    else:
        adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
        support_mode = SupportScheme.Gather

    print(f"case: dim={dim} n={n} numEdges={adjacency.i.shape[0]} adjacency={adjacency_kind}", flush=True)

    densities = warpOperation(
        particles,
        OperationProperties(kernel=kernel, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    particles.densities = densities
    print(f"densities: {densities}", flush=True)

    particles.pressures = densities ** 1.4

    numEdges = adjacency.i.shape[0]
    ap_ij = torch.zeros((numEdges, dim), dtype=DTYPE, device=DEVICE)
    av_ij = torch.ones((numEdges, dim), dtype=DTYPE, device=DEVICE)
    velocities = torch.randn((n, dim), dtype=DTYPE, device=DEVICE)
    energies = torch.rand(n, dtype=DTYPE, device=DEVICE) + 0.5
    pressures = torch.rand(n, dtype=DTYPE, device=DEVICE) + 0.5

    return dict(
        particles=particles, domain=domain, adjacency=adjacency, kernel=kernel, support_mode=support_mode,
        ap_ij=ap_ij, av_ij=av_ij, velocities=velocities, energies=energies, pressures=pressures,
    )


def run_one(mode: str, scheme, case):
    from warpSPHCore import OperationProperties
    from warpSPH.modules.compSPH.balance import computeCompSPHBalanceTermWarp

    grad_leaf = mode != "forward"
    dt = torch.tensor([0.5], dtype=DTYPE, device=DEVICE, requires_grad=grad_leaf)

    print(f"--- scheme={scheme.name} mode={mode} ---", flush=True)
    print("launching...", flush=True)
    out = computeCompSPHBalanceTermWarp(
        queryParticles=case["particles"],
        operationProperties=OperationProperties(kernel=case["kernel"], supportMode=case["support_mode"]),
        domain=case["domain"],
        energyScheme=scheme,
        dt=dt,
        gamma=1.4,
        pairWise_pressureAccel=case["ap_ij"],
        pairWise_viscosityAccel=case["av_ij"],
        queryEnergies=case["energies"],
        queryVelocities=case["velocities"],
        queryPressures=case["pressures"],
        adjacency=case["adjacency"],
    )
    print(f"forward OK: requires_grad={out.requires_grad} shape={tuple(out.shape)}", flush=True)

    if mode == "backward":
        print("calling .backward()...", flush=True)
        out.sum().backward()
        print(f"backward OK: dt.grad={dt.grad}", flush=True)


def main():
    from warpSPH.enumTypes import EnergyScheme

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["forward", "forward-grad", "backward"], default="forward",
                         help="forward: no requires_grad anywhere. forward-grad: dt requires_grad=True, "
                              "forward pass only (tape recorded, no .backward()). backward: forward-grad plus .backward().")
    parser.add_argument("--scheme", default="CRK", help="EnergyScheme name, or 'all' to loop every scheme in one process.")
    parser.add_argument("--dim", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n", type=int, default=7, help="Particle count (1D) or roughly-n^(1/dim) per side (2D/3D).")
    parser.add_argument("--adjacency", choices=["hashmap", "verlet"], default="hashmap",
                         help="hashmap: plain radiusSearchCompactHashMap/Gather. verlet: production's buildVerletList/SuperSymmetric.")
    parser.add_argument("--debug-mode", action="store_true",
                         help="wp.config.mode='debug' + launch_array_access_mode=STRICT. Much slower (full rebuild) "
                              "but turns an OOB segfault into a clean assertion naming the array.")
    args = parser.parse_args()

    if args.debug_mode:
        wp.config.mode = "debug"
        wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.STRICT
        print("debug mode + STRICT array access enabled -- expect a full kernel rebuild.", flush=True)

    wp.init()
    torch.manual_seed(0)

    case = build_case(args.dim, args.n, args.adjacency)

    schemes = list(EnergyScheme) if args.scheme == "all" else [EnergyScheme[args.scheme]]
    for scheme in schemes:
        run_one(args.mode, scheme, case)

    print("\nALL DONE (if you see this, nothing crashed).", flush=True)


if __name__ == "__main__":
    main()
