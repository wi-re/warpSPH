#!/usr/bin/env python3
"""Gradcheck for `modules/artificialCompressible/wp_jstSwitch.py`'s
`computeJstSwitchWarp` (De Courcy et al. 2024 Eq. (37)'s `chi_i` switch,
`ACSPH_PLAN.md` Part 8 step 8 / AC-JST). Not covered by the `/gradcheck`
skill (`modules/artificialCompressible` isn't in its directory list, same
gap noted for AC-4), so a standalone script -- same pattern as
`scripts/gradcheck_michelUChar.py`/`scripts/gradcheck_nearestSurfaceNormal.py`.

Checks both a constant field (`chi` should be exactly zero, sanity only, no
AD) and gradcheck against a jittered lattice with a random pressure field,
w.r.t. both `pressures` and `positions` (positions matter because `x_ij`
feeds the kernel weight `W_ij`).

Usage:
  python scripts/gradcheck_jstSwitch.py
"""
from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from warpSPHCore import ParticleState, radiusSearchCompactHashMap, OperationProperties, WarpOperation, SupportScheme
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.dataTypes import DomainDescription

from warpSPH.modules.artificialCompressible.wp_jstSwitch import computeJstSwitchWarp

DTYPE = torch.float64
DEVICE = torch.device("cpu")
KERNEL = KernelFunctions.Wendland2


def _build_case(nx=6, jitterFraction=0.3, seed=0):
    dx = 1.0 / nx
    coords = (torch.arange(nx, dtype=DTYPE, device=DEVICE) + 0.5) * dx
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    jitter = (torch.rand(positions.shape, dtype=DTYPE, device=DEVICE, generator=generator) - 0.5) * (jitterFraction * dx)
    positions = (positions + jitter) % 1.0

    n = positions.shape[0]
    rho0 = 1.0
    h = 2.0 * dx
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    masses = torch.full((n,), rho0 * dx * dx, dtype=DTYPE, device=DEVICE)
    densities = torch.full((n,), rho0, dtype=DTYPE, device=DEVICE)
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)

    domain = DomainDescription(
        min=torch.zeros(2, dtype=DTYPE, device=DEVICE),
        max=torch.ones(2, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([True, True], device=DEVICE),
        dim=2,
    )
    state = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.Gather)
    return state, domain, adjacency, generator


def main():
    wp.init()
    state, domain, adjacency, generator = _build_case()
    positions = state.positions
    props = OperationProperties(operation=WarpOperation.Density, kernel=KERNEL, supportMode=SupportScheme.Gather)

    # Sanity: a constant pressure field must give chi == 0 exactly (no AD).
    constPressures = torch.full((positions.shape[0],), 5.0, dtype=DTYPE, device=DEVICE)
    chiConst = computeJstSwitchWarp(state, props, domain, constPressures, adjacency=adjacency)
    print(f"constant field: chi range [{chiConst.min().item():.3e}, {chiConst.max().item():.3e}] (expect exactly 0)")
    assert chiConst.abs().max().item() == 0.0, "constant field should give exactly zero chi"

    pressures = torch.randn(positions.shape[0], dtype=DTYPE, device=DEVICE, generator=generator).requires_grad_(True)
    posReq = positions.clone().requires_grad_(True)

    def f_p(pr):
        s = ParticleState(positions=positions, supports=state.supports, masses=state.masses,
                           densities=state.densities, kinds=state.kinds)
        return computeJstSwitchWarp(s, props, domain, pr, adjacency=adjacency)

    def f_x(xx):
        s = ParticleState(positions=xx, supports=state.supports, masses=state.masses,
                           densities=state.densities, kinds=state.kinds)
        return computeJstSwitchWarp(s, props, domain, pressures.detach(), adjacency=adjacency)

    okP = torch.autograd.gradcheck(f_p, (pressures,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"gradcheck wrt pressures: {okP}")
    okX = torch.autograd.gradcheck(f_x, (posReq,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"gradcheck wrt positions: {okX}")

    assert okP and okX, "gradcheck failed"
    print("PASS")


if __name__ == "__main__":
    main()
