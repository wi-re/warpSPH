#!/usr/bin/env python3
"""Probe (`PST_ALE_PLAN.md` Part 7's second acceptance test / Fig. 1's bottom
row): `delta_u_max` vs. resolution at *fixed* `R/Δx`, on a jittered
Taylor-Green-vortex-like periodic lattice -- the actual convergence-rate
reproduction, not the `rotatingSquarePatch` footprint-drift proxy tried
first (`PST_ALE_PLAN.md` §7.1), which turned out not to discriminate the two
laws at all once `computeMichelShift`'s units bug was fixed.

Quinlan et al.'s truncation-error analysis (`PST_ALE_PLAN.md` Part 2.1) says
`R*||grad(C)|| = O(1)` at fixed `R/Δx` -- so whether a PST's *velocity*
`delta_u` vanishes under refinement depends entirely on its `U_char`:

- Michel (`ShiftingScheme.michel2022`): `U_char_i = max_j|(u_j-u_i).x_hat_ij|`
  is a first-order Taylor expansion of a smooth velocity field over the
  neighbor separation -- `O(Δx)`, vanishing under refinement. Measured here
  via `computeMichelShift(..., returnVelocity=True)`'s third return value,
  Eq. (22)'s own `delta_u` *before* the `dt` multiply that converts it to a
  position increment (`dt` is a real-timestep artifact, not part of the
  theoretical claim, and would confound a resolution sweep run at a fixed
  `dt` or under adaptive-CFL rules that themselves depend on resolution).
- Sun-style delta+ (`ShiftingScheme.deltaSPH`): Table 1's `U_char = Ma*c0` is
  the *global* max flow speed over a *fixed* sound speed -- resolution-
  independent, so `U_char` does not vanish. Its own natural velocity form is
  `d_char * U_char * grad(C) = 2h * Ma*c0 * grad(C)`
  (`sample/wp_deltaShift.py`'s module docstring: "an equivalent
  shifting-velocity form... present in a comment but not used" in
  `modules/shifting/delta.py`) -- measured directly here since
  `computeDeltaShift`'s own public contract returns a *position* delta with
  an implicit acoustic-timestep scaling baked in, which is the wrong
  quantity for this specific comparison.

Usage:
  python scripts/probe_michelConvergenceRate.py [--nx 16 24 32 48 64 96 128]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, nargs='+', default=[16, 24, 32, 48, 64, 96, 128])
parser.add_argument('--rOverDx', type=float, default=2.5)
parser.add_argument('--jitterFraction', type=float, default=0.3)
args = parser.parse_args()

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import math
import types

import numpy as np
import torch
import warp as wp

from warpSPHCore import ParticleState, radiusSearchCompactHashMap, OperationProperties, WarpOperation, SupportScheme
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.dataTypes import DomainDescription

from warpSPH.modules.shifting.michel import computeMichelShift
from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2
DIM = 2
L = 1.0


def _build_case(nx: int):
    dx = L / nx
    coords = (torch.arange(nx, dtype=DTYPE, device=DEVICE) + 0.5) * dx
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    # Fixed-fraction-of-dx jitter, so the disorder is self-similar across
    # resolutions -- essential for a fixed-R/Δx convergence claim to mean
    # anything (an absolute jitter magnitude would make coarser resolutions
    # relatively *more* disordered than finer ones, confounding the sweep).
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    jitter = (torch.rand(positions.shape, dtype=DTYPE, device=DEVICE, generator=generator) - 0.5) * (args.jitterFraction * dx)
    positions = (positions + jitter) % L

    k = 2.0 * math.pi / L
    u = torch.cos(k * positions[:, 0]) * torch.sin(k * positions[:, 1])
    v = -torch.sin(k * positions[:, 0]) * torch.cos(k * positions[:, 1])
    velocities = torch.stack([u, v], dim=1)

    n = positions.shape[0]
    rho0 = 1.0
    h = args.rOverDx * dx
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    masses = torch.full((n,), rho0 * dx * dx, dtype=DTYPE, device=DEVICE)
    densities = torch.full((n,), rho0, dtype=DTYPE, device=DEVICE)
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)

    domain = DomainDescription(
        min=torch.zeros(DIM, dtype=DTYPE, device=DEVICE),
        max=torch.full((DIM,), L, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([True] * DIM, device=DEVICE),
        dim=DIM,
    )
    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.Gather)

    return domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0


def _michel_velocity(domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0) -> float:
    state = ParticleState(positions=positions.clone(), supports=supports, masses=masses,
                           densities=densities.clone(), kinds=kinds)
    state.velocities = velocities
    config = types.SimpleNamespace(kernel=KERNEL, dx=dx, dim=DIM, verletScale=1.2, dt=1e-3)
    schemeConfig = types.SimpleNamespace(
        shiftProperties=types.SimpleNamespace(iterations=1),
        fluid=types.SimpleNamespace(restDensity=rho0),
    )
    R_i = state.supports
    achievedDx_i = torch.pow(state.masses / rho0, 1.0 / DIM)
    beta = (R_i / achievedDx_i) ** 3.0

    _, _, delta_u = computeMichelShift(state, config, schemeConfig, domain, adjacency,
                                        beta=beta, dt=config.dt, iters=1, returnVelocity=True)
    return torch.linalg.norm(delta_u, dim=-1).max().item()


def _sun_velocity(domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0) -> float:
    # Table 1's Sun row: delta_u_i = U_char * d_char * grad(C)_i,
    # U_char = Ma*c0 (fixed, resolution-independent), d_char = 2h --
    # matches `delta.py`'s own commented-out "Michel 2022 scaling" form,
    # without its `dt` factor (see this script's module docstring).
    c0 = 10.0
    v_max = torch.linalg.norm(velocities, dim=-1).max().item()
    Ma = v_max / c0

    state = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    gradC = computeDeltaShiftWarp(
        state,
        operationProperties=OperationProperties(operation=WarpOperation.Density, kernel=KERNEL, supportMode=SupportScheme.Gather),
        referenceParticles=state,
        domain=domain,
        adjacency=adjacency,
        CFL=0.0, computeMach=False, c_max=0.0,
        rho0=rho0, dx=dx,
        # Sun's own defaults (R=0.25, n=4, volumeWeighted=False -- the
        # mean-density weight, matching `computeDeltaShift`'s production call).
    )
    h = supports[0].item()
    velocity = (-Ma * c0 * 2.0 * h) * gradC
    return torch.linalg.norm(velocity, dim=-1).max().item()


def _fit_log_slope(dx_values, y_values) -> float:
    dx_values, y_values = np.asarray(dx_values), np.asarray(y_values)
    ok = np.isfinite(y_values) & (y_values > 0)
    if ok.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(dx_values[ok]), np.log(y_values[ok]), 1)[0])


def main():
    wp.init()

    print(f"R/Δx = {args.rOverDx} (fixed), jitter = {args.jitterFraction}*Δx")
    print(f"{'nx':>5} {'dx':>10} {'michel delta_u_max':>20} {'sun delta_u_max':>18}")
    print('-' * 56)

    dxs, michelVals, sunVals = [], [], []
    for nx in args.nx:
        case = _build_case(nx)
        dx = case[8]
        m = _michel_velocity(*case)
        s = _sun_velocity(*case)
        dxs.append(dx)
        michelVals.append(m)
        sunVals.append(s)
        print(f"{nx:5d} {dx:10.5f} {m:20.6e} {s:18.6e}")

    michelSlope = _fit_log_slope(dxs, michelVals)
    sunSlope = _fit_log_slope(dxs, sunVals)

    print()
    print(f"log-log slope (convergence order, d(log delta_u_max)/d(log dx)):")
    print(f"  michel2022: {michelSlope:.3f}  (expected ~1, first order)")
    print(f"  deltaSPH:   {sunSlope:.3f}  (expected ~0, flat / non-convergent)")


if __name__ == "__main__":
    main()
