#!/usr/bin/env python3
"""Probe (`PST_ALE_PLAN.md` Part 7, sec. 4.4): the "nearly free" acceptance
test for Michel et al. 2022's (`literature/michel2022`) PST -- add a uniform
velocity to a Taylor-Green-vortex-like periodic lattice and check whether the
raw shift field changes.

This tests Eq. (20)-(22)'s own Galilean invariance directly (`computeMichelShift`,
`modules/shifting/michel.py`), not the full `solveShifting` driver -- the
driver layers an additional, deliberately *not* Galilean-invariant safety
clamp on top (`ShiftProperties.maxShiftVelocityFraction`, keyed to the
*absolute* `max finite particle speed`), which would confound this specific
claim. Two laws are compared:

- `ShiftingScheme.michel2022`: Eq. (20)'s characteristic velocity is the
  *relative* quantity `max_j |(u_j-u_i).x_hat_ij|` -- this must be exactly
  unchanged (to float64 rounding) when a constant `U0` is added to every
  particle's velocity, since `u_j - u_i` cancels `U0` identically.
- `ShiftingScheme.deltaSPH` (Sun-style, `modules/shifting/delta.py`): its
  `Ma = v_max / c0` uses the *absolute* per-call max speed, so adding `U0`
  changes `Ma` for every particle -- the negative control this script expects
  to fail, reproducing Part 7's discriminating claim ("our current law will
  fail it... Michel's must pass to machine precision").

Free-surface machinery is not exercised (a periodic lattice has none); this
isolates Eqs. (20)-(22) themselves.

    python scripts/probe_michelTranslationInvariance.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import math
import types

import torch
import warp as wp

from warpSPHCore import ParticleState, radiusSearchCompactHashMap
from warpSPHCore.enumTypes import KernelFunctions, SupportScheme
from warpSPHCore.dataTypes import DomainDescription

from warpSPH.modules.shifting.michel import computeMichelShift
from warpSPH.modules.shifting.delta import computeDeltaShift

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2
DIM = 2
NX = 16
L = 1.0
U0 = torch.tensor([3.7, -2.1], dtype=DTYPE, device=DEVICE)


def _build_lattice():
    # A perfectly regular lattice makes the PST term itself vanish by
    # symmetry (opposing neighbors' kernel gradients cancel exactly) for
    # *both* laws, which would make this probe vacuous -- real SPH particle
    # distributions are never perfectly regular, and this is exactly the
    # disordered case a PST exists for. A small deterministic jitter (a
    # fraction of the spacing, well inside the kernel support) breaks the
    # symmetry while keeping the configuration periodic.
    dx = L / NX
    coords = (torch.arange(NX, dtype=DTYPE, device=DEVICE) + 0.5) * dx
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    jitter = (torch.rand(positions.shape, dtype=DTYPE, device=DEVICE, generator=generator) - 0.5) * (0.3 * dx)
    positions = (positions + jitter) % L

    k = 2.0 * math.pi / L
    u = torch.cos(k * positions[:, 0]) * torch.sin(k * positions[:, 1])
    v = -torch.sin(k * positions[:, 0]) * torch.cos(k * positions[:, 1])
    velocities = torch.stack([u, v], dim=1)

    n = positions.shape[0]
    rho0 = 1.0
    h = 2.5 * dx
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


def _michel_shift(domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0):
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

    delta, _ = computeMichelShift(state, config, schemeConfig, domain, adjacency, beta=beta, dt=config.dt, iters=1)
    return delta


def _sun_shift(domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0):
    state = ParticleState(positions=positions.clone(), supports=supports, masses=masses,
                           densities=densities.clone(), kinds=kinds)
    state.velocities = velocities

    config = types.SimpleNamespace(kernel=KERNEL, dx=dx, dim=DIM, verletScale=1.2, dt=1e-3)
    schemeConfig = types.SimpleNamespace(
        shiftProperties=types.SimpleNamespace(iterations=1, CFL=0.3, computeMach=True),
        fluid=types.SimpleNamespace(restDensity=rho0, fixedSoundSpeed=1.0),
    )

    delta, _ = computeDeltaShift(state, config, schemeConfig, domain, adjacency, iters=1)
    return delta


def _check(name, shift_fn, case):
    domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0 = case

    base = shift_fn(domain, positions, velocities, supports, masses, densities, kinds, adjacency, dx, rho0)
    shifted = shift_fn(domain, positions, velocities + U0, supports, masses, densities, kinds, adjacency, dx, rho0)

    maxDiff = (base - shifted).abs().max().item()
    return maxDiff


def main():
    wp.init()
    case = _build_lattice()

    print("Translation-invariance probe: shift(u) vs. shift(u + U0), U0 =", U0.tolist())
    print(f"{'law':>14} {'max |delta|':>14} {'expected':>10} {'result':>8}")

    michelDiff = _check("michel2022", _michel_shift, case)
    michelOk = michelDiff < 1e-10
    print(f"{'michel2022':>14} {michelDiff:14.3e} {'~0':>10} {('PASS' if michelOk else 'FAIL'):>8}")

    sunDiff = _check("deltaSPH", _sun_shift, case)
    sunFailsAsExpected = sunDiff > 1e-6
    print(f"{'deltaSPH':>14} {sunDiff:14.3e} {'>0':>10} "
          f"{('PASS (fails as expected)' if sunFailsAsExpected else 'UNEXPECTED PASS')}")

    ok = michelOk and sunFailsAsExpected
    print()
    print("ALL PASSED." if ok else "FAILED -- see this script's docstring.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
