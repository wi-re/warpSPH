#!/usr/bin/env python3
"""Probe (`PST_ALE_PLAN.md` Part 8 step 2 / Michel §5.4, the "still open" half
of the convergence-rate reproduction): `delta_u_max` vs. resolution at *fixed*
`R/Δx`, on a free-surface-bounded lattice.

`probe_michelConvergenceRate.py` reproduced Fig. 1's headline result --
`michel2022` is first-order convergent, `deltaSPH` is flat -- on a *periodic*
Taylor-Green-vortex lattice. A periodic domain has no free surface, so it
cannot exercise Eq. (48) (the free-surface projection: `beta`'s linear decay
to 1, the inherited normal `n~` from the nearest free-surface neighbour, the
`sigma` ramp, the `lambda < 0.4` gate) at all -- Table 3 records that Eq. (48)
is the only law in Michel's survey with `lim_{Dx->0} delta_u^FS = 0`, and that
claim is untested until a free surface is actually in the picture.

This builds the free-surface analogue directly: the *exact same* jittered
Taylor-Green-like lattice and velocity field as `probe_michelConvergenceRate.py`
(so the two probes isolate one variable -- periodic vs. free-surface-bounded),
except the domain is no longer periodic: positions are not wrapped, so the box
edge is a genuine free surface rather than an artifact-free periodic boundary.

(A rigid-rotation field -- the obvious first choice, matching
`rotatingSquarePatch`'s own t=0 state -- turns out to be degenerate for this
specific measurement: `U_char_i = max_j|(u_j-u_i).x_hat_ij|` is *exactly* zero
for solid-body rotation at every pair, for every resolution, because
`v_j-v_i = omega x_ij` is exactly perpendicular to `x_ij` -- an identity of
rigid rotation, not a resolution effect. That produces a flat all-epsilon
table with no signal to fit a slope to. The Taylor-Green field has real local
strain, giving the genuinely nonzero, curvature-driven `U_char` the periodic
probe already validated -- the free-surface question this script asks is
specifically whether *that* signal's first-order convergence survives Eq.
(48)'s near-surface treatment, which needs a real free surface, not a
different velocity field.)

This measures the *projected* free-surface velocity `delta_u^FS`, built from
the same three real modules production code uses
(`detectFreeSurface`, `computeNearestSurfaceNormalWarp`,
`computeMichelShift(..., returnVelocity=True)`), with Eq. (48)'s projection
(`modules/shifting/wrapper.py`'s `ShiftingProjectionScheme.michel2022` branch)
applied directly to the *velocity* form rather than to the `dt`-scaled
position delta `solveShifting` normally projects -- legitimate because that
projection (`outward`/`sigma`/`lambdaGate` all depend on direction, `d^FS` and
`lMin`, never on the input vector's magnitude) is homogeneous of degree 1 in
the vector it is given, so projecting `delta_u` (velocity) yields exactly
Eq. (48)'s own `delta_u^FS`, the quantity Michel's own convergence claim is
stated for.

**Reports two maxima, not one.** The global `max_i` over every particle turns
out to reproduce the periodic probe's bulk numbers almost exactly at every
`nx` -- because Eq. (48) can only ever *shrink* a shift (project out a normal
component, scale by `lambda^2 <= 1`), the global maximum stays pinned to
whichever bulk saddle point already held it in the periodic case, and the
free-surface treatment never gets to speak. That is a real (if initially
counterintuitive) finding, not a bug: it says a single scalar `max_i` over the
whole domain is the wrong statistic for isolating Eq. (48)'s own behaviour.
So this script *also* reports the maximum restricted to the dilated
free-surface set `V` (`surfaceIndicator`, i.e. exactly the particles Eq. (48)
actually modifies) -- that restricted maximum is the one that speaks to
Table 3's claim.

Usage:
  python scripts/probe_michelFreeSurfaceConvergenceRate.py [--nx 16 24 32 48 64 96 128]
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

from warpSPHCore import ParticleState, buildVerletList, OperationProperties, WarpOperation, SupportScheme
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.dataTypes import DomainDescription

from warpSPH.modules.shifting.michel import computeMichelShift
from warpSPH.modules.surfaceDetection import detectFreeSurface, computeNearestSurfaceNormalWarp
from warpSPH.configurations.moduleConfigurations.surfaceDetection import (
    SurfaceDetectionConfig, SurfaceDetectionScheme, NormalSource,
)

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

    # Same self-similar-jitter device as the periodic probe: an absolute
    # jitter magnitude would make coarse resolutions relatively *more*
    # disordered than fine ones, confounding a fixed-R/Δx sweep. No wrap --
    # this is a bounded patch, not a periodic box, and the whole point is the
    # free surface at its edge.
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    jitter = (torch.rand(positions.shape, dtype=DTYPE, device=DEVICE, generator=generator) - 0.5) * (args.jitterFraction * dx)
    positions = positions + jitter

    # Same Taylor-Green-like field as `probe_michelConvergenceRate.py` --
    # see module docstring for why (rigid rotation is degenerate here).
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

    margin = 4.0 * h
    domain = DomainDescription(
        min=torch.full((DIM,), -margin, dtype=DTYPE, device=DEVICE),
        max=torch.full((DIM,), L + margin, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * DIM, device=DEVICE),
        dim=DIM,
    )
    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    p.velocities = velocities
    # SuperSymmetric to match `solveShifting`'s own adjacency build -- the
    # free-surface operators (Barecasco detection, LambdaGrad renormalisation,
    # dilation) all query with `OperationDirection.AllToAll` +
    # `SupportScheme.SuperSymmetric` internally.
    adjacency = buildVerletList(p, domain, verletScale=1.2, supportMode=SupportScheme.SuperSymmetric)

    return domain, p, adjacency, dx, rho0, h


def _michel_free_surface_velocity(domain, p, adjacency, dx, rho0, h) -> float:
    config = types.SimpleNamespace(kernel=KERNEL, dx=dx, dim=DIM, domain=domain, verletScale=1.2, dt=1e-3)
    schemeConfig = types.SimpleNamespace(
        shiftProperties=types.SimpleNamespace(iterations=1),
        fluid=types.SimpleNamespace(restDensity=rho0),
    )
    # `buildDefaultSurfaceDetectionConfig()`'s own scheme/normalSource
    # (Barecasco/LambdaGrad) -- the pair `rotatingSquarePatch` actually runs
    # under via `configureWeaklyCompressible`'s shared default.
    surfaceConfig = SurfaceDetectionConfig(
        active=True,
        barecascoThreshold=math.pi / 3,
        expansionIterations=1,
        scheme=SurfaceDetectionScheme.Barecasco,
        normalSource=NormalSource.LambdaGrad,
    )

    # `modules/surfaceDetection/wrapper.py::detectFreeSurface` returns
    # `(fsm_raw, fs_dilated, normals, renormalizationState, lambdas)`;
    # `solveShifting` binds position 0 to the name `fs` (the *raw*, undilated
    # set Eq. (47)'s nearest-neighbour search wants) and position 1 to `fsm`
    # (the dilated vicinity set 𝕍 that gates Eq. (48)) -- mirrored here.
    fs, fsm, n, _, lMin = detectFreeSurface(p, config, schemeConfig, surfaceConfig, adjacency, returnNormals=True)
    surfaceIndicator = fsm > 0.5

    R_i = p.supports
    achievedDx_i = torch.pow(p.masses / rho0, 1.0 / DIM)
    betaInterior = (R_i / achievedDx_i) ** 3.0
    michelDFS, michelNTilde = computeNearestSurfaceNormalWarp(
        p,
        operationProperties=OperationProperties(operation=WarpOperation.Density, kernel=KERNEL, supportMode=SupportScheme.Gather),
        domain=domain, adjacency=adjacency,
        freeSurfaceMask=fs.to(DTYPE), normals=n,
    )
    # Eq. (48)'s beta decay: 1 at d^FS=0 (on the surface) to (R/dx)^3 at
    # d^FS=R -- computed before the interior law, since both branches of its
    # norm clamp depend on beta (PST_ALE_PLAN.md §2.3).
    decay = torch.clamp(michelDFS / R_i, 0.0, 1.0)
    beta = 1.0 + (betaInterior - 1.0) * decay

    _, _, delta_u = computeMichelShift(p, config, schemeConfig, domain, adjacency,
                                        beta=beta, dt=config.dt, iters=1, returnVelocity=True)

    # Eq. (48)'s projection (`wrapper.py`'s `ShiftingProjectionScheme.michel2022`
    # branch), applied to the velocity form -- see module docstring for why
    # that is legitimate.
    sigma = torch.clamp((michelDFS - R_i) / (0.5 * R_i - R_i), 0.0, 1.0)
    outward = torch.einsum('ij,ij->i', delta_u, michelNTilde)
    projected = delta_u - (sigma * outward).view(-1, 1) * michelNTilde
    lambdaGate = (lMin >= 0.4).to(delta_u.dtype)
    result = (lambdaGate * lMin.pow(2.0)).view(-1, 1) * projected
    delta_u_fs = torch.where(surfaceIndicator.view(-1, 1), result, delta_u)

    mags = torch.linalg.norm(delta_u_fs, dim=-1)
    globalMax = mags.max().item()
    nSurface = int(surfaceIndicator.sum().item())
    surfaceMax = mags[surfaceIndicator].max().item() if nSurface > 0 else float('nan')
    return globalMax, surfaceMax, nSurface


def _fit_log_slope(dx_values, y_values) -> float:
    dx_values, y_values = np.asarray(dx_values), np.asarray(y_values)
    ok = np.isfinite(y_values) & (y_values > 0)
    if ok.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(dx_values[ok]), np.log(y_values[ok]), 1)[0])


def main():
    wp.init()

    print(f"R/Δx = {args.rOverDx} (fixed), jitter = {args.jitterFraction}*Δx")
    print(f"{'nx':>5} {'dx':>10} {'n':>7} {'nSurface':>8} {'global delta_u_max':>19} {'surface-set delta_u_max':>24}")
    print('-' * 90)

    dxs, globalVals, surfaceVals = [], [], []
    for nx in args.nx:
        case = _build_case(nx)
        dx = case[3]
        g, s, nSurface = _michel_free_surface_velocity(*case)
        dxs.append(dx)
        globalVals.append(g)
        surfaceVals.append(s)
        print(f"{nx:5d} {dx:10.5f} {case[1].positions.shape[0]:7d} {nSurface:8d} {g:19.6e} {s:24.6e}")

    globalSlope = _fit_log_slope(dxs, globalVals)
    surfaceSlope = _fit_log_slope(dxs, surfaceVals)

    print()
    print("log-log slope (convergence order, d(log delta_u_max)/d(log dx)):")
    print(f"  global max (all particles):        {globalSlope:.3f}  (expected ~1 -- reproduces the periodic/bulk result)")
    print(f"  surface-set max (V, Eq. 48's Table 3 claim): {surfaceSlope:.3f}  (expected >= 1 -- Eq. 48's")
    print(f"    projection only ever shrinks the interior shift near the surface, so it converges at")
    print(f"    least as fast as the interior law, possibly faster)")


if __name__ == "__main__":
    main()
