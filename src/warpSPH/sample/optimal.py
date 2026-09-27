"""Adaptive/relaxed particle sampler: starts from a regular lattice
(`sampleRegularParticles`) with a small Gaussian jitter and repeatedly
relaxes the positions with the raw delta-SPH shift term
(`computeDeltaShiftWarp`) scaled by the historical `modules/shifting/
delta.py` factor `-CFL * Ma * 2 h^2` (CFL 0.3, Ma 0.1 = delta.py's
no-velocity fallback, h = support/kernelScale), mimicking a "glass"
(pre-relaxed) configuration. Used by `sample.bySamplingScheme.sampleParticles`
for `SamplingScheme.optimal`, itself currently unreached by any case (see
that module's docstring).

Validated reference recipe (warpSPHCore replication, findings log 2026-09-18;
the glass of Dehnen & Aly 2012 Fig. 3, cached at
warpSPHCore/.tmp/glass_N4096_L1.0_seed42.npz): 16^3 lattice + 0.1*dx
Gaussian jitter (seed 42), Wendland C^2 at N_H = 50, 1000 iters -> density
std 0.30 %, nn/dx 0.86, no clumping. The raw shift kernel is unscaled by
design (its CFL/computeMach/c_max/dx arguments do not affect the output),
so the scaling is applied here, exactly as `delta.computeDeltaShift` does.
"""

from .regular import sampleRegularParticles
from ..geometry import ParticleSet
import torch
from warpSPHCore import *
from .wp_deltaShift import computeDeltaShiftWarp

__all__ = ['sampleOptimal']

# delta.py's historical delta^+ scaling, -CFL * Ma * 2 h^2, with Ma fixed at
# its no-velocity fallback (a sampler has no velocities to estimate one
# from). CFL 0.3 / Ma 0.1 are the validated glass-recipe values.
_DELTA_CFL = 0.3
_DELTA_MA = 0.1


def _as_state(particles: ParticleSet) -> ParticleState:
    """ParticleSet -> ParticleState (kinds = 0): `warpOperation` and
    `computeDeltaShiftWarp` index `kinds` unconditionally, so the set built
    by `sampleRegularParticles` cannot be passed to them directly."""
    return ParticleState(
        positions=particles.positions,
        supports=particles.supports,
        masses=particles.masses,
        kinds=torch.zeros(particles.positions.shape[0], dtype=torch.int32,
                          device=particles.positions.device),
        densities=particles.densities,
    )


def _wrapPeriodic(pos, domain):
    """Wrap positions into [min, max) on the periodic axes (identity on
    non-periodic ones); the kernel math is minimum-image, so the anchor is
    physically irrelevant but keeping particles inside the box keeps the
    hash search well behaved."""
    L = domain.max - domain.min
    wrapped = domain.min + (pos - domain.min) % L
    return torch.where(domain.periodic, wrapped, pos)


def sampleOptimal(nx, domain, targetNeighbors, kernel, jitter = 0.1, shiftIters = 128, shiftScheme = 'Delta', seed = None):
    dim = domain.dim
    device = domain.min.device

    particles = sampleRegularParticles(
                nx = nx,
                targetNeighbors=targetNeighbors,
                domain=domain,
            )

    particleDx = particles.masses.pow(1/dim).mean().item()

    # Gaussian jitter of the lattice start (the relaxation needs a non-
    # crystalline start to settle into a glass); `seed` makes it
    # reproducible (the validated recipe uses seed 42).
    if seed is None:
        noise = torch.randn_like(particles.positions)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(particles.positions.shape, device=device,
                            dtype=particles.positions.dtype, generator=gen)
    particles = particles._replace(
        positions = _wrapPeriodic(particles.positions + jitter * noise * particleDx, domain)
    )

    # h = support / kernelScale (the paper's h = 2 sigma); the delta^+ scale
    # is -CFL * Ma * 2 h^2 per particle (delta.py's historical factor)
    kernelScale = float(sphKernelScale(kernel.value, dim))

    op = OperationProperties(
        operation=WarpOperation.Density,
        kernel = kernel,
        supportMode = SupportScheme.Gather
    )

    for i in range(shiftIters):
        adjacency = radiusSearchCompactHashMap(
            particles, domain,
            mode = SupportScheme.SuperSymmetric,
            hashMapLengthMode = HashMapLengthMode.Fixed, fixedHashMapLength = 4096
        )

        # density first: the shift's mean-density weight uses rho_i/rho_j
        particles = particles._replace(densities = warpOperation(
            _as_state(particles),
            operationProperties = op,
            domain = domain,
            adjacency = adjacency
        ))

        h = particles.supports / kernelScale
        raw = computeDeltaShiftWarp(
            _as_state(particles),
            operationProperties = op,
            domain = domain,
            CFL = _DELTA_CFL, computeMach = False, c_max = _DELTA_MA,
            rho0 = 1.0, dx = particleDx,
            adjacency = adjacency,
        )
        scale = -_DELTA_CFL * _DELTA_MA * 2.0 * h ** 2
        particles = particles._replace(
            positions = _wrapPeriodic(particles.positions + scale.unsqueeze(-1) * raw, domain)
        )
    return particles
