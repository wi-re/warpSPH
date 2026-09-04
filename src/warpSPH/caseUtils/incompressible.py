"""What the `divergenceFree` cases share.

Currently one thing: the lattice pre-relaxation every one of them wants before
its initial condition is imposed. A perfectly regular lattice is an unstable
equilibrium for SPH -- it is also the configuration where the summation density
is *lowest* (`DFSPH_IMPROVEMENT_PLAN.md` §1.1), so a run started on one spends
its first steps resolving lattice noise rather than the physics under test.
Jittering and then pressure-relaxing removes that transient.

Extracted verbatim from `cases/tgv.py`, which had the only copy, when
`shearWave` turned out to need exactly the same thing. `tgv`'s numbers are
asserted by `tests/test_physics.py` and are unchanged by the move.
"""

from __future__ import annotations

__all__ = ['relaxLattice']

import sys

import torch


def _maybeProgress(iterable, enabled, description):
    if not enabled:
        return iterable
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, leave=False)


def relaxLattice(ctx, system, steps: int, dt: float, jitter: float) -> None:
    """Jitter the sampled lattice and pressure-relax it in place.

    Runs `steps` constant-density (shifting) solves on a scratch state with the
    velocities zeroed, displacing positions by `dt**2 * a_p` each time, then
    copies the relaxed positions back onto `system.state`. A no-op when
    `steps` is 0.

    Not for free-surface states: the solve's source is `rho0 - rhoStar` with
    `rhoStar` clamped at `0.9`, so a surface layer (which legitimately reads
    ~0.5) is driven toward that floor every step. On a surface state the
    shifts are O(10%) of the domain per step and the blob collapses within a
    few steps (`DFSPH_IMPROVEMENT_PLAN.md` Part 23) -- the guard below fails
    loudly instead. Jitter only (`shuffleParticles` with `shiftIters=0`) is
    the safe setup for those.
    """
    if not steps:
        return

    from warpSPHCore import SupportScheme, buildVerletList

    from ..modules import computeDensities, shuffleParticles, solveIncompressible

    state = system.initializeNewState()
    # `divergenceFree_step` allocates the pressure array on the first step it runs
    # (see `schemes/divergenceFree.py`); a region-built system has not stepped yet, so
    # give its scratch state one the same way, otherwise the `pressures[:] = 0.0`
    # below has nothing to write into.
    if state.state.pressures is None:
        state.state.pressures = torch.zeros_like(state.state.densities)
    state.state.positions = shuffleParticles(state.state, ctx.config, ctx.schemeConfig, 0,
                                             jitterAmount=jitter)
    state.state.velocities = torch.zeros_like(state.state.velocities)

    adjacency = buildVerletList(state.state, ctx.config.domain, verletScale=1.4,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=None, verbose=False)
    state.state.densities = computeDensities(state.state, ctx.config, ctx.schemeConfig,
                                             adjacency)
    rho0 = ctx.schemeConfig.fluid.restDensity
    minDensity = state.state.densities.min().item()
    if minDensity < 0.9 * rho0:
        raise ValueError(
            f'relaxLattice: the jittered state has a free surface '
            f'(min density {minDensity:.4g} < 0.9 * rho0 = {0.9 * rho0:.4g}); the '
            f'constant-density solve would compact it toward the clamp floor. '
            f'Jitter only instead (shuffleParticles with shiftIters=0).')

    # `progress` is tri-state (None = auto), so resolve it the same way the
    # runner's own loop does rather than treating None as false.
    showProgress = ctx.spec.progress
    if showProgress is None:
        showProgress = sys.stderr.isatty()
    for _ in _maybeProgress(range(steps), showProgress and not ctx.spec.quiet, 'relaxing'):
        adjacency = buildVerletList(state.state, ctx.config.domain, verletScale=1.4,
                                    supportMode=SupportScheme.SuperSymmetric,
                                    priorNeighborhood=adjacency, verbose=False)
        state.state.densities = computeDensities(state.state, ctx.config, ctx.schemeConfig, adjacency)
        state.state.pressures[:] = 0.0
        accel, _, _, _ = solveIncompressible(
            particles=state.state, config=ctx.config, schemeConfig=ctx.schemeConfig,
            adjacency=adjacency,
            dvdt=torch.zeros_like(state.state.velocities), dt=dt, verbose=False)
        state.state.positions = state.state.positions + dt * dt * accel

    system.state.positions = state.state.positions.clone()
