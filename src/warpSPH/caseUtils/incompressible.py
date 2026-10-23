"""Lattice initialization shared by the incompressible periodic cases."""

from __future__ import annotations

from warpSPH.configurations.compute import neighborSearchOptions

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

    Runs `steps` pressure solves on a scratch state with the
    velocities zeroed, displacing positions by `dt**2 * a_p` each time, then
    copies the relaxed positions back onto `system.state`. A no-op when
    `steps` is 0. Native IISPH calibrates uniform masses first and uses its
    own pressure operator; the composed schemes retain their shifting solve.

    Not for free-surface states: their legitimate kernel density deficit
    would be treated as a bulk density error and compact the surface.
    Jitter only (`shuffleParticles` with `shiftIters=0`) is the safe setup
    for those. The guard rejects a minimum density below `0.9*rho0`.
    """
    if not steps:
        return

    from warpSPHCore import SupportScheme, buildVerletList

    from ..modules import computeDensities, shuffleParticles, solveIncompressible
    from ..enumTypes import IncompressibleSPHScheme

    nativeIISPH = ctx.scheme == IncompressibleSPHScheme.iisph
    if nativeIISPH:
        from .projection import initializeProjectionDensity
        from ..modules.iisph import computeIISPHContactPoints, solveIISPH
        initializeProjectionDensity(ctx, system)

    state = system.initializeNewState()
    # `divergenceFree_step` allocates the pressure array on the first step it runs
    # (see `schemes/incompressible/divergenceFree.py`); a region-built system has not stepped yet, so
    # give its scratch state one the same way, otherwise the `pressures[:] = 0.0`
    # below has nothing to write into.
    if state.state.pressures is None:
        state.state.pressures = torch.zeros_like(state.state.densities)
    state.state.positions = shuffleParticles(state.state, ctx.config, ctx.schemeConfig, 0,
                                             jitterAmount=jitter)
    state.state.velocities = torch.zeros_like(state.state.velocities)

    adjacency = buildVerletList(
        state.state,
        ctx.config.domain,
        verletScale=1.4,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=None,
        verbose=False,
        **neighborSearchOptions(ctx.config))
    state.state.densities = computeDensities(state.state, ctx.config, ctx.schemeConfig,
                                             adjacency)
    rho0 = ctx.schemeConfig.fluid.restDensity
    minDensity = state.state.densities.min().item()
    if minDensity < 0.9 * rho0:
        raise ValueError(
            f'relaxLattice: the jittered state has a free surface '
            f'(min density {minDensity:.4g} < 0.9 * rho0 = {0.9 * rho0:.4g}); the '
            f'pressure relaxation would compact its free surface. '
            f'Jitter only instead (shuffleParticles with shiftIters=0).')

    # `progress` is tri-state (None = auto), so resolve it the same way the
    # runner's own loop does rather than treating None as false.
    showProgress = ctx.spec.progress
    if showProgress is None:
        showProgress = sys.stderr.isatty()
    for _ in _maybeProgress(range(steps), showProgress and not ctx.spec.quiet, 'relaxing'):
        adjacency = buildVerletList(
            state.state,
            ctx.config.domain,
            verletScale=1.4,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=adjacency,
            verbose=False,
            **neighborSearchOptions(ctx.config))
        state.state.densities = computeDensities(state.state, ctx.config, ctx.schemeConfig, adjacency)
        state.state.pressures[:] = 0.0
        if nativeIISPH:
            contactPoints, coverageFailure = computeIISPHContactPoints(state.state, state.contactBoundaryProvider)
            pressure = solveIISPH(state.state, ctx.config, ctx.schemeConfig,
                torch.zeros_like(state.state.velocities), dt, adjacency,
                analyticProvider=state.analyticBoundaryProvider, boundarySupport=state.boundarySupport,
                contactPoints=contactPoints, contactCoverageFailure=coverageFailure)
            accel = pressure.acceleration
        else:
            accel, _, _, _ = solveIncompressible(
                particles=state.state, config=ctx.config, schemeConfig=ctx.schemeConfig,
                adjacency=adjacency,
                dvdt=torch.zeros_like(state.state.velocities), dt=dt, verbose=False)
        state.state.positions = state.state.positions + dt * dt * accel

    system.state.positions = state.state.positions.clone()
