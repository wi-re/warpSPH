"""Post-step hook: refresh the stored boundary-particle (`kind == 1`) densities
with a fresh mDBC extrapolation at the *post-step* configuration.

`deltaSPH_step` calls `computeMdbcDensity` near the *start* of a step (on the
pre-move positions) and then the RK integrator advances every particle,
including the wall band, by its continuity `drho/dt`. So the density left on the
boundary particles at the end of a step -- the value the density plots, the
field video and any per-particle density diagnostic read -- is not the mDBC
wall density the scheme actually used for the pressure force; it has drifted.

For validation runs where the wall density field is being *looked at*
(`DELTASPH_VALIDATION_PLAN.md` 5.2.3), re-running the extrapolation in `postStep`
puts the honest value back before anything plots it. Fluid densities are
untouched. Not wired into any production step -- opt-in from a probe script.
"""
from __future__ import annotations


def installMdbcDensityToState(case):
    """Chain an mDBC boundary-density recompute onto ``case.postStep``.

    Idempotent; wraps any existing hook. No-op for incompressible schemes and
    for states with no boundary particles.
    """
    if getattr(case, '_mdbcDensityToState', False):
        return case

    from warpSPH.modules.mdbc import computeMdbcDensity, computeMdbcDensityBand, computeMdbcDensityEnglish2025
    from warpSPH.enumTypes import isIncompressibleScheme

    prev = case.postStep

    def _hook(ctx, state, step):
        if prev is not None:
            prev(ctx, state, step)
        particles = getattr(state, 'state', state)
        kinds = getattr(particles, 'kinds', None)
        if kinds is None or not bool((kinds == 1).any()):
            return
        try:
            if isIncompressibleScheme(ctx.scheme):
                return
        except Exception:  # noqa: BLE001 - scheme not in the WC enum -> just try
            pass
        adjacency = getattr(state, 'adjacency', None)
        # Must match whichever extrapolation `deltaSPH_step` actually used
        # this run (`schemeConfig.mdbcDensityScheme`) -- otherwise this
        # "honest post-step value" hook would silently overwrite an A/B
        # scheme's boundary density back to the other scheme's value before
        # any video/pressure-probe diagnostic reads it, making an A/B test
        # against this hook meaningless.
        _scheme = getattr(ctx.schemeConfig, 'mdbcDensityScheme', 'ramped')
        if _scheme == 'band':
            particles.densities = computeMdbcDensityBand(
                particles, ctx.config, ctx.schemeConfig, adjacency)
        elif _scheme == 'english2025':
            particles.densities = computeMdbcDensityEnglish2025(
                particles, ctx.config, ctx.schemeConfig, adjacency)
        else:
            particles.densities = computeMdbcDensity(
                particles, ctx.config, ctx.schemeConfig, adjacency)

    case.postStep = _hook
    case._mdbcDensityToState = True
    return case
