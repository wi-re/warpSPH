"""Physical step limits for summation-density pressure projections.

A fixed requested step remains a ceiling, not permission to exceed a physical
limit. minDt must never raise a safe step above the advection/diffusion bound.
"""

import math

from warpSPHCore import sphKernelScale


def fixedProjectionRequest(ctx):
    """Keep the initialized ceiling while allowing safe adaptive recovery.

    The current dt is the last capped step, not the requested fixed ceiling.
    Use the existing dtGrowthFactor for recovery, as the common adaptive
    timestep implementation does; the physical cap is applied afterward.
    """
    previous = float(ctx.config.dt)
    nominal = ctx.scratch.setdefault('projectionNominalDt', previous)
    if not ctx.config.adaptiveDt:
        return min(nominal, previous)
    return min(nominal, previous * float(ctx.config.dtGrowthFactor))


def projectionTimestep(ctx, system, requested=None):
    state, config, sc = system.state, ctx.config, ctx.schemeConfig
    fluid = state.kinds == 0
    h = float(state.supports[fluid].min())
    vmax = float(state.velocities[fluid].norm(dim=-1).max())
    scale = float(sphKernelScale(config.kernel.value, config.dim))
    advective = float(config.cflFactor) * float(config.dx) / max(vmax, 1e-3)
    dp = sc.diffusionParams
    nu = float(dp.viscidNu) if not dp.inviscid else 0.0
    viscous = (0.125 * (h / scale) ** 2 / nu
               if nu > 0 and getattr(sc, 'dt_viscosityConstraint', True) else math.inf)
    limit = min(advective, viscous, config.maxDt if config.maxDt is not None else math.inf)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"Invalid projection timestep limit: {limit}")
    if config.minDt is not None and limit < config.minDt:
        raise ValueError(f"Projection timestep limit {limit:g} is below minDt={config.minDt:g}")
    chosen = min(float(requested), limit) if requested is not None else limit
    ctx.scratch['projectionTimestepLimits'] = dict(advective=advective, viscous=viscous,
                                                  maximum=limit, chosen=chosen)
    return chosen


def limitProjectionCase(case):
    """Retain a case's timestep choice, capped by physical projection limits."""
    from dataclasses import replace

    original = case.timestep

    def timestep(ctx, system):
        requested = original(ctx, system) if original else fixedProjectionRequest(ctx)
        if (float(system.t) == 0.0 or not ctx.config.adaptiveDt) and ctx.config.dt is not None:
            requested = min(float(requested), float(ctx.config.dt))
        return projectionTimestep(ctx, system, requested)

    return replace(case, timestep=timestep)
