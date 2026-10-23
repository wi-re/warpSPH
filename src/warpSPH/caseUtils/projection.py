"""Initialize the summation field that the first pressure solve will read.

Free-surface kernel deficiency remains a physical boundary property; it is
not filled by increasing particle masses or clipping the measured density.
"""

import torch

from ..modules.density import computeDensities


def initializeHydrostaticPressure(ctx, system):
    """Seed the at-rest profile in the gauge consumed by the selected route.

    Raw nonnegative pressure is needed by the free-surface density projections.
    Only the explicitly selected legacy VD+PS solver uses a mean-shifted seed.
    Fluid and sampled wall rows share the same profile; ghosts are not sources.
    """
    from ..enumTypes import IncompressibleSPHScheme
    state, sc = system.state, ctx.schemeConfig
    fluid = state.kinds == 0
    surface = state.positions[fluid, 1].max()
    gravity = float(sc.gravityConfig.magnitude) if sc.gravityConfig.active else 0.0
    pressure = sc.fluid.restDensity*gravity*(surface-state.positions[:, 1]).clamp_min(0)
    gauge = 'free-surface-zero'
    if ctx.scheme == IncompressibleSPHScheme.divergenceFree:
        from ..schemes.incompressible import divergenceFree
        if divergenceFree.DIVERGENCE_SOLVER == 'vdps':
            pressure = pressure-pressure[fluid].mean()
            gauge = 'legacy-vdps-mean-zero'
    state.pressures = torch.where(state.kinds != 2, pressure, torch.zeros_like(pressure))
    ctx.scratch['initialPressureCalibration'] = dict(gauge=gauge, surface=float(surface),
        fluid_min=float(pressure[fluid].min()), fluid_max=float(pressure[fluid].max()))


def initializeProjectionDensity(ctx, system):
    state = system.state
    from ..enumTypes import IncompressibleSPHScheme
    if ctx.scheme == IncompressibleSPHScheme.iisph:
        # Report the exact density the native pressure solve consumes,
        # including physical rho0 times the dimensionless solid integral.
        from ..modules.incompressible.operators import rebuildAdjacency
        from ..modules.gravity import computeGravity
        from ..modules.iisph import computeIISPHBoundaryTerms, computeIISPHCoefficients
        config, schemeConfig = ctx.config, ctx.schemeConfig
        adjacency = rebuildAdjacency(state, system, config)
        gravity = computeGravity(state, config, schemeConfig, adjacency)
        boundary = computeIISPHBoundaryTerms(state, config, schemeConfig, gravity)
        coefficients = computeIISPHCoefficients(state, config, schemeConfig, adjacency, boundary)
        state.densities = coefficients.densityRatio * schemeConfig.fluid.restDensity
    else:
        state.densities = computeDensities(state, ctx.config, ctx.schemeConfig, None)
    rho = state.densities[state.kinds == 0] / ctx.schemeConfig.fluid.restDensity
    report = dict(normalized=bool(ctx.config.calibrateNormalization),
                  n_h=float(ctx.spec.n_h), minimum=float(rho.min()),
                  median=float(rho.median()), maximum=float(rho.max()),
                  p99=float(torch.quantile(rho.float(), 0.99)))
    ctx.scratch['initialDensityCalibration'] = report
    return report
