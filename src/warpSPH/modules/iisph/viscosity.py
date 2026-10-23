"""Configured IISPH predictor viscosity, with free-slip static walls."""

from copy import copy


def iisphViscosity(state, config, schemeConfig, adjacency):
    """Configured two-sided fluid viscosity with free-slip static walls.

    The shared viscosity divides its kinematic coefficient by density. A
    normalized shallow view retains physical-unit invariance and leaves the
    physical masses/density used by gravity, forcing and pressure untouched.
    """
    from warpSPHCore import OperationDirection
    from ..deltaSPH import computeVelocityDiffusion
    view = copy(state)
    view.masses = state.masses / schemeConfig.fluid.restDensity
    view.densities = state.densities / schemeConfig.fluid.restDensity
    return computeVelocityDiffusion(view, config, schemeConfig, adjacency,
        approachOnly=False, wall=False, operationMode=OperationDirection.FluidToFluid)
