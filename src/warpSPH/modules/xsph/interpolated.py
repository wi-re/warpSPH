"""Legacy interpolation-form velocity filters with explicit unit conventions.

The Band path uses a velocity increment; the reference path uses a negated,
support-scaled acceleration. The latter is a preserved local experiment,
not an interchangeable implementation of the ordinary XSPH impulse.
"""

import torch
from warpSPHCore import OperationProperties, WarpOperation, SupportScheme, warpOperation


def _difference(state, config, adjacency, fluidMask, fluidCoefficient, boundaryCoefficient):
    coefficients = torch.where(
        fluidMask,
        torch.full_like(state.densities, fluidCoefficient),
        torch.full_like(state.densities, boundaryCoefficient),
    )
    properties = OperationProperties(
        kernel=config.kernel,
        operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric,
    )
    cv = warpOperation(
        state,
        properties,
        domain=config.domain,
        referenceValues=state.velocities * coefficients.unsqueeze(-1),
        adjacency=adjacency,
    )
    cc = warpOperation(
        state, properties, domain=config.domain, referenceValues=coefficients, adjacency=adjacency
    )
    return cv - state.velocities * cc.unsqueeze(-1)


def interpolatedVelocityFilter(
    state, config, adjacency, fluidMask, *, fluidCoefficient, boundaryCoefficient
):
    increment = _difference(
        state, config, adjacency, fluidMask, fluidCoefficient, boundaryCoefficient
    )
    return torch.where(fluidMask.unsqueeze(-1), increment, torch.zeros_like(increment))


def referenceFilterAcceleration(
    state, config, adjacency, fluidMask, *, fluidCoefficient, boundaryCoefficient
):
    increment = _difference(
        state, config, adjacency, fluidMask, fluidCoefficient, boundaryCoefficient
    )
    acceleration = -increment / state.supports.unsqueeze(-1)
    return torch.where(fluidMask.unsqueeze(-1), acceleration, torch.zeros_like(acceleration))
