"""Pressure right-hand sides with explicit units and arithmetic conventions.

Sources are evaluated once per solve at fixed geometry. Row restrictions,
compatibility projections and deficiency guards belong to the chosen solver.
The reference density-rate form deliberately preserves its operation order;
algebraically rewriting it as dt*div can change roundoff and stopping iterates.
"""

import torch
from warpSPHCore import OperationProperties, SupportScheme, WarpOperation, warpOperation

__all__ = [
    "omniSource",
    "referenceSource",
    "pressureBoundarySource",
    "densityInvarianceSource",
    "velocityDivergenceSource",
]


def omniSource(
    state, config, adjacency, *, rho0, dt, divergence, mode, surfaceSource="full", wall=None
):
    """Dimensionless RHS; operator is -dt**2 div(a_p) in both modes."""
    if mode == "divergence":
        return dt * divergence
    if mode != "density":
        raise ValueError(f"Unknown pressure mode: {mode!r}")
    densityError = 1.0 - state.densities / rho0
    if surfaceSource == "clamp":
        densityError = densityError.clamp(max=0.0)
    elif surfaceSource == "mask":
        surf = getattr(state, "surfaceIndicators", None)
        if surf is not None:
            densityError = torch.where(
                surf.to(torch.bool), torch.zeros_like(densityError), densityError
            )
    elif surfaceSource == "shepard":
        props = OperationProperties(
            kernel=config.kernel,
            operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.SuperSymmetric,
        )
        weight = warpOperation(
            state,
            props,
            domain=config.domain,
            referenceValues=torch.ones_like(densityError),
            adjacency=adjacency,
        )
        if wall is not None:
            from ..analyticBoundary import wallDensity

            weight = weight + wallDensity(wall)
        rhoShep = state.densities / weight.clamp_min(1e-6)
        densityError = 1.0 - rhoShep / rho0
    elif surfaceSource != "full":
        raise ValueError(f"Unknown surfaceSource: {surfaceSource!r}")
    return densityError + dt * divergence


def referenceSource(densities, densityRate, *, rho0, dt, mode):
    """CD: dimensionless, opDt=dt**2. DF: 1/time, opDt=dt."""
    if mode == "density":
        return 1.0 - densities / rho0 - dt * densityRate / rho0
    if mode == "divergence":
        return -densityRate / rho0
    raise ValueError(f"Unknown pressure mode: {mode!r}")


def pressureBoundarySource(restVolumes, volumes, *, dt, divergence):
    """Dimensionless volume-constraint RHS on fluid AND boundary unknowns."""
    return (1.0 - restVolumes / volumes.clamp_min(1e-12)) + dt * divergence


def densityInvarianceSource(densities, densityRate, *, rho0, dt):
    """Legacy VD+PS CD convention: form predicted rho before normalizing.

    This grouping differs from referenceSource and is retained for rounding
    and iteration-history compatibility. The matching operator has dt**2.
    """
    predictedDensity = densities + dt * densityRate
    return 1 - predictedDensity / rho0


def velocityDivergenceSource(densityRate):
    """Legacy VD+PS DF convention: -R(v*), without reference's 1/rho0.

    The matching pressure-shift operator is scaled by dt. This convention
    must stay paired with its existing diagonal and convergence units.
    """
    return -densityRate
