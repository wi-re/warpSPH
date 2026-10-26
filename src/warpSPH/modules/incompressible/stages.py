"""Shared projection stages with force and boundary choices stated by callers."""

import torch
from ..boundaryConditions import computeForcing, enforceUpdates
from ..gravity import computeGravity, computeBodyForce
from ..deltaSPH import computeVelocityDiffusion


def nonPressureAcceleration(
    system,
    dt,
    config,
    schemeConfig,
    adjacency,
    *,
    includeBodyForce,
    viscosity,
    boundaryVelocitiesPrepared=True,
    maskViscosity=False,
    extraAcceleration=None,
):
    """Preserve gravity/body/viscosity/extra/forcing accumulation order.

    Band historically omits body force and viscosity; omni opts into
    viscosity; reference DFSPH always invokes its configured diffusion.
    These choices are explicit rather than inferred from solver names.
    """
    state = system.state
    acceleration = computeGravity(state, config, schemeConfig, adjacency)
    if includeBodyForce:
        acceleration = acceleration + computeBodyForce(state, schemeConfig)
    if viscosity:
        viscous = computeVelocityDiffusion(state, config, schemeConfig, adjacency,
            boundaryVelocitiesPrepared=boundaryVelocitiesPrepared)
        if maskViscosity:
            viscous = torch.where(
                (state.kinds == 0).unsqueeze(-1), viscous, torch.zeros_like(viscous)
            )
        acceleration = acceleration + viscous
    if extraAcceleration is not None:
        acceleration = acceleration + extraAcceleration()
    forcing = computeForcing(system, dt, system.t, config, schemeConfig)
    return acceleration + forcing / state.masses.view(-1, 1)


def projectedUpdate(system, dt, config, schemeConfig):
    """Compatibility update for an already advanced map, with the usual BCs.

    ProjectedStep consumes the stage state exactly once. Direct legacy
    single-evaluation callers still receive the historical zero update.
    """
    from ...systems.incompressible import IncompressibleSystemUpdate

    state = system.state
    zeros = torch.zeros_like(state.velocities)
    update = IncompressibleSystemUpdate(
        dxdt=zeros.clone(),
        dvdt=zeros.clone(),
        drhodt=torch.zeros_like(state.densities),
        passive=torch.zeros_like(state.densities, dtype=torch.bool),
    )
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    return update


def carryProjectionFields(destination, source):
    """Copy the advanced map's fields without inventing warm-start semantics."""
    for name in (
        "positions",
        "velocities",
        "densities",
        "supports",
        "pressures",
        "soundspeeds",
        "surfaceIndicators",
        "surfaceNormals",
        "surfaceLambdas",
    ):
        value = getattr(source, name, None)
        if value is None:
            continue
        target = getattr(destination, name, None)
        if target is not None and torch.is_tensor(target) and target.shape == value.shape:
            target.copy_(value)
        else:
            setattr(destination, name, value.clone() if torch.is_tensor(value) else value)


def requireProjectedStep(system, config, schemeName):
    """Refuse a full-map callback under a multistage legacy ODE driver."""
    if getattr(system, "_projectedStepActive", False):
        return
    from warpSPHIntegrators.schemes import IntegrationSchemeType as I

    if getattr(config, "integrationScheme", None) in (
        I.forwardEuler,
        I.explicitEuler,
        I.semiImplicitEuler,
    ):
        return
    raise ValueError(
        f"{schemeName} advances a full projected state map; use the package ProjectedStep driver "
        "or a legacy single-evaluation Euler driver, rather than repeating it as an ODE RHS"
    )
